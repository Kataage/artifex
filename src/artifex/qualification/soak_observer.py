from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import socket
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import make_url

from artifex.config.models import ArtifexSettings
from artifex.render_node import fetch_render_attestation

_MAX_BYTES = 12 * 1024 * 1024
_MAX_SAMPLES = 5000
_REQUIRED = frozenset(
    {"production_checkpoint", "refiner_checkpoint", "vae", "upscale_model"}
)


class SoakHeader(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["start"] = "start"
    schema_version: Literal[1] = 1
    run_id: str
    controller_host: str
    renderer_id: str
    started_at: datetime
    target_seconds: float = Field(gt=0, le=259200)
    interval_seconds: float = Field(ge=10, le=3600)
    config_sha256: str = Field(min_length=64, max_length=64)


class SoakSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["sample"] = "sample"
    observed_at: datetime
    elapsed_seconds: float = Field(ge=0)
    llm_ok: bool
    comfyui_ok: bool
    renderer_ok: bool
    gpu_used_mib: float | None = Field(default=None, ge=0)
    ram_used_mib: float | None = Field(default=None, ge=0)
    free_disk_gib: float | None = Field(default=None, ge=0)
    telemetry_rows: int | None = Field(default=None, ge=0)
    severe_events: int | None = Field(default=None, ge=0)
    finalized_packs: int | None = Field(default=None, ge=0)
    asset_sha256: dict[str, str] = Field(default_factory=dict)
    errors: tuple[str, ...] = ()


class SoakEnd(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["end"] = "end"
    finished_at: datetime
    elapsed_seconds: float = Field(ge=0)
    stopped_early: bool


class SoakAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    ready_for_soak_review: bool
    production_qualified: Literal[False] = False
    elapsed_hours: float
    observed_samples: int
    verified_completed_packs: int
    observed_telemetry_rows: int
    fatal_errors: int
    peak_vram_mib: float | None
    peak_ram_mib: float | None
    minimum_free_disk_gib: float | None
    evidence_sha256: str
    issues: tuple[str, ...]


def _physical_ram_mib() -> float | None:
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            kernel = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
            if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return (status.total_phys - status.avail_phys) / 1048576
        except (OSError, AttributeError):
            return None
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, _, remainder = line.partition(":")
            if name in {"MemTotal", "MemAvailable"}:
                values[name] = int(remainder.strip().split()[0])
        if len(values) == 2:
            return (values["MemTotal"] - values["MemAvailable"]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _existing_directory(path: Path) -> Path:
    current = path.expanduser().resolve(strict=False)
    while not current.exists() and current != current.parent:
        current = current.parent
    if not current.is_dir():
        raise ValueError("Cannot inspect free space of a non-directory")
    return current


def _read_db_counts(database_url: str) -> tuple[int, int, int]:
    parsed = make_url(database_url)
    if parsed.get_backend_name() != "sqlite" or not parsed.database:
        raise ValueError("Soak telemetry requires a local SQLite database")
    database = Path(parsed.database).expanduser().resolve(strict=True)
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        events = connection.execute("SELECT COUNT(*) FROM agent_events").fetchone()
        severe = connection.execute(
            "SELECT COUNT(*) FROM agent_events "
            "WHERE severity IN ('error', 'critical')"
        ).fetchone()
        packs = connection.execute(
            "SELECT COUNT(*) FROM packs WHERE state = 'finalized'"
        ).fetchone()
        if events is None or severe is None or packs is None:
            raise ValueError("Soak database aggregate queries returned no rows")
        return int(events[0]), int(severe[0]), int(packs[0])
    finally:
        connection.close()


def _gpu_usage_mib(stats: dict[str, Any]) -> float | None:
    devices = stats.get("devices")
    if not isinstance(devices, list) or not devices:
        return None
    total_used = 0
    for item in devices:
        if not isinstance(item, dict):
            return None
        total, free = item.get("vram_total"), item.get("vram_free")
        if (
            type(total) is not int or type(free) is not int
            or not 0 <= free <= total
        ):
            return None
        total_used += total - free
    return total_used / 1048576


def sample_soak(settings: ArtifexSettings, elapsed_seconds: float) -> SoakSample:
    """Perform bounded, read-only real controller/renderer/DB measurements."""
    errors: list[str] = []
    primary = settings.render_nodes.primary_node()
    llm_ok = False
    comfyui_ok = False
    renderer_ok = False
    gpu: float | None = None
    ram = _physical_ram_mib()
    if ram is None:
        errors.append("PC-A physical RAM measurement unavailable")
    try:
        disk = shutil.disk_usage(_existing_directory(settings.storage.packs_dir))
        free_gib: float | None = disk.free / (1024**3)
    except (ValueError, OSError) as exc:
        free_gib = None
        errors.append(f"PC-A free disk check: {type(exc).__name__}")
    try:
        telemetry, severe, packs = _read_db_counts(settings.storage.database_url)
    except (OSError, ValueError, sqlite3.Error) as exc:
        telemetry = severe = packs = None
        errors.append(f"SQLite telemetry/Pack counters: {type(exc).__name__}")
    if primary is None:
        errors.append("No primary renderer configured")
    with httpx.Client(
        timeout=httpx.Timeout(connect=5, read=20, write=10, pool=5),
        follow_redirects=False, trust_env=False,
    ) as client:
        try:
            llm_headers: dict[str, str] = {}
            if settings.llm.api_key_env:
                token = os.environ.get(settings.llm.api_key_env)
                if token:
                    llm_headers["Authorization"] = f"Bearer {token}"
            url = settings.llm.base_url.rstrip("/")
            response = client.get(url + "/health", headers=llm_headers)
            if response.status_code == 404:
                response = client.get(url + "/v1/models", headers=llm_headers)
            response.raise_for_status()
            llm_ok = True
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"PC-A LLM /health: {type(exc).__name__}")
        if primary is not None:
            node_id, node = primary
            try:
                response = client.get(node.base_url.rstrip("/") + "/system_stats")
                response.raise_for_status()
                raw: Any = response.json()
                if not isinstance(raw, dict) or not isinstance(raw.get("system"), dict):
                    raise ValueError("ComfyUI system_stats shape is invalid")
                comfyui_ok = True
                gpu = _gpu_usage_mib(raw)
                if gpu is None:
                    errors.append("PC-B VRAM telemetry missing or inconsistent")
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                errors.append(f"PC-B ComfyUI /system_stats: {type(exc).__name__}")
            assets: dict[str, str] = {}
            try:
                remote = fetch_render_attestation(
                    node_id, node, client=client, timeout_seconds=60
                )
                age = (datetime.now(UTC) - remote.created_at).total_seconds()
                if not -300 <= age <= 300:
                    raise ValueError("PC-B authenticated attestation is stale")
                if not remote.nvidia_gpus or remote.inventory_errors:
                    raise ValueError("PC-B GPU or model/LoRA inventory is unhealthy")
                assets = {x.label: x.sha256 for x in remote.assets}
                if not _REQUIRED <= assets.keys():
                    raise ValueError("PC-B attestation is missing required model hashes")
                renderer_ok = True
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                errors.append(f"PC-B authenticated attestation: {type(exc).__name__}")
        else:
            assets = {}
    return SoakSample(
        observed_at=datetime.now(UTC),
        elapsed_seconds=elapsed_seconds,
        llm_ok=llm_ok, comfyui_ok=comfyui_ok, renderer_ok=renderer_ok,
        gpu_used_mib=gpu, ram_used_mib=ram, free_disk_gib=free_gib,
        telemetry_rows=telemetry, severe_events=severe, finalized_packs=packs,
        asset_sha256=assets, errors=tuple(errors),
    )


def _write_record(handle: Any, entry: BaseModel) -> None:
    handle.write(entry.model_dump_json() + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def observe_soak(
    settings: ArtifexSettings,
    *,
    output: Path,
    duration_hours: float | None = None,
    interval_seconds: float = 300,
    sample: Callable[[ArtifexSettings, float], SoakSample] = sample_soak,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> SoakAssessment:
    """Record actual elapsed time, durable per-sample failures and final state.

    A crash/CTRL+C leaves a partial file with no end record, which always
    fails verification. Does not run, pause or change the Artifex daemon.
    """
    target = (
        settings.qualification.minimum_soak_hours
        if duration_hours is None else duration_hours
    )
    if not 0 < target <= 72:
        raise ValueError("Soak duration must be between 0 and 72 hours")
    if not 10 <= interval_seconds <= 3600:
        raise ValueError("Soak sample interval must be between 10 and 3600 seconds")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Soak evidence cannot overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    primary = settings.render_nodes.primary_node()
    run_id = uuid4().hex
    header = SoakHeader(
        run_id=run_id, controller_host=socket.gethostname(),
        renderer_id=primary[0] if primary else "(unconfigured)",
        started_at=now(), target_seconds=target * 3600,
        interval_seconds=interval_seconds,
        config_sha256=hashlib.sha256(settings.model_dump_json().encode()).hexdigest(),
    )
    start = monotonic()
    stopped = False
    with output.open("x", encoding="utf-8") as handle:
        _write_record(handle, header)
        for index in range(_MAX_SAMPLES):
            elapsed = max(0.0, monotonic() - start)
            try:
                observed = sample(settings, elapsed)
            except Exception as exc:  # noqa: BLE001 - preserve monitoring failure
                observed = SoakSample(
                    observed_at=now(), elapsed_seconds=elapsed,
                    llm_ok=False, comfyui_ok=False, renderer_ok=False,
                    errors=(f"Probe exception: {type(exc).__name__}",),
                )
            _write_record(handle, observed)
            if index == 0 and (
                not observed.llm_ok or not observed.comfyui_ok
                or not observed.renderer_ok or observed.errors
            ):
                stopped = True
                break
            remaining = header.target_seconds - (monotonic() - start)
            if remaining <= 0:
                break
            sleep(min(interval_seconds, remaining))
        else:
            stopped = True
        finish = SoakEnd(
            finished_at=now(),
            elapsed_seconds=max(0.0, monotonic() - start),
            stopped_early=stopped,
        )
        _write_record(handle, finish)
    return verify_soak_evidence(
        output, minimum_hours=settings.qualification.minimum_soak_hours
    )


def verify_soak_evidence(
    path: Path, *, minimum_hours: float = 8,
) -> SoakAssessment:
    """Verify monitored span, coverage, live health, actual DB progress.

    Syntactically valid or user-provided self-report fields are not proof.
    Soak evidence is diagnostic, never automatically a qualification stage PASS.
    """
    if not 0 < minimum_hours <= 72:
        raise ValueError("Soak minimum must be between 0 and 72 hours")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError("Soak evidence must be regular JSONL <= 12 MiB, not a symlink")
    raw = path.read_bytes()
    lines = raw.splitlines()
    if len(lines) < 3 or len(lines) > _MAX_SAMPLES + 2:
        raise ValueError("Soak evidence requires start, samples and finish")
    header = SoakHeader.model_validate_json(lines[0])
    samples = tuple(SoakSample.model_validate_json(line) for line in lines[1:-1])
    tail = SoakEnd.model_validate_json(lines[-1])
    problems: list[str] = []
    if tail.stopped_early:
        problems.append("Monitor stopped early due to initial failure or sample limit")
    if tail.elapsed_seconds < minimum_hours * 3600:
        problems.append("Actual monitored duration is shorter than required minimum")
    if len(samples) < 2:
        problems.append("Insufficient observations to establish continuous monitoring")
    if abs(tail.elapsed_seconds - (tail.finished_at - header.started_at).total_seconds()) > 90:
        problems.append("Monotonic elapsed and wall-clock timestamps disagree")
    if tail.elapsed_seconds < header.target_seconds - 1:
        problems.append("Monitor stopped before its configured target duration")
    previous = 0.0
    baseline: dict[str, str] | None = None
    for index, item in enumerate(samples):
        elapsed = item.elapsed_seconds
        if elapsed < previous or (index > 0 and elapsed - previous > header.interval_seconds * 1.5 + 30):
            problems.append(f"Sample {index}: elapsed time reversed or sampling gap exceeded")
        if abs(
            (item.observed_at - header.started_at).total_seconds() - elapsed
        ) > 90:
            problems.append(f"Sample {index}: wall clock disagrees with elapsed time")
        if not (item.llm_ok and item.comfyui_ok and item.renderer_ok):
            problems.append(f"Sample {index}: LLM, ComfyUI or authenticated PC-B unavailable")
        if item.errors:
            problems.append(f"Sample {index}: " + ", ".join(item.errors[:3]))
        if None in (
            item.gpu_used_mib, item.ram_used_mib, item.free_disk_gib,
            item.telemetry_rows, item.severe_events, item.finalized_packs,
        ):
            problems.append(f"Sample {index}: resource or DB telemetry missing")
        if not _REQUIRED <= item.asset_sha256.keys():
            problems.append(f"Sample {index}: required asset hashes missing")
        if baseline is None:
            baseline = item.asset_sha256
        elif baseline != item.asset_sha256:
            problems.append(f"Sample {index}: PC-B model assets changed")
        previous = elapsed
    if samples and tail.elapsed_seconds - samples[-1].elapsed_seconds > (
        header.interval_seconds * 1.5 + 30
    ):
        problems.append("Final monitor period was not sampled")
    first = samples[0]
    last = samples[-1]
    delta_telemetry = (
        last.telemetry_rows - first.telemetry_rows
        if last.telemetry_rows is not None and first.telemetry_rows is not None
        else 0
    )
    delta_packs = (
        last.finalized_packs - first.finalized_packs
        if last.finalized_packs is not None and first.finalized_packs is not None
        else 0
    )
    delta_severe = (
        last.severe_events - first.severe_events
        if last.severe_events is not None and first.severe_events is not None
        else 0
    )
    if delta_telemetry <= 0:
        problems.append("No new persisted agent telemetry during the monitoring window")
    if delta_packs < 3:
        problems.append("Fewer than three newly finalized Packs during unattended run")
    if delta_severe != 0:
        problems.append("Error or critical agent events occurred during the monitoring window")
    gpu_values = [x.gpu_used_mib for x in samples if x.gpu_used_mib is not None]
    ram_values = [x.ram_used_mib for x in samples if x.ram_used_mib is not None]
    disk_values = [x.free_disk_gib for x in samples if x.free_disk_gib is not None]
    if not gpu_values or not ram_values or not disk_values:
        problems.append("GPU, RAM or disk usage not continuously observed")
    return SoakAssessment(
        run_id=header.run_id,
        ready_for_soak_review=not problems,
        elapsed_hours=tail.elapsed_seconds / 3600,
        observed_samples=len(samples),
        verified_completed_packs=max(delta_packs, 0),
        observed_telemetry_rows=max(delta_telemetry, 0),
        fatal_errors=max(delta_severe, 0),
        peak_vram_mib=max(gpu_values) if gpu_values else None,
        peak_ram_mib=max(ram_values) if ram_values else None,
        minimum_free_disk_gib=min(disk_values) if disk_values else None,
        evidence_sha256=hashlib.sha256(raw).hexdigest(),
        issues=tuple(dict.fromkeys(problems)),
    )
