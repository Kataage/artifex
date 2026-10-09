"""Atomic local liveness heartbeat for the independent native PC-B watcher.

The heartbeat proves only recent observer *sampling*, NOT ComfyUI survival.
It is never trusted as process ownership, historical event authentication,
or production/GPU qualification. No scheduler or GPU APIs are called here.
"""
from __future__ import annotations

import ntpath
import os
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings
from artifex.render_node.process_identity import (
    WindowsProcessIdentity,
    windows_process_identity,
)

_NAME = "observer-health.json"
_MAX_SIZE = 4096
SampleState = Literal["verified", "blocked", "unsupported"]


class ObserverHeartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    node_id: str = Field(min_length=1, max_length=128)
    observed_utc: datetime
    process_pid: int = Field(ge=1)
    poll_seconds: float = Field(ge=5, le=300)
    sample_count: int = Field(ge=1)
    last_sample_state: SampleState
    production_qualified: Literal[False] = False


# "fresh" means recent AND verified; a recent failed observation is not healthy.
Status = Literal[
    "fresh", "blocked", "unsupported", "missing", "stale", "unsafe",
    "unavailable", "process_missing", "process_mismatch", "process_unavailable",
]


def _root(settings: ArtifexSettings) -> Path:
    root = settings.render_agent.survival_evidence_dir.expanduser().absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Observer heartbeat directory is symlinked")
    return root


def publish_observer_heartbeat(
    settings: ArtifexSettings,
    *,
    observed_utc: datetime,
    poll_seconds: float,
    sample_count: int,
    state: SampleState,
) -> None:
    """Replace one fixed spool status file atomically, never follow a symlink."""
    record = ObserverHeartbeat(
        node_id=settings.render_agent.node_id,
        observed_utc=observed_utc,
        process_pid=os.getpid(),
        poll_seconds=poll_seconds,
        sample_count=sample_count,
        last_sample_state=state,
    )
    if observed_utc.tzinfo is None:
        raise ValueError("Heartbeat timestamp must have a timezone")
    data = record.model_dump_json().encode("utf-8")
    if len(data) > _MAX_SIZE:
        raise ValueError("Heartbeat exceeded its output limit")
    root = _root(settings)
    root.mkdir(parents=True, exist_ok=True)
    # Validate again after mkdir to avoid following an existing directory link.
    if _root(settings) != root or not root.is_dir():
        raise ValueError("Observer heartbeat directory is unsafe")
    target = root / _NAME
    if target.is_symlink():
        raise ValueError("Refusing symlinked observer heartbeat")
    temp = root / (".observer-health-" + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if target.is_symlink():
            raise ValueError("Refusing symlinked observer heartbeat")
        os.replace(temp, target)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def inspect_observer_heartbeat(
    settings: ArtifexSettings, *, now: datetime | None = None,
    config: Path | None = None,
    process_probe: Callable[[int], WindowsProcessIdentity | None] = windows_process_identity,
) -> Status:
    """Read bounded heartbeat and optionally corroborate its native writer PID.

    A verified PID/argv is only a liveness check, not Scheduler ownership or GPU proof.
    """
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        return "unsafe"
    try:
        root = _root(settings)
        target = root / _NAME
        if target.is_symlink():
            return "unsafe"
        if not target.exists():
            return "missing"
        if not target.is_file() or not 0 < target.stat().st_size <= _MAX_SIZE:
            return "unsafe"
        with target.open("rb") as handle:
            raw = handle.read(_MAX_SIZE + 1)
        if not 0 < len(raw) <= _MAX_SIZE:
            return "unsafe"
        record = ObserverHeartbeat.model_validate_json(raw)
        delta = current - record.observed_utc
        if record.node_id != settings.render_agent.node_id:
            return "unsafe"
        if record.observed_utc.tzinfo is None:
            return "unsafe"
        if (
            delta < -timedelta(seconds=30)
            or delta > timedelta(seconds=max(90, record.poll_seconds * 3))
        ):
            return "stale"
        if record.last_sample_state != "verified":
            return record.last_sample_state
        if config is None:
            return "fresh"
        # A venv launcher can exit while its actual Python interpreter remains.
        # Corroborate the writer itself; never assume TaskScheduler's shim PID.
        try:
            selected = config.expanduser().resolve(strict=True)
            if not selected.is_file() or any(
                item.is_symlink() for item in (config, *config.parents)
            ):
                return "process_mismatch"
            args = subprocess.list2cmdline([
                "-m", "artifex.cli", "render-node", "survival-watch",
                "--config", str(selected),
            ])
            process = process_probe(record.process_pid)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return "process_unavailable"
        if process is None:
            return "process_missing"
        try:
            started = datetime.fromisoformat(process.started_utc)
            matches = (
                process.pid == record.process_pid
                and started.tzinfo is not None
                and started <= record.observed_utc
                and ntpath.basename(process.executable).casefold() in {
                    "python.exe", "pythonw.exe",
                }
                and process.command_line.casefold().endswith(
                    (" " + args).casefold()
                )
            )
        except (TypeError, ValueError):
            matches = False
        return "fresh" if matches else "process_mismatch"
    except (ValueError, TypeError, UnicodeError):
        return "unsafe"
    except OSError:
        return "unavailable"
