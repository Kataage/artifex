"""Periodic, passive PC-B ownership/survival observations for the PC-A daemon.

Observations are saved as *historical diagnostic data*, never qualification
receipts. No GPU work, process control, Discord send, stage mutation, or
implicit permission to restart ComfyUI is involved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import socket
import stat
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.config.models import ArtifexSettings
from artifex.qualification.native_creation_time import native_creation_instant
from artifex.qualification.owner_handoff import correlate_live_pc_b_owner_report
from artifex.qualification.owner_pair import _valid_owner
from artifex.render_node.client import fetch_renderer_owner_audit
from artifex.qualification.survival_review import review_live_pc_b_survival_evidence
from artifex.telemetry import EventSeverity, TelemetryRepository

_SNAPSHOT = "pc-b-passive-monitor.json"
_MAX_BYTES = 32768
IdentityChange = Literal["not_comparable", "unchanged", "changed"]
LiveOwnerState = Literal["not_checked", "observed_stable", "blocked", "unavailable"]


class PCBEvidenceMonitorSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    captured_utc: datetime
    controller_host: str
    session_id: str
    node_id: str
    endpoint_fingerprint: str
    owner_status: str
    survival_status: str
    owner_check_failed: bool
    survival_check_failed: bool
    live_owner_status: LiveOwnerState = "not_checked"
    live_owner_check_failed: bool = False
    owner_identity_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$",
    )
    identity_change: IdentityChange = "not_comparable"
    actual_gpu_qualified: Literal[False] = False
    historical_event_authenticated: Literal[False] = False
    issue_93_closed: Literal[False] = False
    stage_pass_registered: Literal[False] = False
    production_qualified: Literal[False] = False
    services_mutated: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False

    @model_validator(mode="after")
    def validate_capture(self) -> PCBEvidenceMonitorSnapshot:
        if self.captured_utc.tzinfo is None:
            raise ValueError("PC-B monitor timestamp requires timezone")
        if not self.controller_host or not self.session_id or not self.node_id:
            raise ValueError("PC-B monitor context must not be empty")
        if len(self.endpoint_fingerprint) != 64:
            raise ValueError("PC-B monitor endpoint fingerprint is invalid")
        if self.owner_identity_sha256 is not None and (
            self.live_owner_status != "observed_stable"
            or self.live_owner_check_failed
        ):
            raise ValueError("Owner digest requires a verified live owner")
        if self.identity_change != "not_comparable" and (
            self.owner_identity_sha256 is None
        ):
            raise ValueError("Cannot compare absent live owner identities")
        return self


MonitorState = Literal[
    "disabled", "no_session", "missing", "invalid", "context_mismatch",
    "stale", "historical_observation",
]


class PCBEvidenceMonitorView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: MonitorState
    captured_utc: datetime | None = None
    age_seconds: float | None = Field(default=None, ge=0)
    owner_status: str | None = None
    survival_status: str | None = None
    owner_check_failed: bool | None = None
    survival_check_failed: bool | None = None
    live_owner_status: LiveOwnerState = "not_checked"
    live_owner_check_failed: bool | None = None
    identity_change: IdentityChange = "not_comparable"
    source: Literal["saved_pc_a_monitor_only"] = "saved_pc_a_monitor_only"
    live_rechecked_by_this_overview: Literal[False] = False
    historical_event_authenticated: Literal[False] = False
    actual_gpu_qualified: Literal[False] = False
    issue_93_closed: Literal[False] = False
    stage_pass_registered: Literal[False] = False
    production_qualified: Literal[False] = False
    next_action: str


def _fingerprint(settings: ArtifexSettings) -> str:
    primary = settings.render_nodes.primary_node()
    if primary is None:
        return ""
    node_id, node = primary
    # Bind a saved observation to the exact configured target, not to
    # credentials (tokens never appear in either snapshot or fingerprint).
    key = json.dumps(
        [node_id, node.base_url, node.attestation_url],
        ensure_ascii=True, separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _snapshot_path(settings: ArtifexSettings) -> Path:
    return settings.qualification.evidence_dir.expanduser().absolute() / _SNAPSHOT


def _safe_existing_parent(path: Path) -> bool:
    return (
        path.parent.is_dir()
        and not path.is_symlink()
        and not any(p.is_symlink() for p in (path.parent, *path.parent.parents))
    )


def _read_bounded_monitor_snapshot(path: Path) -> bytes:
    """Read only the same bounded regular file identified before and after open.

    In particular, do not stat one inode and read another after a path swap.
    Linux O_NOFOLLOW rejects final-component symlinks at open; Windows uses
    lstat/fstat identity checks because O_NOFOLLOW is not available there.
    """
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_BYTES:
        raise ValueError("Unsafe or oversized monitor snapshot")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size != before.st_size
            or opened.st_mtime_ns != before.st_mtime_ns
        ):
            raise ValueError("Monitor snapshot changed before open")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(_MAX_BYTES + 1)
        finished = os.fstat(fd)
        after = path.lstat()
        if (
            len(payload) > _MAX_BYTES
            or not stat.S_ISREG(after.st_mode)
            or (finished.st_dev, finished.st_ino) != (opened.st_dev, opened.st_ino)
            or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            or (finished.st_size, finished.st_mtime_ns)
            != (opened.st_size, opened.st_mtime_ns)
            or (after.st_size, after.st_mtime_ns)
            != (opened.st_size, opened.st_mtime_ns)
        ):
            raise ValueError("Monitor snapshot changed during read")
        return payload
    finally:
        os.close(fd)


def _identity_digest(pid: int | None, started: str | None) -> str | None:
    """Hash a verified PID plus native creation time; never store raw identity."""
    instant = native_creation_instant(started)
    if pid is None or pid < 1 or instant is None:
        return None
    value = json.dumps([pid, instant.isoformat()], separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _previous_owner_digest(
    path: Path, *, settings: ArtifexSettings, session_id: str,
    now: datetime, node_id: str,
) -> str | None:
    """Only recent, context-matched verified snapshots are comparable."""
    if not _safe_existing_parent(path):
        return None
    try:
        prior = PCBEvidenceMonitorSnapshot.model_validate_json(
            _read_bounded_monitor_snapshot(path),
        )
    except (OSError, ValueError):
        return None
    age = (now.astimezone(UTC) - prior.captured_utc.astimezone(UTC)).total_seconds()
    if (
        prior.controller_host != socket.gethostname()
        or prior.session_id != session_id
        or prior.node_id != node_id
        or prior.endpoint_fingerprint != _fingerprint(settings)
        or prior.live_owner_status != "observed_stable"
        or prior.live_owner_check_failed
        or not -30 <= age <= settings.qualification.auto_pc_b_monitor_max_age_seconds
    ):
        return None
    return prior.owner_identity_sha256


def read_pc_b_evidence_monitor(
    settings: ArtifexSettings, *, session_id: str | None,
    now: datetime, max_age_seconds: int,
) -> PCBEvidenceMonitorView:
    """Read a bounded prior report, but never treat it as a live observation."""
    if settings.render_nodes.primary_node() is None:
        return PCBEvidenceMonitorView(
            status="disabled", next_action="No primary remote PC-B renderer configured.",
        )
    if session_id is None:
        return PCBEvidenceMonitorView(
            status="no_session",
            next_action="No selected PC-A qualification session. Do not reuse older proof.",
        )
    path = _snapshot_path(settings)
    if not _safe_existing_parent(path) or not path.is_file():
        return PCBEvidenceMonitorView(
            status="missing",
            next_action="No safe saved PC-B monitor observation exists yet; use live read-only checks.",
        )
    try:
        payload = _read_bounded_monitor_snapshot(path)
        snapshot = PCBEvidenceMonitorSnapshot.model_validate_json(payload)
    except (OSError, ValueError):
        return PCBEvidenceMonitorView(
            status="invalid",
            next_action="Saved PC-B monitor evidence cannot be safely parsed; recheck live.",
        )
    if (
        snapshot.controller_host != socket.gethostname()
        or snapshot.session_id != session_id
        or snapshot.endpoint_fingerprint != _fingerprint(settings)
        or snapshot.node_id != settings.render_nodes.primary
    ):
        return PCBEvidenceMonitorView(
            status="context_mismatch",
            next_action="Saved PC-B monitor belongs to another PC-A, session or renderer configuration.",
        )
    current = now.astimezone(UTC)
    delta = (current - snapshot.captured_utc.astimezone(UTC)).total_seconds()
    if delta < -30 or delta > max_age_seconds:
        return PCBEvidenceMonitorView(
            status="stale", captured_utc=snapshot.captured_utc,
            next_action="PC-B monitor sample has expired or clock is inconsistent; live recheck required.",
        )
    return PCBEvidenceMonitorView(
        status="historical_observation",
        captured_utc=snapshot.captured_utc,
        age_seconds=max(0.0, delta),
        owner_status=snapshot.owner_status,
        survival_status=snapshot.survival_status,
        owner_check_failed=snapshot.owner_check_failed,
        survival_check_failed=snapshot.survival_check_failed,
        live_owner_status=snapshot.live_owner_status,
        live_owner_check_failed=snapshot.live_owner_check_failed,
        identity_change=snapshot.identity_change,
        next_action=(
            "A different verified PC-B listener PID/creation identity was observed "
            "between two historical polls. Recheck the live owner read-only; "
            "never restart ComfyUI automatically. "
            if snapshot.identity_change == "changed" else ""
        ) + (
            "These are saved PC-A polling results, NOT current owner checks, "
            "authenticated supervisor-loss history, GPU evidence or stage PASS. "
            "Run live PC-B owner/survival checks before real-machine acceptance."
        ),
    )


class PCBEvidenceRefreshMaintenance:
    """Rate-limited, failure-isolated read-only PC-B monitor on the native daemon."""

    def __init__(
        self,
        settings: ArtifexSettings,
        telemetry: TelemetryRepository,
        *, interval_seconds: int = 900,
    ) -> None:
        if interval_seconds < 300:
            raise ValueError("PC-B monitor interval must be at least 300 seconds")
        self._settings = settings
        self._telemetry = telemetry
        self._interval_seconds = interval_seconds
        self._last_run: float | None = None

    def _refresh(self) -> PCBEvidenceMonitorSnapshot | None:
        settings = self._settings
        primary = settings.render_nodes.primary_node()
        if primary is None:
            return None
        # Deferred import keeps the report compiler independent of daemon wiring.
        from artifex.qualification.overview import latest_saved_session_id

        # Only monitor a saved, locally owned qualification session.
        session_id = latest_saved_session_id(settings.qualification.evidence_dir)
        if session_id is None:
            return None
        now = datetime.now(UTC)
        owner_status = "probe_failed"
        survival_status = "probe_failed"
        owner_failed = survival_failed = False
        live_owner_status: LiveOwnerState = "unavailable"
        live_owner_failed = False
        current_digest: str | None = None
        try:
            owner_status = correlate_live_pc_b_owner_report(settings, now=now).status
        except Exception:  # noqa: BLE001 - one failed GET must not mask the other
            owner_failed = True
        try:
            survival_status = review_live_pc_b_survival_evidence(settings, now=now).status
        except Exception:  # noqa: BLE001 - one failed GET must not mask the other
            survival_failed = True
        # A separate, non-cached authenticated owner GET is meaningful even
        # after the cached launcher-readiness report has expired.
        try:
            remote = fetch_renderer_owner_audit(primary[0], primary[1])
            live_owner_status = "blocked"
            audit = remote.audit
            if remote.node_id == primary[0] and _valid_owner(
                audit, current=datetime.now(UTC),
            ):
                current_digest = _identity_digest(
                    audit.actual_listener_pid, audit.actual_process_started_utc,
                )
                if current_digest is not None:
                    live_owner_status = "observed_stable"
        except Exception:  # noqa: BLE001 - no tokens/remote errors persisted
            live_owner_failed = True
        captured = datetime.now(UTC)
        path = _snapshot_path(settings)
        previous_digest = (
            _previous_owner_digest(
                path, settings=settings, session_id=session_id,
                now=captured, node_id=primary[0],
            )
            if current_digest is not None else None
        )
        identity_change: IdentityChange = (
            "not_comparable" if previous_digest is None else
            "unchanged" if previous_digest == current_digest else "changed"
        )
        sample = PCBEvidenceMonitorSnapshot(
            captured_utc=captured, controller_host=socket.gethostname(),
            session_id=session_id, node_id=primary[0],
            endpoint_fingerprint=_fingerprint(settings),
            owner_status=owner_status, survival_status=survival_status,
            owner_check_failed=owner_failed, survival_check_failed=survival_failed,
            live_owner_status=live_owner_status,
            live_owner_check_failed=live_owner_failed,
            owner_identity_sha256=current_digest,
            identity_change=identity_change,
        )
        if not _safe_existing_parent(path):
            raise ValueError("Unsafe or absent existing qualification directory")
        # Atomic replacement of our *own* bounded monitor file. Never write
        # qualification.json, never migrate DB, never follow any symlink.
        temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            with temp.open("x", encoding="utf-8") as handle:
                handle.write(sample.model_dump_json() + "\n")
            if not _safe_existing_parent(path):
                raise ValueError("PC-B monitor path changed during write")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return sample

    async def maintain(self) -> PCBEvidenceMonitorSnapshot | None:
        now = time.monotonic()
        if self._last_run is not None and now - self._last_run < self._interval_seconds:
            return None
        self._last_run = now
        try:
            snapshot = await asyncio.to_thread(self._refresh)
        except Exception as exc:  # noqa: BLE001 - monitoring never breaks production
            try:
                self._telemetry.record(
                    "qualification.pc_b_monitor_failed", EventSeverity.WARNING,
                    {"error_type": type(exc).__name__},
                )
            except Exception:  # noqa: BLE001 - best effort only
                logging.getLogger(__name__).warning("PC-B monitor failure audit unavailable")
            return None
        if snapshot is not None and snapshot.identity_change == "changed":
            # This is only an operational warning, never a qualification stage.
            try:
                self._telemetry.record(
                    "qualification.pc_b_owner_identity_changed",
                    EventSeverity.WARNING,
                    {"node_id": snapshot.node_id, "effect": "read_only_review"},
                )
            except Exception:  # noqa: BLE001 - best effort only
                logging.getLogger(__name__).warning(
                    "PC-B owner identity-change telemetry unavailable",
                )
        return snapshot
