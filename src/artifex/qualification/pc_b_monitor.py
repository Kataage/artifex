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
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.config.models import ArtifexSettings
from artifex.qualification.overview import latest_saved_session_id
from artifex.qualification.owner_handoff import correlate_live_pc_b_owner_report
from artifex.qualification.survival_review import review_live_pc_b_survival_evidence
from artifex.telemetry import EventSeverity, TelemetryRepository

_SNAPSHOT = "pc-b-passive-monitor.json"
_MAX_BYTES = 32768


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
        if path.stat().st_size > _MAX_BYTES:
            raise ValueError("oversized")
        snapshot = PCBEvidenceMonitorSnapshot.model_validate_json(path.read_bytes())
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
        next_action=(
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
        # Only monitor a saved, locally owned qualification session.
        session_id = latest_saved_session_id(settings.qualification.evidence_dir)
        if session_id is None:
            return None
        now = datetime.now(UTC)
        owner_status = "probe_failed"
        survival_status = "probe_failed"
        owner_failed = survival_failed = False
        try:
            owner_status = correlate_live_pc_b_owner_report(settings, now=now).status
        except Exception:  # noqa: BLE001 - one failed GET must not mask the other
            owner_failed = True
        try:
            survival_status = review_live_pc_b_survival_evidence(settings, now=now).status
        except Exception:  # noqa: BLE001 - no untrusted exception details persisted
            survival_failed = True
        sample = PCBEvidenceMonitorSnapshot(
            captured_utc=now, controller_host=socket.gethostname(),
            session_id=session_id, node_id=primary[0],
            endpoint_fingerprint=_fingerprint(settings),
            owner_status=owner_status, survival_status=survival_status,
            owner_check_failed=owner_failed, survival_check_failed=survival_failed,
        )
        path = _snapshot_path(settings)
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
        return snapshot
