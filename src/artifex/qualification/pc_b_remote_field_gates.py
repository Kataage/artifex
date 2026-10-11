"""Combine existing authenticated PC-B installation and cached launcher GETs.

This is a *remote passive* PC-A diagnostic. It never executes the native
disposable launcher fixture on request, contacts the ComfyUI GPU API,
changes Task Scheduler, or elevates evidence to qualification PASS.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.native_creation_time import native_creation_instant
from artifex.render_node.client import (
    fetch_remote_installation_audit,
    fetch_remote_owner_readiness,
)
from artifex.render_node.installation_audit import RemoteRendererInstallationAudit
from artifex.render_node.models import RemoteOwnerReadinessEvidence

GateStatus = Literal[
    "unconfigured", "unavailable", "needs_evidence", "observed_prerequisites",
]
ProbeStatus = Literal["not_requested", "unavailable", "observed"]
_MAX_CACHE_AGE = timedelta(minutes=10)
_MAX_INSTALL_AGE = timedelta(minutes=2)
_CLOCK_SKEW = timedelta(seconds=30)


class RemotePCBPassiveFieldGates(BaseModel):
    """Status-only summary. No arbitrary PC-B strings, URLs or raw process IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checked_utc: datetime
    node_id: str | None
    status: GateStatus
    installation_probe_status: ProbeStatus
    cached_owner_probe_status: ProbeStatus
    renderer_task_status: str = "not_observed"
    observer_task_status: str = "not_observed"
    observer_heartbeat_status: str = "not_observed"
    launcher_fixture_status: Literal["pass", "fail", "not_observed"] = "not_observed"
    real_comfyui_owner_status: Literal["pass", "fail", "not_observed"] = "not_observed"
    blockers: tuple[str, ...]
    authenticated_read_only_gets_attempted: int = Field(ge=0, le=2)
    # These are two independent point-in-time GETs, not proof of an event.
    verified_actual_supervisor_loss: Literal[False] = False
    uninterrupted_renderer_survival_proven: Literal[False] = False
    live_gpu_work_qualified: Literal[False] = False
    session_stage_pass_registered: Literal[False] = False
    service_mutated: Literal[False] = False
    disposable_fixture_started_by_get: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    production_qualified: Literal[False] = False


def _fresh(value: datetime, now: datetime, limit: timedelta) -> bool:
    return (
        value.tzinfo is not None
        and now.tzinfo is not None
        and -_CLOCK_SKEW <= now - value <= limit
    )


def _safe_cached_status(report: dict[str, Any], key: str) -> Literal[
    "pass", "fail", "not_observed"
]:
    checks = report.get("checks")
    if not isinstance(checks, dict):
        return "not_observed"
    entry = checks.get(key)
    if not isinstance(entry, dict):
        return "not_observed"
    status = entry.get("status")
    if status == "pass":
        return "pass"
    if status == "fail":
        return "fail"
    return "not_observed"


def compile_remote_pc_b_passive_gates(
    *,
    node_id: str | None,
    configured: bool,
    installation: RemoteRendererInstallationAudit | None,
    cached_owner: RemoteOwnerReadinessEvidence | None,
    attempted: int,
    checked_utc: datetime | None = None,
) -> RemotePCBPassiveFieldGates:
    """Treat missing/stale/foreign/protected-failure evidence as unresolved."""
    now = checked_utc or datetime.now(UTC)
    blockers: list[str] = []
    renderer = observer = heartbeat = "not_observed"
    fixture: Literal["pass", "fail", "not_observed"] = "not_observed"
    live_owner: Literal["pass", "fail", "not_observed"] = "not_observed"
    if not configured:
        blockers.append("pc_b_attestation_or_bearer_not_configured")
    elif installation is None:
        blockers.append("pc_b_installation_get_unavailable")
    else:
        report = installation.audit
        if (
            installation.node_id != node_id or report.node_id != node_id
            or not _fresh(report.captured_utc, now, _MAX_INSTALL_AGE)
        ):
            blockers.append("pc_b_installation_node_or_time_mismatch")
        renderer = report.renderer_task.status
        observer = report.survival_observer_task.status
        heartbeat = report.observer_heartbeat
        if (
            not report.native_windows
            or not report.managed_renderer_configured
            or not report.safe_for_passive_observation
            or renderer != "running" or observer != "running"
            or not report.renderer_task.managed
            or not report.survival_observer_task.managed
            or not report.renderer_task.configuration_verified
            or not report.survival_observer_task.configuration_verified
            or heartbeat != "fresh"
            or report.evidence_spool in {"unsafe", "unavailable"}
        ):
            blockers.append("pc_b_native_tasks_or_observer_not_ready")
    if configured and cached_owner is None:
        blockers.append("pc_b_cached_launcher_get_unavailable")
    elif configured and cached_owner is not None:
        data = cached_owner.report
        fixture = _safe_cached_status(data, "disposable_launcher_fixture")
        live_owner = _safe_cached_status(data, "live_comfyui_owner")
        started = data.get("collected_utc")
        try:
            collected_at = (
                datetime.fromisoformat(started) if isinstance(started, str) else None
            )
        except ValueError:
            collected_at = None
        owner = data.get("owner_audit")
        owner_obj = owner if isinstance(owner, dict) else {}
        started_utc = native_creation_instant(
            owner_obj.get("actual_process_started_utc")
            if isinstance(owner_obj.get("actual_process_started_utc"), str)
            else None
        )
        pid = owner_obj.get("actual_listener_pid")
        valid_pid = isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
        origin = installation.audit if installation is not None else None
        if (
            cached_owner.node_id != node_id
            or origin is None
            or data.get("status") != "observed_independently"
            or data.get("observations_are_independent") is not True
            or data.get("actual_comfyui_inspected_by_owner_audit") is not True
            or data.get("production_qualified") is not False
            or data.get("renderer_restart_authorized") is not False
            or fixture != "pass" or live_owner != "pass"
            or owner_obj.get("status") != "observed_stable"
            or not valid_pid or started_utc is None
            or origin.captured_utc.tzinfo is None
            or started_utc > origin.captured_utc.astimezone(UTC)
            or not isinstance(data.get("host"), str)
            or data["host"].strip().casefold() != origin.hostname.strip().casefold()
            or collected_at is None
            or not _fresh(collected_at, now, _MAX_CACHE_AGE)
        ):
            blockers.append("pc_b_cached_owner_stale_or_unverified")
    if not configured:
        state: GateStatus = "unconfigured"
    elif installation is None and cached_owner is None:
        state = "unavailable"
    elif blockers:
        state = "needs_evidence"
    else:
        state = "observed_prerequisites"
    return RemotePCBPassiveFieldGates(
        checked_utc=now, node_id=node_id, status=state,
        installation_probe_status=(
            "not_requested" if not configured else
            "observed" if installation is not None else "unavailable"
        ),
        cached_owner_probe_status=(
            "not_requested" if not configured else
            "observed" if cached_owner is not None else "unavailable"
        ),
        renderer_task_status=renderer,
        observer_task_status=observer,
        observer_heartbeat_status=heartbeat,
        launcher_fixture_status=fixture,
        real_comfyui_owner_status=live_owner,
        blockers=tuple(dict.fromkeys(blockers)),
        authenticated_read_only_gets_attempted=attempted,
    )


def inspect_remote_pc_b_passive_gates(
    settings: ArtifexSettings, *,
    now: datetime | None = None,
    installation_probe: Callable[..., RemoteRendererInstallationAudit] = (
        fetch_remote_installation_audit
    ),
    owner_cache_probe: Callable[..., RemoteOwnerReadinessEvidence] = (
        fetch_remote_owner_readiness
    ),
) -> RemotePCBPassiveFieldGates:
    """Only existing protected GET endpoints; independent bounded failures."""
    checked = now or datetime.now(UTC)
    primary = settings.render_nodes.primary_node()
    if primary is None:
        return compile_remote_pc_b_passive_gates(
            node_id=None, configured=False, installation=None,
            cached_owner=None, attempted=0, checked_utc=checked,
        )
    node_id, node = primary
    configured = bool(
        node.attestation_url and node.attestation_token_env
        and os.environ.get(node.attestation_token_env)
    )
    if not configured:
        return compile_remote_pc_b_passive_gates(
            node_id=node_id, configured=False, installation=None,
            cached_owner=None, attempted=0, checked_utc=checked,
        )
    installation: RemoteRendererInstallationAudit | None = None
    cached: RemoteOwnerReadinessEvidence | None = None
    # No fallback to unprotected HTTP, shell, local fixture or service action.
    # A failed GET cannot hide the other independent observation.
    try:
        installation = installation_probe(node_id, node, timeout_seconds=8.0)
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError):
        pass
    try:
        cached = owner_cache_probe(node_id, node, timeout_seconds=8.0)
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError):
        pass
    return compile_remote_pc_b_passive_gates(
        node_id=node_id, configured=True, installation=installation,
        cached_owner=cached, attempted=2, checked_utc=datetime.now(UTC)
        if now is None else now,
    )
