"""PC-A correlation of a transferred PC-B report with fresh authenticated facts.

Transferred JSON is NOT signed or authenticated. A matching report cannot
promote a production qualification stage, authorize renderer lifecycle changes,
or prove a real GPU child survives supervisor loss.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.native_creation_time import (
    native_creation_instant,
    same_native_creation_instant,
)
from artifex.qualification.renderer_owner_evidence import _REQUIRED_CHECKS
from artifex.render_node.client import (
    fetch_remote_owner_readiness,
    fetch_render_attestation,
    fetch_renderer_owner_audit,
)
from artifex.render_node.models import (
    RemoteOwnerReadinessEvidence,
    RemoteRendererOwnerAudit,
    RendererOwnerAudit,
    RendererOwnerAuditCheck,
    RenderNodeAttestation,
)
from artifex.render_node.native_launcher_probe import LauncherEvidence

_MAX_REPORT_BYTES = 128 * 1024
_MAX_AGE = timedelta(minutes=10)
_MAX_CLOCK_SKEW = timedelta(seconds=30)
_MAX_LIVE_AGE = timedelta(minutes=2)


class TransferredPCBOwnerReport(BaseModel):
    """Strict PR #116 payload. Its contents remain operator-supplied data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    collected_utc: datetime
    host: str
    status: Literal["observed_independently", "blocked", "unsupported"]
    configured_comfyui_python: str | None
    checks: dict[str, RendererOwnerAuditCheck]
    launcher_fixture: LauncherEvidence | None
    owner_audit: RendererOwnerAudit
    remaining_real_machine_evidence: tuple[str, ...]
    observations_are_independent: Literal[True]
    actual_comfyui_inspected_by_owner_audit: bool
    production_child_survival_qualified: Literal[False]
    real_machine_gpu_qualified: Literal[False]
    issue_93_closure_authorized: Literal[False]
    issue_40_closure_authorized: Literal[False]
    renderer_restart_authorized: Literal[False]
    gpu_jobs_submitted: Literal[False]
    production_qualified: Literal[False]
    mutated_services: Literal[False]


CorrelationState = Literal[
    "correlated_read_only", "blocked", "stale", "mismatch", "unavailable",
]


class PCBOwnerCorrelation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checked_utc: datetime
    status: CorrelationState
    reason: str
    node_id: str | None = None
    report_sha256: str | None = None
    saved_host: str | None = None
    live_host: str | None = None
    saved_listener_pid: int | None = None
    live_listener_pid: int | None = None
    live_listener_started_utc: str | None = None
    remaining_real_machine_evidence: tuple[str, ...]
    file_source_authenticated: Literal[False] = False
    source_mode: Literal["transferred_file", "authenticated_remote"] = "transferred_file"
    # Bearer check authenticates the remote request, NOT the saved provenance,
    # and does not imply that HTTP transport is encrypted.
    remote_bearer_checked: bool = False
    launcher_fixture_is_real_comfyui: Literal[False] = False
    real_gpu_qualified: Literal[False] = False
    supervisor_loss_survival_qualified: Literal[False] = False
    stage_pass_registered: Literal[False] = False
    renderer_restart_authorized: Literal[False] = False
    services_mutated: Literal[False] = False
    production_qualified: Literal[False] = False


def _fresh(ts: datetime, now: datetime, *, window: timedelta) -> bool:
    return (
        ts.tzinfo is not None and now.tzinfo is not None
        and -_MAX_CLOCK_SKEW <= now - ts <= window
    )


def _stable(report: RendererOwnerAudit) -> bool:
    started = native_creation_instant(report.actual_process_started_utc)
    return (
        report.status == "observed_stable"
        and report.process_observation_verified
        and report.actual_listener_pid is not None
        and started is not None
        and report.captured_utc.tzinfo is not None
        and started <= report.captured_utc
        and report.restart_authorized is False
        and report.production_qualified is False
        and _REQUIRED_CHECKS.issubset(report.checks)
        and all(report.checks[key].status == "pass" for key in _REQUIRED_CHECKS)
    )


def correlate_pc_b_owner_report(
    settings: ArtifexSettings,
    source: Path | None,
    *,
    remote: bool = False,
    now: datetime | None = None,
    remote_probe: Callable[
        [str, RenderNodeConfig], RemoteOwnerReadinessEvidence
    ] = fetch_remote_owner_readiness,
    owner_probe: Callable[
        [str, RenderNodeConfig], RemoteRendererOwnerAudit
    ] = fetch_renderer_owner_audit,
    attestation_probe: Callable[..., RenderNodeAttestation] = fetch_render_attestation,
) -> PCBOwnerCorrelation:
    """Compare a bounded copied file to current authenticated PC-B observations.

    Never fetch arbitrary URLs from the JSON: network targets come exclusively
    from the trusted PC-A configuration. Never mutate stage or service state.
    """
    current = now or datetime.now(UTC)
    primary = settings.render_nodes.primary_node()
    node_id = primary[0] if primary else None
    report: TransferredPCBOwnerReport | None = None
    sha: str | None = None

    def result(
        state: CorrelationState, reason: str, *,
        live_host: str | None = None, live_pid: int | None = None,
        live_started_utc: str | None = None,
    ) -> PCBOwnerCorrelation:
        outstanding = list(
            report.remaining_real_machine_evidence if report is not None else ()
        )
        # A matching copied snapshot does not prove these remaining tests.
        for item in (
            "real_comfyui_child_survival_after_supervisor_loss",
            "real_gpu_production_workflow_result",
            "eight_hour_unattended_gpu_soak",
            "all_fourteen_real_machine_qualification_stages",
        ):
            if item not in outstanding:
                outstanding.append(item)
        if state != "correlated_read_only":
            outstanding.insert(0, "fresh_pc_b_local_owner_report_correlated_to_live_host")
        return PCBOwnerCorrelation(
            checked_utc=current, status=state, reason=reason, node_id=node_id,
            report_sha256=sha,
            saved_host=report.host if report is not None else None,
            live_host=live_host,
            saved_listener_pid=(
                report.owner_audit.actual_listener_pid if report is not None else None
            ),
            live_listener_pid=live_pid,
            live_listener_started_utc=live_started_utc,
            remaining_real_machine_evidence=tuple(dict.fromkeys(outstanding)),
            source_mode="authenticated_remote" if remote else "transferred_file",
            remote_bearer_checked=bool(remote and report is not None),
        )

    if remote:
        # Bearer-authenticated endpoint serves a previously produced local
        # cache. GET never launches even a disposable Python child.
        if primary is None:
            return result("blocked", "pc_a_primary_render_node_missing")
        node, config = primary
        try:
            received = remote_probe(node, config)
        except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError):
            return result("unavailable", "authenticated_pc_b_evidence_unavailable")
        if received.node_id != node:
            return result("blocked", "remote_pc_b_evidence_node_mismatch")
        try:
            raw = json.dumps(
                received.report, ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")
            if not 0 < len(raw) <= _MAX_REPORT_BYTES:
                return result("blocked", "remote_pc_b_evidence_oversized")
            report = TransferredPCBOwnerReport.model_validate(received.report)
            sha = hashlib.sha256(raw).hexdigest()
        except (ValueError, TypeError):
            return result("blocked", "remote_pc_b_evidence_invalid_schema")
    else:
        if source is None:
            return result("blocked", "pc_b_file_not_supplied")
        path = source.expanduser().absolute()
        try:
            if any(item.is_symlink() for item in (path, *path.parents)):
                return result("blocked", "symlinked_pc_b_evidence_path")
            if not path.is_file() or not 0 < path.stat().st_size <= _MAX_REPORT_BYTES:
                return result("blocked", "pc_b_evidence_missing_empty_or_oversized")
            raw = path.read_bytes()
            if not 0 < len(raw) <= _MAX_REPORT_BYTES:
                return result("blocked", "pc_b_evidence_invalid_size")
            sha = hashlib.sha256(raw).hexdigest()
            report = TransferredPCBOwnerReport.model_validate_json(raw)
        except (OSError, ValueError, TypeError):
            return result("blocked", "pc_b_evidence_invalid_schema_or_unreadable")

    if primary is None:
        return result("blocked", "pc_a_primary_render_node_missing")
    if report.status != "observed_independently":
        return result("blocked", "pc_b_report_did_not_pass_both_observations")
    fixture = report.launcher_fixture
    owner = report.owner_audit
    if (
        fixture is None or fixture.status != "observed"
        or fixture.host.casefold() != report.host.casefold()
        or not fixture.verified_runtime_provenance
        or not fixture.original_launcher_verified
        or not fixture.listener_identity_stable or not fixture.tcp_owner_stable
        or not report.actual_comfyui_inspected_by_owner_audit
        or any(
            report.checks.get(key) is None
            or report.checks[key].status != "pass"
            for key in ("disposable_launcher_fixture", "live_comfyui_owner")
        )
        or not _stable(owner)
    ):
        return result("blocked", "pc_b_report_internally_inconsistent")
    if not (
        _fresh(report.collected_utc, current, window=_MAX_AGE)
        and _fresh(owner.captured_utc, current, window=_MAX_AGE)
        and _fresh(fixture.observed_at_utc, current, window=_MAX_AGE)
        and owner.captured_utc <= report.collected_utc + _MAX_CLOCK_SKEW
        and fixture.observed_at_utc <= report.collected_utc + _MAX_CLOCK_SKEW
    ):
        return result("stale", "pc_b_report_timestamps_stale_or_inconsistent")

    node, cfg = primary
    try:
        live_remote_owner = owner_probe(node, cfg)
        # Existing authenticated attestation channel; fresh=true prevents
        # using an old cached host identity. No remote process launch.
        att = attestation_probe(node, cfg, fresh=True, timeout_seconds=35.0)
    except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError):
        return result("unavailable", "authenticated_pc_b_snapshot_unavailable")

    live = live_remote_owner.audit
    if not (
        live_remote_owner.node_id == node and att.node_id == node
        and _fresh(live.captured_utc, current, window=_MAX_LIVE_AGE)
        and _fresh(att.created_at, current, window=_MAX_LIVE_AGE)
        and _stable(live)
    ):
        return result(
            "blocked", "current_authenticated_pc_b_owner_not_stably_verified",
            live_host=att.hostname, live_pid=live.actual_listener_pid,
            live_started_utc=live.actual_process_started_utc,
        )
    if not (
        report.host.casefold() == att.hostname.casefold()
        and owner.actual_listener_pid == live.actual_listener_pid
        and same_native_creation_instant(
            owner.actual_process_started_utc, live.actual_process_started_utc,
        )
        and owner.launcher_pid == live.launcher_pid
        and owner.receipt_schema == live.receipt_schema
        and owner.scheduler_state == live.scheduler_state
        and live.captured_utc >= owner.captured_utc - _MAX_CLOCK_SKEW
    ):
        return result(
            "mismatch", "saved_pc_b_host_or_renderer_identity_differs_from_live",
            live_host=att.hostname, live_pid=live.actual_listener_pid,
            live_started_utc=live.actual_process_started_utc,
        )
    return result(
        "correlated_read_only", "same_host_and_renderer_observed_at_two_times",
        live_host=att.hostname, live_pid=live.actual_listener_pid,
            live_started_utc=live.actual_process_started_utc,
    )



def correlate_live_pc_b_owner_report(
    settings: ArtifexSettings,
    *,
    now: datetime | None = None,
) -> PCBOwnerCorrelation:
    """PC-A one-command remote evidence check; no copy, no remote process start."""
    return correlate_pc_b_owner_report(settings, None, remote=True, now=now)
