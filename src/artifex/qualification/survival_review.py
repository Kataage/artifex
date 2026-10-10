"""PC-A read-only review of copied or shared PC-B supervisor-survival traces.

All JSON evidence is UNTRUSTED. Replaying a coherent trace is not proof it
came from the physical PC-B or proof of a real supervisor-loss event.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.render_node.client import (
    fetch_remote_survival_trace,
    fetch_render_attestation,
    fetch_renderer_owner_audit,
)
from artifex.render_node.models import (
    RemoteRendererOwnerAudit,
    RemoteSurvivalTrace,
    RenderNodeAttestation,
)
from artifex.render_node.supervisor_survival import (
    SurvivalAssessment,
    assess_survival,
)

_MAX_BYTES = 12 * 1024 * 1024
_MAX_SAMPLES = 4096
_CLOCK_SKEW = timedelta(seconds=30)
_LIVE_AGE = timedelta(minutes=2)
_EXACT_CHECKS = {
    "receipt", "process_identity", "launcher_identity", "tcp_ownership",
    "snapshot_consistency",
}
State = Literal[
    "replayed_and_live_identity_matched",
    "replayed_historical_identity", "replayed_live_unavailable",
    "blocked", "mismatch", "remote_unavailable",
]


class PCBSurvivalEvidenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    reviewed_utc: datetime
    status: State
    reason: str
    evidence_sha256: str | None = None
    claimed_event: str | None = None
    trace_replayed: bool = False
    sample_count: int = 0
    node_id: str | None = None
    recorded_host: str | None = None
    live_host: str | None = None
    historical_listener_pid: int | None = None
    current_listener_pid: int | None = None
    current_listener_started_utc: str | None = None
    source_authenticated: Literal[False] = False
    source_mode: Literal["transferred_file", "bearer_remote"] = "transferred_file"
    remote_bearer_checked: bool = False
    historical_event_authenticated: Literal[False] = False
    independent_supervisor_survival_qualified: Literal[False] = False
    real_gpu_qualified: Literal[False] = False
    issue_93_closure_authorized: Literal[False] = False
    issue_40_closure_authorized: Literal[False] = False
    stage_pass_registered: Literal[False] = False
    renderer_restart_authorized: Literal[False] = False
    services_mutated: Literal[False] = False
    production_qualified: Literal[False] = False


def _recent(ts: datetime, now: datetime) -> bool:
    return (
        ts.tzinfo is not None
        and -_CLOCK_SKEW <= now - ts <= _LIVE_AGE
    )


def review_pc_b_survival_evidence(
    settings: ArtifexSettings,
    path: Path | None,
    *,
    remote: bool = False,
    now: datetime | None = None,
    remote_probe: Callable[[str, RenderNodeConfig], RemoteSurvivalTrace] = (
        fetch_remote_survival_trace
    ),
    owner_probe: Callable[[str, RenderNodeConfig], RemoteRendererOwnerAudit] = (
        fetch_renderer_owner_audit
    ),
    attestation_probe: Callable[..., RenderNodeAttestation] = fetch_render_attestation,
) -> PCBSurvivalEvidenceReview:
    """Replay bounded trace, then check host/identity against live PC-B if reachable.

    No URL from JSON is followed. Live endpoints always come from PC-A config.
    No shell, process, scheduler or qualification writes.
    """
    current = now or datetime.now(UTC)
    expected = settings.render_nodes.primary_node()
    recorded: SurvivalAssessment | None = None
    digest: str | None = None
    replayed = False
    live_host: str | None = None
    live_pid: int | None = None
    live_started: str | None = None
    remote_ok = False

    def result(status: State, reason: str) -> PCBSurvivalEvidenceReview:
        return PCBSurvivalEvidenceReview(
            reviewed_utc=current, status=status, reason=reason,
            source_mode="bearer_remote" if remote else "transferred_file",
            remote_bearer_checked=remote_ok,
            evidence_sha256=digest,
            claimed_event=recorded.status if recorded else None,
            trace_replayed=replayed,
            sample_count=len(recorded.samples) if recorded else 0,
            node_id=recorded.node_id if recorded else None,
            recorded_host=recorded.host if recorded else None,
            live_host=live_host,
            historical_listener_pid=(
                recorded.samples[-1].actual_listener_pid
                if recorded and recorded.samples else None
            ),
            current_listener_pid=live_pid,
            current_listener_started_utc=live_started,
        )

    if expected is None:
        return result("blocked", "pc_a_primary_renderer_not_configured")
    node, cfg = expected
    if remote:
        if path is not None:
            return result("blocked", "survival_file_and_remote_are_mutually_exclusive")
        try:
            received = remote_probe(node, cfg)
        except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError):
            return result("remote_unavailable", "pc_b_saved_survival_trace_unavailable")
        if received.node_id != node:
            return result("blocked", "remote_survival_node_mismatch")
        try:
            raw = received.content.encode("utf-8")
            if not 0 < len(raw) <= _MAX_BYTES:
                return result("blocked", "remote_survival_content_size_invalid")
            digest = hashlib.sha256(raw).hexdigest()
            if digest != received.sha256:
                return result("blocked", "remote_survival_sha256_mismatch")
            remote_ok = True
        except (ValueError, UnicodeError):
            return result("blocked", "remote_survival_payload_invalid")
    else:
        if path is None:
            return result("blocked", "survival_file_not_supplied")
        source = path.expanduser().absolute()
        try:
            if any(entry.is_symlink() for entry in (source, *source.parents)):
                return result("blocked", "evidence_path_is_symlinked")
            if not source.is_file() or not 0 < source.stat().st_size <= _MAX_BYTES:
                return result("blocked", "evidence_missing_empty_or_oversized")
            raw = source.read_bytes()
            if not 0 < len(raw) <= _MAX_BYTES:
                return result("blocked", "evidence_oversized")
            digest = hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, TypeError):
            return result("blocked", "evidence_invalid_or_unreadable")
    try:
        recorded = SurvivalAssessment.model_validate_json(raw)
    except (ValueError, TypeError):
        return result("blocked", "evidence_invalid_or_unreadable")
    if (
        len(recorded.samples) < 3 or len(recorded.samples) > _MAX_SAMPLES
        or recorded.host != recorded.samples[0].host
        or recorded.node_id != node
        or recorded.status != "observed_after_supervisor_absence"
        or not recorded.same_comfyui_seen_before_and_after
        or not recorded.supervisor_absence_observed
        or recorded.samples[0].task_state != "Running"
        or recorded.samples[-1].task_state != "Ready"
    ):
        return result("blocked", "evidence_not_a_complete_matching_observation")

    previous_at: datetime | None = None
    previous_elapsed = -1.0
    start = recorded.samples[0].observed_utc
    if start.tzinfo is None:
        return result("blocked", "observation_clock_missing_timezone")
    for item in recorded.samples:
        if (
            item.observed_utc.tzinfo is None
            or item.observed_utc > current + _CLOCK_SKEW
            or item.host != recorded.host
            or item.node_id != recorded.node_id
            or item.elapsed_seconds <= previous_elapsed
            or (previous_at is not None and item.observed_utc <= previous_at)
            or abs(
                (item.observed_utc - start).total_seconds()
                - (item.elapsed_seconds - recorded.samples[0].elapsed_seconds)
            ) > 30
        ):
            return result("blocked", "trace_time_identity_or_monotonicity_invalid")
        previous_at = item.observed_utc
        previous_elapsed = item.elapsed_seconds
    interval = recorded.samples[1].elapsed_seconds - recorded.samples[0].elapsed_seconds
    if interval < 5 or interval > 3600:
        return result("blocked", "trace_initial_interval_out_of_range")
    try:
        recalculated = assess_survival(
            recorded.samples, min_separation_seconds=interval,
        )
    except (ValueError, TypeError):
        return result("blocked", "trace_cannot_be_replayed")
    if (
        recalculated.status != recorded.status
        or recalculated.same_comfyui_seen_before_and_after
        != recorded.same_comfyui_seen_before_and_after
        or recalculated.supervisor_absence_observed
        != recorded.supervisor_absence_observed
        or recalculated.reason != recorded.reason
    ):
        return result("blocked", "saved_verdict_differs_from_replayed_observations")
    replayed = True

    # Current authenticated transport is independent, but cannot authenticate
    # what happened historically. Absence of the attestation service after an
    # exit is expected and must not be recast as evidence of a real event.
    try:
        owner = owner_probe(node, cfg)
        attestation = attestation_probe(
            node, cfg, fresh=True, timeout_seconds=35.0,
        )
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError):
        return result("replayed_live_unavailable", "live_pc_b_unavailable_for_identity_check")

    live = owner.audit
    live_host = attestation.hostname
    live_pid = live.actual_listener_pid
    live_started = live.actual_process_started_utc
    if (
        owner.node_id != node or attestation.node_id != node
        or not _recent(attestation.created_at, current)
        or not _recent(live.captured_utc, current)
        or not live.process_observation_verified
        or live.actual_listener_pid is None
        or live.actual_process_started_utc is None
        or not _EXACT_CHECKS.issubset(live.checks)
        or any(live.checks[k].status != "pass" for k in _EXACT_CHECKS)
    ):
        return result("blocked", "live_pc_b_identity_cannot_be_authenticated")
    if live_host.casefold() != recorded.host.casefold():
        return result("mismatch", "recorded_host_differs_from_live_pc_b")
    historical = recorded.samples[-1]
    if (
        historical.actual_listener_pid != live.actual_listener_pid
        or historical.actual_listener_started_utc != live.actual_process_started_utc
        or historical.launcher_pid != live.launcher_pid
        or historical.receipt_schema != live.receipt_schema
    ):
        return result(
            "replayed_historical_identity",
            "recorded_process_not_the_current_pc_b_listener",
        )
    return result(
        "replayed_and_live_identity_matched",
        "copied_trace_replayed_and_current_pc_b_process_matches",
    )


def review_live_pc_b_survival_evidence(
    settings: ArtifexSettings,
    *,
    now: datetime | None = None,
) -> PCBSurvivalEvidenceReview:
    """Retrieve the latest immutable PC-B trace with no manual file transfer."""
    return review_pc_b_survival_evidence(settings, None, remote=True, now=now)
