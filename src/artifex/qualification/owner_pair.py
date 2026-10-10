"""Two authenticated PC-A observations of real PC-B owner identity.

This only correlates two read-only point-in-time snapshots. It is NOT
uninterrupted survival evidence, GPU qualification, or launch authorization.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.renderer_owner_evidence import _REQUIRED_CHECKS
from artifex.render_node.client import fetch_renderer_owner_audit
from artifex.render_node.models import RemoteRendererOwnerAudit, RendererOwnerAudit

PairStatus = Literal["consistent_samples", "blocked", "unavailable", "unconfigured"]
CheckState = Literal["pass", "fail", "unknown"]


class OwnerPairObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    status: PairStatus
    reason: str
    node_id: str | None
    checked_utc: datetime
    sample_count: int = Field(ge=0, le=2)
    first_captured_utc: datetime | None = None
    second_captured_utc: datetime | None = None
    first_listener_pid: int | None = None
    second_listener_pid: int | None = None
    first_process_started_utc: str | None = None
    second_process_started_utc: str | None = None
    first_launcher_pid: int | None = None
    second_launcher_pid: int | None = None
    first_receipt_schema: int | None = None
    second_receipt_schema: int | None = None
    # Save bounded check outcomes only: no OS command line, executable path,
    # raw reason strings, credentials, or ownership receipt contents.
    first_required_checks: dict[str, CheckState] = Field(default_factory=dict)
    second_required_checks: dict[str, CheckState] = Field(default_factory=dict)
    elapsed_between_samples_seconds: float = Field(ge=0)
    requested_gap_seconds: float = Field(ge=1, le=60)
    # Two clean snapshots can never authorize live GPU lifecycle operations.
    uninterrupted_child_survival_proven: Literal[False] = False
    real_gpu_qualified: Literal[False] = False
    issue_93_closure_authorized: Literal[False] = False
    issue_40_closure_authorized: Literal[False] = False
    task_actions_executed: Literal[False] = False
    renderer_restart_authorized: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    production_qualified: Literal[False] = False


def _valid_owner(audit: RendererOwnerAudit, *, current: datetime) -> bool:
    if (
        audit.status != "observed_stable"
        or audit.process_observation_verified is not True
        or audit.scheduler_state != "Running"
        or audit.actual_listener_pid is None
        or audit.actual_process_started_utc is None
        or audit.receipt_schema not in {1, 2}
        or not _REQUIRED_CHECKS.issubset(audit.checks)
        or any(audit.checks[k].status != "pass" for k in _REQUIRED_CHECKS)
        or audit.captured_utc.tzinfo is None
        or current.tzinfo is None
    ):
        return False
    try:
        started = datetime.fromisoformat(audit.actual_process_started_utc)
    except (TypeError, ValueError):
        return False
    return (
        started.tzinfo is not None
        and started <= audit.captured_utc
        and -30 <= (current - audit.captured_utc).total_seconds() <= 120
    )


def inspect_remote_owner_pair(
    settings: ArtifexSettings, *,
    gap_seconds: float = 5.0,
    owner_probe: Callable[
        [str, RenderNodeConfig], RemoteRendererOwnerAudit
    ] = fetch_renderer_owner_audit,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> OwnerPairObservation:
    """Check two noncached Bearer-gated owner reports without service actions."""
    if not math.isfinite(gap_seconds) or not 1 <= gap_seconds <= 60:
        raise ValueError("Sample gap must be between 1 and 60 seconds")
    primary = settings.render_nodes.primary_node()
    node_id = primary[0] if primary else None
    first: RemoteRendererOwnerAudit | None = None
    second: RemoteRendererOwnerAudit | None = None
    elapsed = 0.0

    def summary(sample: RemoteRendererOwnerAudit | None) -> dict[str, CheckState]:
        if sample is None:
            return {}
        return {
            key: sample.audit.checks[key].status
            for key in sorted(_REQUIRED_CHECKS)
            if key in sample.audit.checks
        }

    def result(status: PairStatus, reason: str) -> OwnerPairObservation:
        return OwnerPairObservation(
            status=status, reason=reason, node_id=node_id,
            checked_utc=now(), requested_gap_seconds=gap_seconds,
            sample_count=int(first is not None) + int(second is not None),
            first_captured_utc=first.audit.captured_utc if first else None,
            second_captured_utc=second.audit.captured_utc if second else None,
            first_listener_pid=first.audit.actual_listener_pid if first else None,
            second_listener_pid=second.audit.actual_listener_pid if second else None,
            first_process_started_utc=(
                first.audit.actual_process_started_utc if first else None
            ),
            second_process_started_utc=(
                second.audit.actual_process_started_utc if second else None
            ),
            first_launcher_pid=first.audit.launcher_pid if first else None,
            second_launcher_pid=second.audit.launcher_pid if second else None,
            first_receipt_schema=first.audit.receipt_schema if first else None,
            second_receipt_schema=second.audit.receipt_schema if second else None,
            first_required_checks=summary(first),
            second_required_checks=summary(second),
            elapsed_between_samples_seconds=elapsed,
        )

    if primary is None:
        return result("unconfigured", "primary_renderer_not_configured")
    selected_id, config = primary
    if (
        not config.attestation_url
        or not config.attestation_token_env
        or not os.environ.get(config.attestation_token_env)
    ):
        return result("unconfigured", "protected_pc_b_attestation_not_configured")
    try:
        first = owner_probe(selected_id, config)
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError):
        return result("unavailable", "first_authenticated_owner_probe_unavailable")
    first_now = now()
    if first.node_id != selected_id or not _valid_owner(first.audit, current=first_now):
        return result("blocked", "first_owner_identity_unverified")
    measured_at = monotonic()
    sleep(gap_seconds)
    elapsed = max(0.0, monotonic() - measured_at)
    if elapsed + 0.000001 < gap_seconds:
        return result("blocked", "independent_sampling_gap_not_observed")
    try:
        second = owner_probe(selected_id, config)
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError):
        return result("unavailable", "second_authenticated_owner_probe_unavailable")
    second_now = now()
    if second.node_id != selected_id or not _valid_owner(second.audit, current=second_now):
        return result("blocked", "second_owner_identity_unverified")
    previous, latest = first.audit, second.audit
    # Never let a replayed cached response pass as a second live observation.
    time_delta = (latest.captured_utc - previous.captured_utc).total_seconds()
    if time_delta < gap_seconds * 0.75:
        return result("blocked", "remote_owner_snapshot_not_advanced")
    if (
        previous.actual_listener_pid != latest.actual_listener_pid
        or previous.actual_process_started_utc != latest.actual_process_started_utc
        or previous.launcher_pid != latest.launcher_pid
        or previous.receipt_schema != latest.receipt_schema
        or previous.scheduler_state != latest.scheduler_state
    ):
        return result("blocked", "renderer_identity_changed_between_snapshots")
    return result("consistent_samples", "same_renderer_identity_at_two_observations")
