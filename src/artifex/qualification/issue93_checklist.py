"""Single PC-A read-only Issue #93 evidence checklist.

Aggregates three independent existing evidence paths. Even agreement of all
three is an observational correlation, NEVER real-host acceptance or permission
to change a live renderer.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.qualification.owner_handoff import (
    PCBOwnerCorrelation,
    correlate_live_pc_b_owner_report,
)
from artifex.qualification.owner_pair_review import (
    OwnerPairReview,
    review_owner_pair_evidence,
)
from artifex.qualification.survival_review import (
    PCBSurvivalEvidenceReview,
    review_live_pc_b_survival_evidence,
)

State = Literal["observations_correlated", "needs_evidence", "conflict", "unconfigured"]
CheckState = Literal["observed", "missing", "conflict"]


class Issue93Check(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal[
        "isolated_launcher_and_live_owner",
        "two_time_separated_owner_samples",
        "natural_exit_survival_trace",
        "cross_evidence_listener_identity",
    ]
    state: CheckState
    reason: str


class Issue93EvidenceChecklist(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    checked_utc: datetime
    node_id: str | None
    status: State
    checks: tuple[Issue93Check, ...]
    observed_checks: int
    missing_checks: int
    conflicting_checks: int
    current_comfyui_listener_pid: int | None = None
    owner_readiness_status: str
    saved_owner_pair_status: str
    natural_exit_trace_status: str
    next_safe_actions: tuple[str, ...]
    # Correlation is not historical authenticity or independent real GPU proof.
    historical_supervisor_exit_authenticated: Literal[False] = False
    native_pc_b_launcher_accepted: Literal[False] = False
    uninterrupted_comfyui_survival_qualified: Literal[False] = False
    reattachment_qualified: Literal[False] = False
    real_gpu_qualified: Literal[False] = False
    fourteen_stage_qualification_proven: Literal[False] = False
    eight_hour_soak_proven: Literal[False] = False
    issue_93_closure_authorized: Literal[False] = False
    issue_40_closure_authorized: Literal[False] = False
    production_qualified: Literal[False] = False
    services_mutated: Literal[False] = False
    task_actions_executed: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False


_ACTIONS = {
    "isolated_launcher_and_live_owner": (
        "Check protected PC-B attestation and its cached no-GPU Python "
        "launcher fixture plus read-only owner audit; do not touch live ComfyUI."
    ),
    "two_time_separated_owner_samples": (
        "On PC-A, run 'uv run artifex qualify owner-pair-check "
        "--config config/local.yaml --save --json' and review the newest report."
    ),
    "natural_exit_survival_trace": (
        "Check the PC-B independent survival observer and its immutable trace; "
        "wait for a naturally occurring supervisor exit; never force one."
    ),
    "cross_evidence_listener_identity": (
        "Inspect mismatched PC-B node/PID/creation-time evidence on PC-A and "
        "PC-B read-only; do not adopt, kill, or restart ComfyUI."
    ),
}
_ALWAYS_PENDING = (
    "Verify the physical PC-B native launcher and real managed ComfyUI "
    "ancestry using protected, on-host read-only diagnostics.",
    "Authenticate the actual historical supervisor-loss observation "
    "independently; a saved/cached trace and its SHA do not prove its origin.",
    "Witness safe reattachment after a natural supervisor exit without "
    "restarting or interrupting the existing ComfyUI process.",
    "Complete real PC-A/PC-B GPU production, all fourteen stages and "
    "eight-hour unattended resource/quality qualification (Issue #40).",
)


def compile_issue93_checklist(
    settings: ArtifexSettings, *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    owner_fetch: Callable[[ArtifexSettings], PCBOwnerCorrelation] = (
        correlate_live_pc_b_owner_report
    ),
    pair_fetch: Callable[[ArtifexSettings], OwnerPairReview] = (
        review_owner_pair_evidence
    ),
    survival_fetch: Callable[[ArtifexSettings], PCBSurvivalEvidenceReview] = (
        review_live_pc_b_survival_evidence
    ),
) -> Issue93EvidenceChecklist:
    """Only existing authenticated reads; no remote process or task actions."""
    primary = settings.render_nodes.primary_node()
    node = primary[0] if primary else None
    samples: dict[str, object | None] = {
        "owner": None, "pair": None, "survival": None,
    }
    statuses: dict[str, str] = dict.fromkeys(samples, "not_observed")
    enabled = bool(
        primary is not None
        and primary[1].attestation_url
        and primary[1].attestation_token_env
        and os.environ.get(primary[1].attestation_token_env)
    )
    if enabled:
        for key, fetch in (
            ("owner", owner_fetch), ("pair", pair_fetch),
            ("survival", survival_fetch),
        ):
            try:
                sample = fetch(settings)
                samples[key] = sample
                statuses[key] = sample.status
            except (OSError, RuntimeError, TypeError, ValueError):
                statuses[key] = "unavailable"

    owner = samples["owner"]
    pair = samples["pair"]
    survival = samples["survival"]
    owner_ok = (
        isinstance(owner, PCBOwnerCorrelation)
        and owner.status == "correlated_read_only"
        and owner.source_mode == "authenticated_remote"
        and owner.remote_bearer_checked
        and owner.node_id == node
        and owner.live_listener_pid is not None
    )
    pair_ok = (
        isinstance(pair, OwnerPairReview)
        and pair.status == "correlated_read_only"
        and pair.remote_bearer_checked
        and pair.saved_required_checks_consistent
        and pair.saved_identity_consistent
        and pair.live_pc_b_owner_correlated
        and pair.node_id == node
        and pair.current_listener_pid is not None
    )
    survival_ok = (
        isinstance(survival, PCBSurvivalEvidenceReview)
        and survival.status == "replayed_and_live_identity_matched"
        and survival.source_mode == "bearer_remote"
        and survival.remote_bearer_checked
        and survival.trace_replayed
        and survival.node_id == node
        and survival.current_listener_pid is not None
    )
    values: tuple[tuple[str, bool, str], ...] = (
        ("isolated_launcher_and_live_owner", owner_ok, statuses["owner"]),
        ("two_time_separated_owner_samples", pair_ok, statuses["pair"]),
        ("natural_exit_survival_trace", survival_ok, statuses["survival"]),
    )
    observed_pids: list[int] = []
    if owner_ok and isinstance(owner, PCBOwnerCorrelation):
        assert owner.live_listener_pid is not None
        observed_pids.append(owner.live_listener_pid)
    if pair_ok and isinstance(pair, OwnerPairReview):
        assert pair.current_listener_pid is not None
        observed_pids.append(pair.current_listener_pid)
    if survival_ok and isinstance(survival, PCBSurvivalEvidenceReview):
        assert survival.current_listener_pid is not None
        observed_pids.append(survival.current_listener_pid)
    conflict = len(set(observed_pids)) > 1
    all_three = owner_ok and pair_ok and survival_ok
    # Never announce cross-source agreement from just a single observation.
    cross_ok = all_three and not conflict
    checks = tuple(
        Issue93Check(
            name=name, state="observed" if ok else "missing", reason=reason,
        )
        for name, ok, reason in values
    ) + (
        Issue93Check(
            name="cross_evidence_listener_identity",
            state="conflict" if conflict else "observed" if cross_ok else "missing",
            reason=(
                "authenticated_current_listener_pid_conflict" if conflict
                else "three_observations_agree_but_historical_origin_unproven"
                if cross_ok else "insufficient_independent_observations"
            ),
        ),
    )
    missing = sum(c.state == "missing" for c in checks)
    conflicting = sum(c.state == "conflict" for c in checks)
    checked_at = now()
    state: State = (
        "unconfigured" if not enabled else
        "conflict" if conflict else
        "observations_correlated" if cross_ok else
        "needs_evidence"
    )
    next_steps = tuple(_ACTIONS[c.name] for c in checks if c.state != "observed")
    return Issue93EvidenceChecklist(
        checked_utc=checked_at, node_id=node, status=state,
        checks=checks,
        observed_checks=4 - missing - conflicting,
        missing_checks=missing,
        conflicting_checks=conflicting,
        current_comfyui_listener_pid=(
            observed_pids[0] if observed_pids and not conflict else None
        ),
        owner_readiness_status=statuses["owner"],
        saved_owner_pair_status=statuses["pair"],
        natural_exit_trace_status=statuses["survival"],
        next_safe_actions=(*next_steps, *_ALWAYS_PENDING),
    )
