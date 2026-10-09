"""Reconcile an old action plan with fresh, allowlisted read-only probes.

No action-plan argv is ever executed. The only active work is the already
existing PC-A controller preflight and authenticated PC-B ownership GET
inside diagnose_qualification_readiness. A PC-B-local preflight cannot be
remotely run and must never be represented as completed.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.qualification.readiness_action_plan import (
    QualificationActionPlan,
    RemediationStep,
    compile_qualification_action_plan,
)
from artifex.qualification.readiness_diagnostics import (
    QualificationReadiness,
    diagnose_qualification_readiness,
)

ObservationState = Literal[
    "observed_live", "unavailable", "requires_local_pc_b", "not_executed"
]
CheckStatus = Literal["pass", "fail", "unknown", "missing"]

# These are inspection paths, not permitted subprocess commands. Only the
# existing in-process readiness diagnostic may run.
PC_A_READ_ONLY = frozenset({
    "pc-a-preflight", "pc-b-connectivity", "refresh-readiness",
    "pc-a-remote-renderer-safety",
})
PC_B_OWNER_READ_ONLY = frozenset({"pc-b-owner"})
PC_B_LOCAL_READ_ONLY = frozenset({"pc-b-inventory"})


class ReadOnlyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    state: ObservationState
    check_names: tuple[str, ...]
    explanation: str
    cli_commands_executed: Literal[False] = False


class ReadinessChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    check_name: str
    previous: CheckStatus
    current: CheckStatus
    kind: Literal["improved", "regressed", "changed", "unavailable"]


class ReadOnlyReconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    before_source: Literal["saved", "live"]
    previous_captured_utc: datetime
    current_captured_utc: datetime
    before_environment_ready: bool
    latest_environment_ready: bool
    observations: tuple[ReadOnlyObservation, ...]
    changes: tuple[ReadinessChange, ...]
    latest_readiness: QualificationReadiness
    updated_plan: QualificationActionPlan
    # All production claims permanently false, even if every probe passes.
    production_qualified: Literal[False] = False
    gpu_soak_qualified: Literal[False] = False
    mutated_services: Literal[False] = False
    arbitrary_plan_commands_executed: Literal[False] = False


def _status_index(
    report: QualificationReadiness,
) -> dict[str, CheckStatus]:
    """Use worst status for duplicate keys, never let duplicates hide failures."""
    ranks: dict[CheckStatus, int] = {
        "pass": 0, "unknown": 1, "fail": 2, "missing": 3,
    }
    results: dict[str, CheckStatus] = {}
    for item in report.checks:
        key = f"{item.target}:{item.name}"
        if key not in results or ranks[item.status] > ranks[results[key]]:
            results[key] = item.status
    return results


def _changes(
    older: QualificationReadiness,
    newer: QualificationReadiness,
    *,
    has_prior: bool,
) -> tuple[ReadinessChange, ...]:
    if not has_prior:
        return ()
    before = _status_index(older)
    after = _status_index(newer)
    out: list[ReadinessChange] = []
    severity = {"pass": 0, "unknown": 1, "fail": 2}
    for key in sorted(before.keys() | after.keys()):
        a: CheckStatus = before.get(key, "missing")
        z: CheckStatus = after.get(key, "missing")
        if a == z:
            continue
        kind: Literal["improved", "regressed", "changed", "unavailable"]
        if z == "missing":
            kind = "unavailable"
        elif a == "missing":
            kind = "changed"
        elif severity[z] < severity[a]:
            kind = "improved"
        elif severity[z] > severity[a]:
            kind = "regressed"
        else:
            kind = "changed"
        out.append(ReadinessChange(
            check_name=key, previous=a, current=z, kind=kind,
        ))
    return tuple(out)


def _observed(
    step: RemediationStep,
    latest: QualificationReadiness,
) -> ReadOnlyObservation:
    """Honor only known probe identities, never saved/untrusted argv."""
    keys = _status_index(latest)
    if step.safety != "read_only":
        return ReadOnlyObservation(
            step_id=step.id, state="not_executed",
            check_names=step.blocked_checks,
            explanation="Review/real-machine stage cannot run automatically.",
        )
    if step.id in PC_B_LOCAL_READ_ONLY:
        return ReadOnlyObservation(
            step_id=step.id, state="requires_local_pc_b",
            check_names=step.blocked_checks,
            explanation=(
                "PC-B native render-node preflight requires a local PC-B process. "
                "Remote inventory attestations are not equivalent to that check."
            ),
        )
    if step.id in PC_B_OWNER_READ_ONLY:
        visible = (
            "pc_b:owner:overall" in keys
            and "pc_b:owner:freshness" in keys
        )
        return ReadOnlyObservation(
            step_id=step.id,
            state="observed_live" if visible else "unavailable",
            check_names=step.blocked_checks,
            explanation=(
                "Authenticated PC-B owner audit was sampled by the fixed live "
                "diagnostic; a FAIL status remains a FAIL."
                if visible else
                "Authenticated PC-B owner audit unavailable; no local PC-B command run."
            ),
        )
    if step.id in PC_A_READ_ONLY:
        if step.id == "refresh-readiness":
            visible = bool(latest.checks)
        elif step.id == "pc-a-remote-renderer-safety":
            visible = all(
                f"pc_b:safety:{name}" in keys
                for name in ("overall", "freshness", "three_ports", "pid_consistency")
            )
        else:
            visible = any(
                key.startswith(("pc_a:preflight:", "pc_b:preflight:"))
                for key in keys
            ) and "pc_a:preflight:probe" not in keys
        return ReadOnlyObservation(
            step_id=step.id,
            state="observed_live" if visible else "unavailable",
            check_names=step.blocked_checks,
            explanation=(
                "PC-A in-process read-only readiness probes were evaluated; "
                "FAIL and UNKNOWN remain unresolved."
                if visible else "PC-A live preflight or remote safety could not be fully observed."
            ),
        )
    return ReadOnlyObservation(
        step_id=step.id, state="not_executed",
        check_names=step.blocked_checks,
        explanation=(
            "Unrecognized read-only step is not on the in-process allowlist. "
            "Never execute action-plan argv."
        ),
    )


def reconcile_read_only_checks(
    settings: ArtifexSettings,
    *,
    previous: QualificationReadiness | None = None,
    controller_config: Path = Path("config/local.yaml"),
    renderer_config: Path = Path("config/render-node.yaml"),
    session_id: str | None = None,
    now: datetime | None = None,
    live_probe: Callable[..., QualificationReadiness] = diagnose_qualification_readiness,
) -> ReadOnlyReconciliation:
    """Collect actual current observations once and recompile the remaining work.

    Previous saved JSON influences only the historical diff and what actions
    are listed as previously pending; it never supplies executable commands
    or overwrites current observations. All probes are fixed code paths.
    """
    latest = live_probe(settings, session_id=session_id)
    current = now or datetime.now(UTC)
    initial = previous or latest
    plan_before = compile_qualification_action_plan(
        initial, controller_config=controller_config,
        renderer_config=renderer_config,
        source="saved" if previous is not None else "live",
        now=current,
    )
    new_plan = compile_qualification_action_plan(
        latest, controller_config=controller_config,
        renderer_config=renderer_config, source="live", now=current,
    )
    observations = tuple(
        _observed(step, latest) for step in plan_before.steps
    )
    return ReadOnlyReconciliation(
        before_source="saved" if previous is not None else "live",
        previous_captured_utc=initial.captured_utc,
        current_captured_utc=latest.captured_utc,
        before_environment_ready=initial.environment_ready,
        latest_environment_ready=new_plan.environment_ready_observed,
        observations=observations,
        changes=_changes(initial, latest, has_prior=previous is not None),
        latest_readiness=latest,
        updated_plan=new_plan,
    )
