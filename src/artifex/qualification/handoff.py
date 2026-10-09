"""Read-only PC-A qualification handoff combining distinct existing gates.

An environment observation and a workflow dependency check both must pass
before we *suggest* manually starting a native qualification session. Nothing
here authorizes GPU submission, PC-B restart, stage PASS or production rollout.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.deployment import DeploymentReport, verify_deployment
from artifex.qualification.overview import (
    QualificationOverview,
    compile_qualification_overview,
)

HandoffState = Literal["blocked", "session_start_candidate", "saved_session_review"]


class QualificationHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    state: HandoffState
    environment_ready: bool
    deployment_workflows_ready: bool
    authenticated_owner_evidence_required: bool = False
    authenticated_owner_evidence_correlated: bool = False
    authenticated_owner_evidence_state: str | None = None
    passive_survival_trace_status: str | None = None
    passive_survival_trace_authenticated: Literal[False] = False
    can_suggest_qualification_start: bool
    saved_session_id: str | None
    saved_session_not_assumed_active: Literal[True] = True
    blockers: tuple[str, ...]
    remaining_native_evidence: tuple[str, ...]
    next_action: str
    advisory_argv: tuple[str, ...] | None
    overview: QualificationOverview
    deployment: DeploymentReport
    config_written: Literal[False] = False
    shell_commands_executed: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    services_mutated: Literal[False] = False
    renderer_start_authorized: Literal[False] = False
    actual_eight_hour_soak_verified: Literal[False] = False
    production_qualified: Literal[False] = False


def compile_handoff(
    overview: QualificationOverview,
    deployment: DeploymentReport,
    *,
    controller_config: Path,
    expected_workflow_ids: tuple[str, ...],
    require_authenticated_owner: bool = False,
) -> QualificationHandoff:
    """Fail closed if one gate is missing, contradicts the other or is stale."""
    if deployment.role != "controller":
        raise ValueError("PC-A handoff must use a controller deployment report")
    if deployment.smoke is not None or any(
        check.name == "render_smoke" for check in deployment.checks
    ):
        raise ValueError("PC-A handoff must not consume GPU smoke evidence")
    blocking = tuple(
        check.name for check in deployment.checks if check.blocking and not check.ready
    )
    checked_controller = any(
        check.name.startswith("controller:") for check in deployment.checks
    )
    # The two configured templates may coincide; compare the exact distinct
    # workflow IDs, not an arbitrary number of responses.
    workflow_checks = tuple(
        check for check in deployment.checks
        if check.name.startswith("workflow:")
    )
    unique_workflows = {check.name for check in workflow_checks}
    expected = {"workflow:" + name for name in expected_workflow_ids}
    workflow_ready = (
        bool(expected)
        and unique_workflows == expected
        and len(unique_workflows) == len(workflow_checks)
        and all(check.ready for check in workflow_checks)
    )
    blocked: list[str] = []
    if not overview.environment_ready or not overview.action_plan.snapshot_fresh:
        blocked.append("pc_a_pc_b_live_readiness")
    if not deployment.ready or blocking:
        blocked.extend(f"deployment:{name}" for name in blocking)
        if not blocking:
            blocked.append("deployment:inconsistent_ready_flag")
    if not checked_controller:
        blocked.append("deployment:controller_preflight_missing")
    if not workflow_ready:
        blocked.append("deployment:workflow_requirements_missing_or_failed")
    # Issue #93: a valid file or a good owner audit alone cannot satisfy the
    # additional authenticated remote PC-B evidence gate. Fail closed on
    # missing, inconsistent, stale or mismatched proof. A broken endpoint
    # must never silently fall back to legacy advisory session readiness.
    evidence = overview.pc_b_owner_evidence
    correlated_owner = bool(
        evidence is not None
        and evidence.status == "correlated_read_only"
        and evidence.source_mode == "authenticated_remote"
        and evidence.remote_bearer_checked is True
        and evidence.node_id == overview.readiness.primary_node_id
        and evidence.node_id is not None
        and evidence.saved_listener_pid is not None
        and evidence.saved_listener_pid == evidence.live_listener_pid
        and evidence.saved_host is not None
        and evidence.saved_host.casefold() == (evidence.live_host or "").casefold()
        and evidence.checked_utc.tzinfo is not None
        and overview.captured_utc.tzinfo is not None
        and abs(overview.captured_utc - evidence.checked_utc) <= timedelta(minutes=2)
        and evidence.file_source_authenticated is False
        and evidence.stage_pass_registered is False
        and evidence.production_qualified is False
        and evidence.renderer_restart_authorized is False
    )
    if require_authenticated_owner and not correlated_owner:
        blocked.append("pc_b_authenticated_owner_evidence_missing_or_unverified")
    # A configured PC-B attestation/PID is not in itself independent evidence
    # that Windows virtualenv launcher identity and child survival were
    # verified on the operator's actual machine (Issue #93).
    independent: tuple[str, ...] = (
        (
            "Confirm actual native PC-B launcher, Task Scheduler and child PID "
            "survival on the owner's machine (Issue #93)."
        ),
        (
            "Complete independent real-PC 14-stage acceptance including live "
            "GPU recovery, archive reproduction and eight-hour soak (Issue #40)."
        ),
    )
    if evidence is not None:
        independent = (*independent, *evidence.remaining_real_machine_evidence)
    survival = overview.pc_b_survival_review
    if survival is not None:
        independent = (
            *independent,
            (
                "Independently establish real PC-B provenance of the natural "
                "supervisor-loss event (a copied JSON replay is not machine proof)."
            ),
        )
        if survival.status != "replayed_and_live_identity_matched":
            blocked.append("pc_b_passive_survival_trace_not_live_correlated")
    selected = overview.session_id
    command: tuple[str, ...] | None
    if blocked:
        state: HandoffState = "blocked"
        action = (
            "Fix the detailed live readiness and workflow dependency blockers "
            "shown in this report. Do not restart or adopt existing ComfyUI."
        )
        command = None
    elif selected is None:
        state = "session_start_candidate"
        action = (
            "Read-only PC-A and protected PC-B preflight plus the configured "
            "ComfyUI workflow requirements agree. An operator may now START "
            "a qualification session; this does not start GPU generation."
        )
        command = (
            "uv", "run", "artifex", "qualify", "start",
            "--config", str(controller_config), "--json",
        )
    else:
        state = "saved_session_review"
        action = (
            "A saved qualification session exists; it is not assumed active "
            "or independently verified. Review missing evidence and continue "
            "the strict manual/automatic collection process."
        )
        command = overview.next_safe_command
    return QualificationHandoff(
        state=state,
        environment_ready=overview.environment_ready,
        deployment_workflows_ready=workflow_ready,
        authenticated_owner_evidence_required=require_authenticated_owner,
        authenticated_owner_evidence_correlated=correlated_owner,
        authenticated_owner_evidence_state=evidence.status if evidence else None,
        passive_survival_trace_status=survival.status if survival else None,
        can_suggest_qualification_start=not blocked,
        saved_session_id=selected,
        blockers=tuple(dict.fromkeys(blocked)),
        remaining_native_evidence=tuple(dict.fromkeys(independent)),
        next_action=action,
        advisory_argv=command,
        overview=overview, deployment=deployment,
    )


async def inspect_qualification_handoff(
    settings: ArtifexSettings,
    *,
    controller_config: Path,
    session_id: str | None = None,
    renderer_config: Path = Path("config/render-node.yaml"),
    pc_b_owner_live: bool = True,
    pc_b_survival_report: Path | None = None,
    overview_fn: Callable[..., QualificationOverview] = compile_qualification_overview,
    deployment_fn: Callable[..., Awaitable[DeploymentReport]] = verify_deployment,
) -> QualificationHandoff:
    """Perform bounded existing read-only diagnostics; no render smoke ever."""
    # Run the synchronous authenticated/remote owner probes off the event
    # loop, then workflow checks sequentially to reduce PC-B probe load.
    overview = await asyncio.to_thread(
        overview_fn,
        settings,
        session_id=session_id,
        controller_config=controller_config,
        renderer_config=renderer_config,
        pc_b_owner_live=pc_b_owner_live,
        pc_b_survival_report=pc_b_survival_report,
    )
    deployment = await deployment_fn(
        settings, role="controller", require_autostart=False, render_smoke=False,
    )
    return compile_handoff(
        overview, deployment, controller_config=controller_config,
        expected_workflow_ids=(
            settings.comfyui.default_template,
            settings.production.repair_workflow_template,
        ),
        require_authenticated_owner=pc_b_owner_live,
    )
