"""Single PC-A read-only field preflight across installation, Issue #93 and #40.

The three existing probes are executed sequentially to avoid concurrent
native PC-B CIM reads. Never issue a GPU prompt, create a qualification session,
change services, or infer physical-PC provenance from CI/saved data.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.qualification.handoff import (
    QualificationHandoff,
    inspect_qualification_handoff,
)
from artifex.qualification.issue93_checklist import (
    Issue93EvidenceChecklist,
    compile_issue93_checklist,
)
from artifex.qualification.issue93_field_plan import (
    Issue93FieldPlan,
    compile_issue93_field_plan,
)
from artifex.qualification.native_creation_time import same_native_creation_instant
from artifex.qualification.pair_installation import (
    TwoPCInstallationAudit,
    inspect_two_pc_installation,
)

FieldStatus = Literal[
    "observational_gates_aligned", "needs_evidence", "identity_conflict",
]


class NativeFieldPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    observed_utc: datetime
    status: FieldStatus
    node_id: str | None
    installation_status: str
    issue93_status: str
    handoff_state: str
    installation_blockers: tuple[str, ...]
    issue93_unresolved: tuple[str, ...]
    handoff_blockers: tuple[str, ...]
    probe_errors: tuple[str, ...]
    cross_probe_identity_consistent: bool
    recommended_field_plan: Issue93FieldPlan | None
    next_safe_argv: tuple[str, ...] | None
    # Even all-green observational probes are NOT a physical acceptance gate.
    commands_executed_by_report: Literal[False] = False
    local_config_written: Literal[False] = False
    service_mutated: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    natural_supervisor_exit_authenticated: Literal[False] = False
    native_comfyui_reattachment_qualified: Literal[False] = False
    fourteen_stage_qualification_verified: Literal[False] = False
    eight_hour_gpu_soak_verified: Literal[False] = False
    issue_93_closure_authorized: Literal[False] = False
    issue_40_closure_authorized: Literal[False] = False
    production_qualified: Literal[False] = False


def compile_native_field_preflight(
    *, controller_config: Path, renderer_config: Path,
    installation: TwoPCInstallationAudit | None,
    issue93: Issue93EvidenceChecklist | None,
    handoff: QualificationHandoff | None,
    probe_errors: tuple[str, ...] = (),
    observed_utc: datetime | None = None,
) -> NativeFieldPreflight:
    """Correlate results without elevating any observation to real host proof."""
    install_state = installation.status if installation else "unavailable"
    issue_state = issue93.status if issue93 else "unavailable"
    handoff_state = handoff.state if handoff else "unavailable"
    names = {
        node for node in (
            installation.pc_b_node_id if installation else None,
            issue93.node_id if issue93 else None,
            handoff.overview.readiness.primary_node_id if handoff else None,
        )
        if node is not None
    }
    node_conflict = len(names) > 1
    identity_conflict = node_conflict or issue_state == "conflict"
    identity_correlated = False
    if issue93 is not None and handoff is not None:
        owner = handoff.overview.pc_b_owner_evidence
        if (
            issue93.status == "observations_correlated"
            and handoff.authenticated_owner_evidence_correlated
            and owner is not None
        ):
            identity_correlated = bool(
                issue93.current_comfyui_listener_pid is not None
                and issue93.current_comfyui_listener_pid == owner.live_listener_pid
                and same_native_creation_instant(
                    issue93.current_comfyui_started_utc,
                    owner.live_listener_started_utc,
                )
                and issue93.current_pc_b_hostname is not None
                and issue93.current_pc_b_hostname.casefold() == (
                    owner.live_host or ""
                ).casefold()
                and owner.node_id == issue93.node_id
            )
            if not identity_correlated:
                identity_conflict = True
    if (
        not identity_conflict
        and installation is not None
        and issue93 is not None
        and handoff is not None
        and not probe_errors
        and install_state == "ready_for_passive_monitoring"
        and issue_state == "observations_correlated"
        and handoff_state in {"session_start_candidate", "saved_session_review"}
        and identity_correlated
    ):
        state: FieldStatus = "observational_gates_aligned"
    else:
        state = "identity_conflict" if identity_conflict else "needs_evidence"

    plan = (
        compile_issue93_field_plan(
            issue93, controller_config=controller_config,
            renderer_config=renderer_config,
        )
        if issue93 is not None else None
    )
    next_argv = (
        # Do not promote a session-start action if the other observations
        # are unavailable or inconsistent.
        handoff.advisory_argv
        if state == "observational_gates_aligned" and handoff is not None
        else (
            plan.steps[0].argv if plan is not None and plan.steps else
            ("uv", "run", "artifex", "onboard", "first-run",
             "--role", "controller", "--config", str(controller_config), "--json")
        )
    )
    return NativeFieldPreflight(
        observed_utc=observed_utc or datetime.now(UTC), status=state,
        node_id=next(iter(names)) if len(names) == 1 else None,
        installation_status=install_state,
        issue93_status=issue_state,
        handoff_state=handoff_state,
        installation_blockers=installation.blockers if installation else (),
        issue93_unresolved=(
            tuple(check.name for check in issue93.checks
                  if check.state != "observed")
            if issue93 is not None else ()
        ),
        handoff_blockers=handoff.blockers if handoff else (),
        probe_errors=probe_errors,
        cross_probe_identity_consistent=identity_correlated and not identity_conflict,
        recommended_field_plan=plan,
        next_safe_argv=next_argv,
    )


async def inspect_native_field_preflight(
    settings: ArtifexSettings, *,
    controller_config: Path,
    renderer_config: Path = Path("config/render-node.yaml"),
    installation_fn: Callable[..., TwoPCInstallationAudit] = inspect_two_pc_installation,
    issue93_fn: Callable[..., Issue93EvidenceChecklist] = compile_issue93_checklist,
    handoff_fn: Callable[..., Awaitable[QualificationHandoff]] = (
        inspect_qualification_handoff
    ),
) -> NativeFieldPreflight:
    """Serially run only existing GETs/local reads and preserve partial failures.

    A failed network/scheduler probe must not hide the other independent
    diagnostics. Exceptions are returned by class name only: never leak
    private paths, LAN URLs, response bodies or Bearer token values.
    """
    installation: TwoPCInstallationAudit | None = None
    issue93: Issue93EvidenceChecklist | None = None
    handoff: QualificationHandoff | None = None
    errors: list[str] = []
    try:
        installation = await asyncio.to_thread(
            installation_fn, settings, controller_config=controller_config,
        )
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError) as exc:
        errors.append(f"installation:{type(exc).__name__}")
    try:
        issue93 = await asyncio.to_thread(issue93_fn, settings)
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError) as exc:
        errors.append(f"issue93:{type(exc).__name__}")
    try:
        handoff = await handoff_fn(
            settings, controller_config=controller_config,
            renderer_config=renderer_config,
            pc_b_owner_live=True, pc_b_survival_live=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError) as exc:
        errors.append(f"handoff:{type(exc).__name__}")
    return compile_native_field_preflight(
        controller_config=controller_config,
        renderer_config=renderer_config,
        installation=installation, issue93=issue93, handoff=handoff,
        probe_errors=tuple(errors),
    )
