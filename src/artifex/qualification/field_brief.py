"""One safe PC-A field brief from the existing Issue #93 and 14-stage observers.

This module deliberately does not initiate a qualification session, start GPU
work, send Discord messages, or treat point-in-time observations as production
acceptance. Real PC-B physical evidence still requires independent validation.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings
from artifex.qualification.issue93_checklist import (
    Issue93EvidenceChecklist,
    compile_issue93_checklist,
)
from artifex.qualification.issue93_field_plan import (
    Issue93FieldPlan,
    compile_issue93_field_plan,
)
from artifex.qualification.native_creation_time import native_creation_instant
from artifex.qualification.overview import (
    QualificationOverview,
    compile_qualification_overview,
)
from artifex.qualification.pc_b_remote_field_gates import (
    RemotePCBPassiveFieldGates,
    inspect_remote_pc_b_passive_gates,
)


class FieldRoleTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["pc_a", "pc_b", "discord"]
    id: str
    safety: str
    next_action: str
    # These are advisory only. No command is invoked by this report.
    executed: Literal[False] = False


class QualificationFieldBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checked_utc: datetime
    session_id: str | None
    session_selection: Literal["explicit", "latest_saved", "none"]
    node_id: str | None
    issue_93_status: str
    missing_issue_93_checks: int = Field(ge=0)
    conflicting_issue_93_checks: int = Field(ge=0)
    owner_live_status: str
    survival_live_status: str
    saved_monitor_status: str
    cross_observation_identity: Literal["not_comparable", "match", "conflict"]
    # Status-only failure diagnostics: never persist remote exception text,
    # headers, token values or incomplete raw owner/survival trace contents.
    failed_live_probes: tuple[Literal["owner", "survival"], ...] = ()
    partial_live_diagnostics: bool = False
    environment_ready: bool
    recorded_pass_stages: int = Field(ge=0)
    recorded_skipped_stages: int = Field(ge=0)
    unresolved_stages: int = Field(ge=0)
    next_priority: str
    issue_93_field_plan: Issue93FieldPlan
    next_role_tasks: tuple[FieldRoleTask, ...]
    pc_b_remote_passive_gates: RemotePCBPassiveFieldGates | None = None
    # Correlation, saved stages, and "ready" diagnostics are not independent
    # physical-host proof or authorization to start a real GPU job.
    authenticated_historical_supervisor_loss_proven: Literal[False] = False
    live_comfyui_survival_qualified: Literal[False] = False
    real_gpu_qualified: Literal[False] = False
    fourteen_stages_independently_verified: Literal[False] = False
    eight_hour_soak_qualified: Literal[False] = False
    issue_93_closed: Literal[False] = False
    issue_40_closed: Literal[False] = False
    production_qualified: Literal[False] = False
    service_or_task_mutation: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    discord_messages_sent: Literal[False] = False
    commands_executed: Literal[False] = False


def _cross_observation_identity(
    issue93: Issue93EvidenceChecklist,
    overview: QualificationOverview,
) -> Literal["not_comparable", "match", "conflict"]:
    """Require agreement across the *separately acquired* current identities.

    Positive statuses on independent reads do not prove they refer to the
    same process. Compare (PID, UTC creation instant, host), not PID alone.
    """
    owner = overview.pc_b_owner_evidence
    survival = overview.pc_b_survival_review
    if (
        issue93.status != "observations_correlated"
        or owner is None or owner.status != "correlated_read_only"
        or not owner.remote_bearer_checked
        or survival is None
        or survival.status != "replayed_and_live_identity_matched"
        or not survival.remote_bearer_checked
    ):
        return "not_comparable"
    raw = (
        (issue93.current_comfyui_listener_pid,
         issue93.current_comfyui_started_utc,
         issue93.current_pc_b_hostname),
        (owner.live_listener_pid, owner.live_listener_started_utc,
         owner.live_host),
        (survival.current_listener_pid, survival.current_listener_started_utc,
         survival.live_host),
    )
    identities: list[tuple[int, datetime, str]] = []
    for pid, timestamp, host in raw:
        started = native_creation_instant(timestamp)
        if pid is None or pid < 1 or started is None or not host or not host.strip():
            return "not_comparable"
        identities.append((pid, started, host.strip().casefold()))
    return "match" if len(set(identities)) == 1 else "conflict"


def compile_field_brief(
    issue93: Issue93EvidenceChecklist,
    overview: QualificationOverview,
    *,
    controller_config: Path,
    renderer_config: Path,
    checked_utc: datetime | None = None,
    pc_b_remote_gates: RemotePCBPassiveFieldGates | None = None,
) -> QualificationFieldBrief:
    """Summarize already observed data; never re-run another remote probe."""
    field_plan = compile_issue93_field_plan(
        issue93,
        controller_config=controller_config,
        renderer_config=renderer_config,
    )
    owner = overview.pc_b_owner_evidence
    survival = overview.pc_b_survival_review
    owner_status = owner.status if owner is not None else "not_requested"
    survival_status = survival.status if survival is not None else "not_requested"
    cross_identity = _cross_observation_identity(issue93, overview)
    failed: list[Literal["owner", "survival"]] = []
    if owner is not None and owner.status == "unavailable":
        failed.append("owner")
    if survival is not None and survival.status in {
        "remote_unavailable", "replayed_live_unavailable",
    }:
        failed.append("survival")
    next_tasks: list[FieldRoleTask] = []
    # One role-owned unresolved task per machine/service keeps the brief small.
    # The complete worklist remains accessible through qualify overview.
    roles: tuple[Literal["pc_a", "pc_b", "discord"], ...] = (
        "pc_a", "pc_b", "discord",
    )
    for role in roles:
        item = next(
            (entry for entry in overview.evidence_handoff.items
             if entry.responsible_role == role),
            None,
        )
        if item is not None:
            next_tasks.append(FieldRoleTask(
                role=role, id=item.id, safety=item.safety,
                next_action=item.next_action,
            ))

    if cross_identity == "conflict":
        priority = (
            "Independent authenticated PC-B observations identify different "
            "listener processes or hosts. Read-only inspect PID and creation "
            "time before trusting any historical report; never restart ComfyUI."
        )
    elif issue93.status == "conflict":
        priority = (
            "PC-B evidence shows conflicting listener identities. Review "
            "native PID/creation-time and source timestamps read-only; "
            "never restart or adopt ComfyUI to make reports match."
        )
    elif issue93.status == "unconfigured":
        priority = (
            "Configure protected PC-B attestation before real-machine evidence "
            "collection. Never guess an IP or token."
        )
    elif not overview.environment_ready:
        priority = overview.next_priority
    elif failed:
        priority = (
            "Some authenticated PC-B read-only probes are unavailable. "
            "Inspect failed_live_probes and network/attestation configuration; "
            "never restart a busy ComfyUI process to force an observation."
        )
    elif issue93.status != "observations_correlated":
        priority = (
            "Review missing Issue #93 checks in issue_93_field_plan. "
            "Saved evidence is not an authenticated supervisor-loss event."
        )
    elif owner_status != "correlated_read_only" or (
        survival_status != "replayed_and_live_identity_matched"
    ):
        priority = (
            "Current PC-B evidence sources disagree or are unavailable. "
            "Review the live read-only owner/survival reports."
        )
    else:
        priority = overview.next_priority

    return QualificationFieldBrief(
        checked_utc=checked_utc or datetime.now(UTC),
        session_id=overview.session_id,
        session_selection=overview.session_selection,
        node_id=issue93.node_id,
        issue_93_status=issue93.status,
        missing_issue_93_checks=issue93.missing_checks,
        conflicting_issue_93_checks=issue93.conflicting_checks,
        owner_live_status=owner_status,
        survival_live_status=survival_status,
        saved_monitor_status=overview.pc_b_saved_monitor.status,
        cross_observation_identity=cross_identity,
        failed_live_probes=tuple(failed),
        partial_live_diagnostics=bool(failed),
        environment_ready=overview.environment_ready,
        recorded_pass_stages=overview.recorded_pass_count,
        recorded_skipped_stages=overview.recorded_skipped_count,
        unresolved_stages=overview.unresolved_stage_count,
        next_priority=priority,
        issue_93_field_plan=field_plan,
        next_role_tasks=tuple(next_tasks),
        pc_b_remote_passive_gates=pc_b_remote_gates,
    )


def collect_field_brief(
    settings: ArtifexSettings,
    *,
    controller_config: Path,
    renderer_config: Path,
    session_id: str | None = None,
) -> QualificationFieldBrief:
    """One PC-A command collects two independent observational diagnostics.

    Each source is queried through its existing Bearer-authenticated, bounded,
    GET-only paths. The independent reads are intentionally not claimed to be
    a single atomic PC-B state; neither source can promote a stage PASS.
    """
    issue93 = compile_issue93_checklist(settings)
    overview = compile_qualification_overview(
        settings,
        controller_config=controller_config,
        renderer_config=renderer_config,
        session_id=session_id,
        pc_b_owner_live=settings.render_nodes.primary_node() is not None,
        pc_b_survival_live=settings.render_nodes.primary_node() is not None,
    )
    # This final check uses only the PC-B API's existing protected
    # installation report and background-cached no-GPU launcher evidence.
    # It does not run the PC-B local fixture as part of a request.
    remote_gates = inspect_remote_pc_b_passive_gates(settings)
    return compile_field_brief(
        issue93, overview,
        controller_config=controller_config,
        renderer_config=renderer_config,
        pc_b_remote_gates=remote_gates,
    )
