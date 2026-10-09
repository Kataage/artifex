"""Single safe PC-A overview of live readiness, saved stage status, and next work.

This is observational only. A recorded PASS is NOT a revalidated production
qualification; no remote command, stage registration or renderer lifecycle
operation is executed by this module.
"""
from __future__ import annotations

import re
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.qualification.models import (
    REQUIRED_STAGES,
    QualificationStage,
    QualificationStatus,
)
from artifex.qualification.owner_handoff import (
    PCBOwnerCorrelation,
    correlate_pc_b_owner_report,
)
from artifex.qualification.readiness_action_plan import (
    QualificationActionPlan,
    RemediationStep,
    compile_qualification_action_plan,
)
from artifex.qualification.readiness_diagnostics import (
    QualificationReadiness,
    _read_session,
    diagnose_qualification_readiness,
)
from artifex.qualification.readiness_recheck import _observed

_SESSION_NAME = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
# Do not scan an unbounded foreign / malformed directory tree.
_MAX_SESSION_FOLDERS = 2048
_STAGE_KIND: dict[QualificationStage, Literal[
    "baseline", "existing_pack", "controlled_recovery", "live_soak",
    "real_discord", "archive_reproduction",
]] = {
    QualificationStage.DOCTOR: "baseline",
    QualificationStage.SINGLE_CHARACTER: "existing_pack",
    QualificationStage.LORA_REQUIRED: "existing_pack",
    QualificationStage.DUO: "existing_pack",
    QualificationStage.GROUP: "existing_pack",
    QualificationStage.PUBLIC_MEMBER: "existing_pack",
    QualificationStage.SERIES_CONTINUATION: "existing_pack",
    QualificationStage.FORCED_RETRY: "existing_pack",
    QualificationStage.RESTART_GENERATION: "controlled_recovery",
    QualificationStage.BACKEND_RECOVERY: "controlled_recovery",
    QualificationStage.UNATTENDED_MULTI_PACK: "existing_pack",
    QualificationStage.OVERNIGHT_SOAK: "live_soak",
    QualificationStage.DISCORD_CONTROLS: "real_discord",
    QualificationStage.ARCHIVE_REPRODUCTION: "archive_reproduction",
}
_GUIDANCE = {
    "baseline": "Review the native PC-A doctor and start a new explicit session after repairs.",
    "existing_pack": "Use ordinary Artifex production, then preview persisted evidence with qualify collect; never fabricate a Pack.",
    "controlled_recovery": "Requires a controlled real-machine recovery observation; never restart or terminate an existing GPU job merely to clear this stage.",
    "live_soak": "Requires real eight-hour target-machine GPU/resource observation; simulated or shorter traces do not qualify.",
    "real_discord": "Verify actual delivered Discord controls, or confirm an explicitly disabled Discord configuration.",
    "archive_reproduction": "Reproduce an actual archived output with verified provenance; never claim success from an existing file alone.",
}


StageRecordStatus = Literal["not_inspected", "pending", "pass", "fail", "skipped"]


class StageOverview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    recorded_status: StageRecordStatus
    evidence_kind: str
    next_action: str | None
    independently_revalidated: Literal[False] = False


TriageState = Literal[
    "observed_live", "unavailable", "requires_local_pc_b",
    "not_executed", "operator_review", "real_machine_evidence",
]


class RemediationTriage(BaseModel):
    """Action-plan disposition after one live snapshot; no commands executed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    role: Literal["pc_a", "pc_b"]
    state: TriageState
    blocked_checks: tuple[str, ...]
    next_action: str
    commands_executed: Literal[False] = False


def _triage_step(
    step: RemediationStep, report: QualificationReadiness, *,
    fresh: bool,
) -> RemediationTriage:
    state: TriageState
    if step.safety == "read_only":
        sample = _observed(step, report)
        state = sample.state if fresh else "unavailable"
        if state == "observed_live":
            guidance = (
                "Already sampled from PC-A in the current read-only diagnostic. "
                "Check the FAIL/UNKNOWN values; observation is not resolution."
            )
        elif state == "requires_local_pc_b":
            guidance = (
                "Needs a native PC-B inspection during an appropriate safe window. "
                "Never remotely execute the suggested PC-B command."
            )
        else:
            guidance = (
                "Read-only probe unavailable or not allowlisted. Check configuration "
                "and connectivity; retry only through the fixed PC-A readiness path."
            )
    elif step.safety == "review_required":
        state = "operator_review"
        guidance = step.description
    else:
        state = "real_machine_evidence"
        guidance = step.description
    return RemediationTriage(
        step_id=step.id, role=step.role, state=state,
        blocked_checks=step.blocked_checks, next_action=guidance,
    )


class QualificationOverview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    captured_utc: datetime
    session_id: str | None
    session_selection: Literal["explicit", "latest_saved", "none"]
    auto_collection_binding: Literal["not_checked"] = "not_checked"
    session_stages_are_revalidated: Literal[False] = False
    real_machine_qualification_complete: Literal[False] = False
    actual_eight_hour_soak_verified: Literal[False] = False
    production_qualified: Literal[False] = False
    mutated_services: Literal[False] = False
    commands_executed: Literal[False] = False
    environment_ready: bool
    recorded_pass_count: int
    recorded_skipped_count: int
    unresolved_stage_count: int
    stages: tuple[StageOverview, ...]
    next_priority: str
    next_safe_command: tuple[str, ...] | None
    remediation_triage: tuple[RemediationTriage, ...]
    already_sampled_read_only: tuple[str, ...]
    unavailable_read_only: tuple[str, ...]
    pc_b_local_checks_required: tuple[str, ...]
    operator_review_required: tuple[str, ...]
    readiness: QualificationReadiness
    action_plan: QualificationActionPlan
    pc_b_owner_evidence: PCBOwnerCorrelation | None = None


def latest_saved_session_id(root: Path) -> str | None:
    """Choose latest named file *without* promoting it to an active session.

    The latest session is authoritative, even if blocked/foreign/corrupted:
    never silently fall back to a prior PASS or apparently healthy session.
    """
    path = root.expanduser().absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Qualification evidence location is symlinked")
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError("Qualification evidence location is not a directory")
    ids: list[str] = []
    for entry in path.iterdir():
        if _SESSION_NAME.fullmatch(entry.name):
            ids.append(entry.name)
            if len(ids) > _MAX_SESSION_FOLDERS:
                raise ValueError("Qualification session directory scan limit exceeded")
    if not ids:
        return None
    newest = max(ids)
    session = _read_session(path, newest)
    if session.hostname != socket.gethostname():
        raise ValueError("Latest saved qualification session belongs to another host")
    return newest


def compile_qualification_overview(
    settings: ArtifexSettings,
    *,
    session_id: str | None = None,
    controller_config: Path = Path("config/local.yaml"),
    renderer_config: Path = Path("config/render-node.yaml"),
    pc_b_owner_report: Path | None = None,
    readiness: QualificationReadiness | None = None,
    now: datetime | None = None,
) -> QualificationOverview:
    """One current read-only observation, one plan, no independent PASS."""
    current = now or datetime.now(UTC)
    source: Literal["explicit", "latest_saved", "none"] = "none"
    selected = session_id
    if selected is not None:
        # Explicit missing/unsafe evidence is a hard error, not an empty PASS.
        _read_session(settings.qualification.evidence_dir, selected)
        source = "explicit"
    else:
        selected = latest_saved_session_id(settings.qualification.evidence_dir)
        if selected is not None:
            source = "latest_saved"
    # Readiness is obtained exactly once; callers may inject a tested snapshot
    # but CLI always performs the actual live authenticated PC-B probes.
    report = readiness if readiness is not None else diagnose_qualification_readiness(
        settings, session_id=selected, now=current,
    )
    if readiness is not None and selected is not None and not report.recorded_stages:
        raise ValueError("Injected readiness does not include selected session stages")
    correlation = (
        correlate_pc_b_owner_report(settings, pc_b_owner_report, now=current)
        if pc_b_owner_report is not None else None
    )
    plan = compile_qualification_action_plan(
        report, controller_config=controller_config,
        renderer_config=renderer_config, source="live", now=current,
    )
    stages: list[StageOverview] = []
    passed = skipped = unresolved = 0
    for stage in REQUIRED_STAGES:
        stored = report.recorded_stages.get(stage.value, "not_inspected")
        if stored not in {"pending", "pass", "fail", "skipped", "not_inspected"}:
            raise ValueError("Unexpected qualification stage status")
        if stored == QualificationStatus.PASS.value:
            passed += 1
        elif stored == QualificationStatus.SKIPPED.value:
            skipped += 1
        else:
            unresolved += 1
        kind = _STAGE_KIND[stage]
        stages.append(StageOverview(
            stage=stage.value, recorded_status=cast(StageRecordStatus, stored),
            evidence_kind=kind,
            next_action=None if stored in ("pass", "skipped") else _GUIDANCE[kind],
        ))
    triage = tuple(
        _triage_step(step, report, fresh=plan.snapshot_fresh)
        for step in plan.steps
    )
    already_sampled = tuple(
        item.step_id for item in triage if item.state == "observed_live"
    )
    unavailable = tuple(
        item.step_id for item in triage
        if item.state in ("unavailable", "not_executed")
    )
    pc_b_local = tuple(
        item.step_id for item in triage if item.state == "requires_local_pc_b"
    )
    review = tuple(
        item.step_id for item in triage if item.state == "operator_review"
    )
    if not plan.environment_ready_observed:
        # PC-A already ran the fixed read-only probes exactly once. Repeating
        # the same command immediately would not repair failed checks.
        # Prefer local PC-A human/config review over requesting PC-B action.
        manual_pc_a = next(
            (item for item in triage
             if item.role == "pc_a" and item.state == "operator_review"), None,
        )
        manual_pc_b = next(
            (item for item in triage
             if item.state == "requires_local_pc_b"), None,
        )
        manual_other = next(
            (item for item in triage if item.state == "operator_review"), None,
        )
        blocker = manual_pc_a or manual_pc_b or manual_other
        priority = (
            blocker.next_action if blocker is not None else
            "PC-A has attempted the available read-only checks. Inspect the "
            "reported FAIL/UNKNOWN results and restore access or prerequisites "
            "before rechecking; never force a ComfyUI restart."
        )
        command = None
    elif selected is None:
        priority = (
            "Start a real PC-A qualification session explicitly after confirming the "
            "native environment; starting a session does not start GPU work."
        )
        command = None
    elif unresolved:
        outstanding = next(item for item in stages if item.next_action is not None)
        priority = outstanding.next_action or "Inspect recorded stages"
        command = (
            "uv", "run", "artifex", "qualify", "collect", selected,
            "--config", str(controller_config), "--json",
        ) if outstanding.evidence_kind == "existing_pack" else None
    else:
        priority = (
            "All 14 stages have saved terminal statuses, but each stage and "
            "real target-machine assets still require authoritative qualify verify."
        )
        command = None
    if correlation is not None and correlation.status != "correlated_read_only":
        priority = (
            "Transferred PC-B ownership evidence has not been correlated with "
            "fresh authenticated observations. Inspect pc_b_owner_evidence "
            "and gather a new local PC-B report without restarting ComfyUI."
        )
        command = None
    return QualificationOverview(
        captured_utc=current, session_id=selected, session_selection=source,
        environment_ready=(
            plan.environment_ready_observed
            and (correlation is None or correlation.status == "correlated_read_only")
        ),
        recorded_pass_count=passed, recorded_skipped_count=skipped,
        unresolved_stage_count=unresolved, stages=tuple(stages),
        next_priority=priority, next_safe_command=command,
        remediation_triage=triage,
        already_sampled_read_only=already_sampled,
        unavailable_read_only=unavailable,
        pc_b_local_checks_required=pc_b_local,
        operator_review_required=review,
        readiness=report, action_plan=plan, pc_b_owner_evidence=correlation,
    )
