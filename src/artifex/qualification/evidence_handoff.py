"""Advisory PC-A / PC-B / Discord handoff for real-machine qualification.

Only consumes existing qualification status and evidence diagnostics. The
handoff never runs commands, promotes stored PASS records, or authorizes
restarting an occupied renderer.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from artifex.qualification.discord_completion_evidence import DiscordCompletionEvidence
from artifex.qualification.discord_control_evidence import DiscordControlEvidence
from artifex.qualification.models import REQUIRED_STAGES, QualificationStage

HandoffRole = Literal["pc_a", "pc_b", "discord"]
HandoffSafety = Literal[
    "read_only_review",
    "explicit_session_start",
    "authoritative_verification",
    "normal_production",
    "controlled_gpu_window",
    "long_running_observation",
    "authorized_interaction",
    "archive_reproduction",
]
_StageInstructions = tuple[HandoffRole, tuple[HandoffRole, ...], HandoffSafety, str]

_STAGE_HANDOFF: dict[QualificationStage, _StageInstructions] = {
    QualificationStage.DOCTOR: (
        "pc_a", ("pc_b",), "read_only_review",
        ("Review native PC-A doctor baseline, configured models and authenticated "
        "PC-B readiness. A stored PASS is not a fresh observation."),
    ),
    QualificationStage.SINGLE_CHARACTER: (
        "pc_a", ("pc_b",), "normal_production",
        ("Use normal Artifex production for a real single-character Pack created "
        "after this session started. Preview its persisted archive evidence."),
    ),
    QualificationStage.LORA_REQUIRED: (
        "pc_a", ("pc_b",), "normal_production",
        ("Observe a real post-session Pack whose render actually used a required "
        "LoRA. Do not simulate a LoRA load or manufacture a Pack."),
    ),
    QualificationStage.DUO: (
        "pc_a", ("pc_b",), "normal_production",
        ("Observe an actual two-character Pack from normal production with "
        "reproducible post-session archive evidence."),
    ),
    QualificationStage.GROUP: (
        "pc_a", ("pc_b",), "normal_production",
        ("Observe an actual three-or-more-character Pack from normal production "
        "with independently inspectable provenance."),
    ),
    QualificationStage.PUBLIC_MEMBER: (
        "pc_a", ("pc_b",), "normal_production",
        ("Use existing publication-safe production to verify public/member "
        "variants from actual post-session finalized Pack records."),
    ),
    QualificationStage.SERIES_CONTINUATION: (
        "pc_a", ("pc_b",), "normal_production",
        ("Observe genuine series continuation in post-session Pack records. "
        "Do not insert artificial continuity markers."),
    ),
    QualificationStage.FORCED_RETRY: (
        "pc_a", ("pc_b",), "normal_production",
        ("Require an actual recorded production retry; do not inject a failure "
        "into an occupied GPU job merely to satisfy this stage."),
    ),
    QualificationStage.RESTART_GENERATION: (
        "pc_b", ("pc_a",), "controlled_gpu_window",
        ("Schedule a controlled real-machine restart/recovery observation only "
        "after confirming renderer ownership and no conflicting active GPU work."),
    ),
    QualificationStage.BACKEND_RECOVERY: (
        "pc_b", ("pc_a",), "controlled_gpu_window",
        ("Prove real ComfyUI/backend recovery in a separately approved safe "
        "maintenance window. Never terminate an unknown running process."),
    ),
    QualificationStage.UNATTENDED_MULTI_PACK: (
        "pc_a", ("pc_b",), "long_running_observation",
        ("Observe genuine autonomous post-session multi-Pack production with "
        "continuous daemon and renderer provenance, without fabricating runs."),
    ),
    QualificationStage.OVERNIGHT_SOAK: (
        "pc_a", ("pc_b",), "long_running_observation",
        ("Run and independently review a real minimum eight-hour target-machine "
        "trace covering PC-A, PC-B GPU/resources, ownership and failures."),
    ),
    QualificationStage.DISCORD_CONTROLS: (
        "discord", ("pc_a",), "authorized_interaction",
        ("Use real authorized Discord controls after session start; complete "
        "each missing successful response and preserve the exact event IDs."),
    ),
    QualificationStage.ARCHIVE_REPRODUCTION: (
        "pc_a", ("pc_b",), "archive_reproduction",
        ("Replay a genuine archived Pack with verified original provenance and "
        "matching output; an existing image alone cannot satisfy this stage."),
    ),
}


class EvidenceHandoffItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    stage: str | None
    recorded_status: Literal["pending", "fail", "not_inspected"] | None
    responsible_role: HandoffRole
    supporting_roles: tuple[HandoffRole, ...] = ()
    safety: HandoffSafety
    next_action: str
    blocked_by_environment: bool = False
    automated_execution: Literal[False] = False
    stage_independently_revalidated: Literal[False] = False


class QualificationEvidenceHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str | None
    environment_blocked: bool
    items: tuple[EvidenceHandoffItem, ...]
    outstanding_stage_items: int
    pc_a_items: int
    pc_b_items: int
    discord_items: int
    commands_executed: Literal[False] = False
    qualifies_production: Literal[False] = False


def compile_evidence_handoff(
    *,
    session_id: str | None,
    recorded_stages: dict[str, str],
    environment_ready: bool,
    discord_enabled: bool,
    control_evidence: DiscordControlEvidence,
    completion_evidence: DiscordCompletionEvidence,
) -> QualificationEvidenceHandoff:
    """Build only advisory, role-scoped next work from already observed facts."""
    items: list[EvidenceHandoffItem] = []
    if not environment_ready:
        items.append(EvidenceHandoffItem(
            id="environment-prerequisites", stage=None, recorded_status=None,
            responsible_role="pc_a", supporting_roles=("pc_b",),
            safety="read_only_review",
            next_action=(
                "Inspect the existing PC-A/PC-B readiness action_plan first. "
                "Restore configuration, ownership and connectivity safely; "
                "do not force a ComfyUI restart or attempt GPU qualification."
            ),
            blocked_by_environment=True,
        ))

    if session_id is None:
        items.append(EvidenceHandoffItem(
            id="start-session", stage=None, recorded_status=None,
            responsible_role="pc_a", supporting_roles=("pc_b",),
            safety="explicit_session_start",
            next_action=(
                "After readiness is confirmed, explicitly run qualify start on "
                "PC-A to capture real baseline and session identity. Starting "
                "a session is an intentional write, not a read-only probe, "
                "and does not launch a GPU job."
            ),
            blocked_by_environment=not environment_ready,
        ))
    else:
        for stage in REQUIRED_STAGES:
            status = recorded_stages.get(stage.value, "not_inspected")
            if status in ("pass", "skipped"):
                continue
            if status not in ("pending", "fail", "not_inspected"):
                raise ValueError("Unexpected qualification stage status in handoff")
            role, helpers, safety, description = _STAGE_HANDOFF[stage]
            if stage is QualificationStage.DISCORD_CONTROLS:
                if not discord_enabled:
                    role, helpers, safety = "pc_a", (), "read_only_review"
                    description = (
                        "Discord is disabled: inspect this session's frozen "
                        "configuration. Only the authoritative qualification "
                        "service can accept an explicitly disabled SKIPPED stage."
                    )
                elif control_evidence.status == "incomplete":
                    description += (
                        " Missing candidate controls: "
                        + ", ".join(control_evidence.missing_commands)
                        + ". Candidates are not a verified stage PASS."
                    )
                elif control_evidence.status == "observed":
                    description += (
                        " All six have saved candidates; use authoritative "
                        "record/verify rather than repeating Discord actions."
                    )
                else:
                    description += (
                        " Control evidence status: " + control_evidence.status
                        + ". Do not interpret unknown evidence as zero."
                    )
            items.append(EvidenceHandoffItem(
                id="stage-" + stage.value,
                stage=stage.value,
                recorded_status=cast(
                    Literal["pending", "fail", "not_inspected"], status,
                ),
                responsible_role=role,
                supporting_roles=helpers,
                safety=safety,
                next_action=description,
                blocked_by_environment=not environment_ready,
            ))

        # Operational issues must not be disguised as a missing qualification
        # stage, nor must they automatically resend possibly delivered posts.
        if discord_enabled and completion_evidence.status == "inconsistent":
            items.append(EvidenceHandoffItem(
                id="discord-notice-inconsistency",
                stage=None, recorded_status=None,
                responsible_role="pc_a", supporting_roles=("discord",),
                safety="read_only_review",
                next_action=completion_evidence.next_action,
            ))
        elif discord_enabled and completion_evidence.status == "observed" and (
            (completion_evidence.uncertain or 0)
            or (completion_evidence.sending or 0)
        ):
            items.append(EvidenceHandoffItem(
                id="discord-notice-uncertain",
                stage=None, recorded_status=None,
                responsible_role="pc_a", supporting_roles=("discord",),
                safety="read_only_review",
                next_action=completion_evidence.next_action,
            ))

        if not any(item.stage is not None for item in items):
            items.append(EvidenceHandoffItem(
                id="authoritative-final-verification",
                stage=None, recorded_status=None,
                responsible_role="pc_a", supporting_roles=("pc_b", "discord"),
                safety="authoritative_verification",
                blocked_by_environment=not environment_ready,
                next_action=(
                    "Saved stage statuses have no outstanding entries, but none "
                    "has been independently revalidated by this overview. "
                    "Run authoritative qualify verify with a live PC-B owner "
                    "observation; never infer production qualification."
                ),
            ))

    result = tuple(items)
    return QualificationEvidenceHandoff(
        session_id=session_id, environment_blocked=not environment_ready,
        items=result,
        outstanding_stage_items=sum(item.stage is not None for item in result),
        pc_a_items=sum(item.responsible_role == "pc_a" for item in result),
        pc_b_items=sum(item.responsible_role == "pc_b" for item in result),
        discord_items=sum(item.responsible_role == "discord" for item in result),
    )
