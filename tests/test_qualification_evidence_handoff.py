from __future__ import annotations

import pytest

from artifex.qualification.discord_completion_evidence import DiscordCompletionEvidence
from artifex.qualification.discord_control_evidence import DiscordControlEvidence
from artifex.qualification.evidence_handoff import compile_evidence_handoff
from artifex.qualification.models import REQUIRED_STAGES


def _control(
    status: str = "incomplete",
    missing: tuple[str, ...] = ("approve", "pause", "reject"),
) -> DiscordControlEvidence:
    return DiscordControlEvidence(
        status=status,
        session_id="session-1",
        missing_commands=missing,
        next_action="Observe real interactions only.",
    )


def _completion(
    status: str = "observed", uncertain: int | None = 0,
) -> DiscordCompletionEvidence:
    return DiscordCompletionEvidence(
        status=status, uncertain=uncertain,
        next_action="Inspect uncertain notices without resending.",
    )


def test_without_session_only_suggests_safe_environment_and_start() -> None:
    result = compile_evidence_handoff(
        session_id=None, recorded_stages={},
        environment_ready=False, discord_enabled=True,
        control_evidence=_control(), completion_evidence=_completion(),
    )
    assert [task.id for task in result.items] == [
        "environment-prerequisites", "start-session",
    ]
    assert result.outstanding_stage_items == 0
    assert result.pc_a_items == 2
    assert result.pc_b_items == result.discord_items == 0
    assert all(task.automated_execution is False for task in result.items)
    assert all(task.blocked_by_environment for task in result.items)
    assert result.commands_executed is False
    assert result.qualifies_production is False


def test_role_specific_handoff_covers_all_fourteen_unresolved_stages() -> None:
    result = compile_evidence_handoff(
        session_id="session-1",
        recorded_stages={stage.value: "pending" for stage in REQUIRED_STAGES},
        environment_ready=True, discord_enabled=True,
        control_evidence=_control(), completion_evidence=_completion(),
    )
    assert result.outstanding_stage_items == 14
    assert len(result.items) == 14
    assert len({task.stage for task in result.items}) == 14
    assert [task.stage for task in result.items] == [
        stage.value for stage in REQUIRED_STAGES
    ]
    by_stage = {task.stage: task for task in result.items}
    assert by_stage["restart_generation"].responsible_role == "pc_b"
    assert by_stage["backend_recovery"].safety == "controlled_gpu_window"
    assert "conflicting active GPU work" in by_stage["restart_generation"].next_action
    assert by_stage["overnight_soak"].responsible_role == "pc_a"
    assert by_stage["overnight_soak"].safety == "long_running_observation"
    assert "eight-hour" in by_stage["overnight_soak"].next_action
    assert by_stage["discord_controls"].responsible_role == "discord"
    assert "approve, pause, reject" in by_stage["discord_controls"].next_action
    assert by_stage["discord_controls"].stage_independently_revalidated is False
    assert result.pc_a_items == 11
    assert result.pc_b_items == 2
    assert result.discord_items == 1
    assert all(task.automated_execution is False for task in result.items)


def test_completed_stages_not_relisted_and_final_verification_never_promoted() -> None:
    result = compile_evidence_handoff(
        session_id="session-1",
        recorded_stages={stage.value: "pass" for stage in REQUIRED_STAGES},
        environment_ready=True, discord_enabled=True,
        control_evidence=_control(status="observed", missing=()),
        completion_evidence=_completion(),
    )
    assert [task.id for task in result.items] == [
        "authoritative-final-verification",
    ]
    assert result.outstanding_stage_items == 0
    assert result.items[0].stage is None
    assert result.items[0].stage_independently_revalidated is False
    assert result.qualifies_production is False


def test_disabled_discord_never_requests_real_discord_actions() -> None:
    statuses = {stage.value: "pass" for stage in REQUIRED_STAGES}
    statuses["discord_controls"] = "pending"
    result = compile_evidence_handoff(
        session_id="session-1", recorded_stages=statuses,
        environment_ready=True, discord_enabled=False,
        control_evidence=_control(status="disabled"),
        completion_evidence=_completion(status="disabled", uncertain=None),
    )
    assert len(result.items) == 1
    assert "Discord is disabled" in result.items[0].next_action
    assert "SKIPPED" in result.items[0].next_action


def test_completion_uncertainty_is_distinct_from_six_control_stage() -> None:
    statuses = {stage.value: "pass" for stage in REQUIRED_STAGES}
    statuses["discord_controls"] = "fail"
    result = compile_evidence_handoff(
        session_id="session-1", recorded_stages=statuses,
        environment_ready=True, discord_enabled=True,
        control_evidence=_control(status="observed", missing=()),
        completion_evidence=_completion(uncertain=3),
    )
    assert [task.id for task in result.items] == [
        "stage-discord_controls", "discord-notice-uncertain",
    ]
    assert result.items[0].recorded_status == "fail"
    assert "authoritative" in result.items[0].next_action
    assert result.items[1].stage is None
    assert result.items[1].safety == "read_only_review"
    assert "without resending" in result.items[1].next_action


def test_inconsistent_notice_rows_request_separate_manual_read_only_review() -> None:
    statuses = {stage.value: "skipped" for stage in REQUIRED_STAGES}
    result = compile_evidence_handoff(
        session_id="session-1", recorded_stages=statuses,
        environment_ready=True, discord_enabled=True,
        control_evidence=_control(),
        completion_evidence=_completion(status="inconsistent"),
    )
    assert result.outstanding_stage_items == 0
    assert [task.id for task in result.items] == [
        "discord-notice-inconsistency", "authoritative-final-verification",
    ]
    assert all(task.automated_execution is False for task in result.items)


def test_unknown_stage_status_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="Unexpected qualification"):
        compile_evidence_handoff(
            session_id="session-1",
            recorded_stages={"doctor": "impossible"},
            environment_ready=True, discord_enabled=True,
            control_evidence=_control(), completion_evidence=_completion(),
        )
