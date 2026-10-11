from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from artifex.qualification.discord_completion_evidence import DiscordCompletionEvidence
from artifex.qualification.discord_control_evidence import DiscordControlEvidence
from artifex.qualification.evidence_handoff import compile_evidence_handoff
from artifex.qualification.owner_handoff import PCBOwnerCorrelation
from artifex.qualification.pc_b_evidence_gaps import (
    PCBEvidenceCheck,
    compile_pc_b_evidence_checks,
)
from artifex.qualification.survival_review import PCBSurvivalEvidenceReview

NOW = datetime(2026, 10, 11, 12, tzinfo=UTC)
CONFIG = Path("D:/Artifex User/config/local.yaml")


def _owner(status: str, *, source_mode: str = "transferred_file") -> PCBOwnerCorrelation:
    return PCBOwnerCorrelation(
        checked_utc=NOW, status=status, reason="real owner correlation result",
        remaining_real_machine_evidence=("eight_hour_unattended_gpu_soak",),
        source_mode=source_mode,
    )


def _survival(
    status: str, *, replayed: bool = True,
    source_mode: str = "transferred_file",
) -> PCBSurvivalEvidenceReview:
    return PCBSurvivalEvidenceReview(
        reviewed_utc=NOW, status=status, reason="historical trace assessment",
        trace_replayed=replayed, source_mode=source_mode,
    )


def test_without_configured_pc_b_never_invents_hardware_or_commands() -> None:
    checks = compile_pc_b_evidence_checks(
        pc_b_configured=False,
        owner=None, survival=None, controller_config=CONFIG,
    )
    assert checks == ()


def test_missing_two_pc_reports_produce_two_safe_pc_a_requests() -> None:
    checks = compile_pc_b_evidence_checks(
        pc_b_configured=True,
        owner=None, survival=None, controller_config=CONFIG,
    )
    assert len(checks) == 2
    assert [c.assessment for c in checks] == ["not_requested", "not_requested"]
    assert [c.proof for c in checks] == [
        "renderer_owner_identity", "supervisor_survival_trace",
    ]
    for c in checks:
        assert c.responsible_role == "pc_a"
        assert c.recorded_status is None and c.source_mode is None
        assert c.identity_correlated is False
        assert c.read_only_argv is not None
        assert c.read_only_argv[:5] == (
            "uv", "run", "artifex", "qualify", "overview",
        )
        assert "--config" in c.read_only_argv
        assert str(CONFIG) in c.read_only_argv
        assert "--apply" not in c.read_only_argv
        assert c.commands_executed is False
        assert c.real_gpu_qualified is False
        assert c.supervisor_loss_qualified is False
        assert c.issue_93_closed is False
    assert "--pc-b-owner-live" in checks[0].read_only_argv
    assert "--pc-b-survival-live" in checks[1].read_only_argv


def test_correlated_owner_and_survival_still_cannot_qualify_physical_loss() -> None:
    checks = compile_pc_b_evidence_checks(
        pc_b_configured=True,
        owner=_owner("correlated_read_only", source_mode="authenticated_remote"),
        survival=_survival(
            "replayed_and_live_identity_matched", source_mode="bearer_remote",
        ),
        controller_config=CONFIG,
    )
    assert len(checks) == 2
    assert all(c.assessment == "identity_correlated" for c in checks)
    assert all(c.identity_correlated for c in checks)
    assert checks[0].trace_replayed is False
    assert checks[1].trace_replayed is True
    assert checks[0].source_mode == "authenticated_remote"
    assert checks[1].source_mode == "bearer_remote"
    assert all(c.read_only_argv is None for c in checks)
    assert all(c.supervisor_loss_qualified is False for c in checks)
    assert all(c.issue_93_closed is False for c in checks)
    assert "not authenticated" in checks[1].next_action

    handoff = compile_evidence_handoff(
        session_id="session-1",
        recorded_stages={"doctor": "pass"},
        environment_ready=True,
        discord_enabled=False,
        control_evidence=DiscordControlEvidence(
            status="disabled", next_action="Discord disabled",
        ),
        completion_evidence=DiscordCompletionEvidence(
            status="disabled", next_action="Discord disabled",
        ),
        pc_b_evidence_checks=checks,
    )
    assert handoff.pc_b_evidence_checks == checks
    assert handoff.unresolved_pc_b_evidence_checks == 0
    assert not any(item.id.startswith("pc-b-") for item in handoff.items)
    assert handoff.qualifies_production is False
    assert handoff.commands_executed is False


def test_historical_survival_mismatch_is_pc_b_review_not_success() -> None:
    checks = compile_pc_b_evidence_checks(
        pc_b_configured=True,
        owner=_owner("stale"),
        survival=_survival("replayed_historical_identity"),
        controller_config=CONFIG,
    )
    assert checks[0].assessment == "needs_review"
    assert checks[0].responsible_role == "pc_b"
    assert checks[1].assessment == "historical_only"
    assert checks[1].trace_replayed is True
    assert checks[1].identity_correlated is False
    assert checks[1].responsible_role == "pc_b"
    assert "not live survivor qualification" in checks[1].next_action

    handoff = compile_evidence_handoff(
        session_id="session-1",
        recorded_stages={"doctor": "pass"},
        environment_ready=False,
        discord_enabled=False,
        control_evidence=DiscordControlEvidence(
            status="disabled", next_action="Discord disabled",
        ),
        completion_evidence=DiscordCompletionEvidence(
            status="disabled", next_action="Discord disabled",
        ),
        pc_b_evidence_checks=checks,
    )
    assert handoff.unresolved_pc_b_evidence_checks == 2
    assert [item.id for item in handoff.items[:3]] == [
        "environment-prerequisites",
        "pc-b-renderer_owner_identity",
        "pc-b-supervisor_survival_trace",
    ]
    assert all(item.stage is None for item in handoff.items[:3])
    assert all(item.blocked_by_environment for item in handoff.items[:3])
    assert handoff.pc_b_items >= 2
    assert handoff.items[1].read_only_argv is None
    assert handoff.items[1].automated_execution is False
    assert handoff.qualifies_production is False


def test_failed_remote_snapshot_does_not_fall_back_to_fake_empty_result() -> None:
    checks = compile_pc_b_evidence_checks(
        pc_b_configured=True,
        owner=_owner("unavailable"),
        survival=_survival("remote_unavailable", replayed=False),
        controller_config=CONFIG,
    )
    assert all(c.assessment == "needs_review" for c in checks)
    assert all(c.responsible_role == "pc_a" for c in checks)
    assert all(c.identity_correlated is False for c in checks)
    assert all(c.read_only_argv is None for c in checks)
    assert all(c.issue_93_closed is False for c in checks)
    assert all(c.real_gpu_qualified is False for c in checks)
    assert PCBEvidenceCheck.model_validate(
        checks[1].model_dump(mode="json")
    ) == checks[1]
