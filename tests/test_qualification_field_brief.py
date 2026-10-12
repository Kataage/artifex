"""Native-PC field brief: one safe command, no GPU, no false qualification."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from artifex import cli
from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification import field_brief as brief_module
from artifex.qualification.evidence_handoff import (
    EvidenceHandoffItem,
    QualificationEvidenceHandoff,
)
from artifex.qualification.field_brief import (
    QualificationFieldBrief,
    collect_field_brief,
    compile_field_brief,
)
from artifex.qualification.issue93_checklist import (
    Issue93Check,
    Issue93EvidenceChecklist,
)
from artifex.qualification.overview import QualificationOverview
from artifex.qualification.pc_b_monitor import PCBEvidenceMonitorView

NOW = datetime(2026, 10, 12, 1, 0, tzinfo=UTC)
PC_B_CONFIG = Path("E:/PC B/Artifex/custom renderer.yaml")


def _issue93(
    status: str = "needs_evidence",
    *, all_observed: bool = False,
) -> Issue93EvidenceChecklist:
    names = (
        "isolated_launcher_and_live_owner",
        "two_time_separated_owner_samples",
        "natural_exit_survival_trace",
        "cross_evidence_listener_identity",
    )
    checks = tuple(
        Issue93Check(
            name=name,  # type: ignore[arg-type]
            state="observed" if all_observed else "missing",
            reason="fixture",
        )
        for name in names
    )
    return Issue93EvidenceChecklist.model_construct(
        checked_utc=NOW, node_id="gpu-b", status=status,
        checks=checks, observed_checks=4 if all_observed else 0,
        missing_checks=0 if all_observed else 4,
        conflicting_checks=0, owner_readiness_status="not_observed",
        saved_owner_pair_status="not_observed",
        natural_exit_trace_status="not_observed",
        next_safe_actions=("No physical Windows evidence yet",),
        current_comfyui_listener_pid=101 if all_observed else None,
        current_comfyui_started_utc=(
            "2026-10-10T01:00:00+00:00" if all_observed else None
        ),
        current_pc_b_hostname="GPU-B" if all_observed else None,
    )


def _overview(
    *, ready: bool = False,
    owner: str = "blocked",
    survival: str = "blocked",
    session: str | None = None,
) -> QualificationOverview:
    tasks = (
        EvidenceHandoffItem(
            id="environment-prerequisites", stage=None, recorded_status=None,
            responsible_role="pc_a", safety="read_only_review",
            next_action="Review PC-A configuration",
        ),
        EvidenceHandoffItem(
            id="pc-b-owner-check", stage=None, recorded_status=None,
            responsible_role="pc_b", safety="read_only_review",
            next_action="Inspect PC-B owner",
        ),
        EvidenceHandoffItem(
            id="stage-discord_controls", stage="discord_controls",
            recorded_status="pending", responsible_role="discord",
            safety="authorized_interaction", next_action="Review Discord controls",
        ),
    )
    return QualificationOverview.model_construct(
        captured_utc=NOW, session_id=session,
        session_selection="latest_saved" if session else "none",
        pc_b_owner_evidence=SimpleNamespace(
            status=owner, remote_bearer_checked=True,
            live_listener_pid=101,
            live_listener_started_utc="2026-10-10T10:00:00+09:00",
            live_host="gpu-b",
        ),
        pc_b_survival_review=SimpleNamespace(
            status=survival, remote_bearer_checked=True,
            current_listener_pid=101,
            current_listener_started_utc="2026-10-10T01:00:00+00:00",
            live_host="GPU-B",
        ),
        pc_b_saved_monitor=PCBEvidenceMonitorView(
            status="missing", next_action="No passive sample",
        ),
        environment_ready=ready,
        recorded_pass_count=2,
        recorded_skipped_count=1,
        unresolved_stage_count=11,
        next_priority="Review actual production Packs",
        evidence_handoff=QualificationEvidenceHandoff.model_construct(
            items=tasks,
        ),
    )


def _compiled(*, controller: Path = Path("C:/PC A/my config.yaml")) -> QualificationFieldBrief:
    return compile_field_brief(
        _issue93(), _overview(), controller_config=controller,
        renderer_config=PC_B_CONFIG, checked_utc=NOW,
    )


def test_missing_proof_stays_unqualified_and_next_steps_are_role_scoped() -> None:
    result = _compiled()
    assert result.issue_93_status == "needs_evidence"
    assert result.cross_observation_identity == "not_comparable"
    assert result.missing_issue_93_checks == 4
    assert result.owner_live_status == "blocked"
    assert result.survival_live_status == "blocked"
    assert result.saved_monitor_status == "missing"
    assert result.recorded_pass_stages == 2
    assert result.recorded_skipped_stages == 1
    assert result.unresolved_stages == 11
    assert [task.role for task in result.next_role_tasks] == [
        "pc_a", "pc_b", "discord",
    ]
    assert all(task.executed is False for task in result.next_role_tasks)
    assert result.issue_93_field_plan.commands_executed is False
    assert result.commands_executed is False
    assert result.service_or_task_mutation is False
    assert result.gpu_jobs_submitted is False
    assert result.discord_messages_sent is False
    assert result.production_qualified is False
    assert result.authenticated_historical_supervisor_loss_proven is False
    assert result.eight_hour_soak_qualified is False
    assert any(
        str(PC_B_CONFIG) in step.argv
        for step in result.issue_93_field_plan.steps if step.role == "pc_b"
    )
    assert all(
        "--apply" not in step.argv and "--confirm" not in step.argv
        for step in result.issue_93_field_plan.steps
    )


def _remote_gates(
    *, status: str = "observed_prerequisites",
    agreement: str = "match",
):
    from artifex.qualification.pc_b_remote_field_gates import (
        compile_remote_pc_b_passive_gates,
    )

    base = compile_remote_pc_b_passive_gates(
        node_id="gpu-b", configured=False,
        installation=None, cached_owner=None,
        attempted=0, checked_utc=NOW,
    )
    return base.model_copy(update={
        "status": status,
        "cached_vs_live_owner_identity": agreement,
    })


def test_verified_cache_and_current_owner_defer_only_redundant_pc_b_rechecks() -> None:
    report = compile_field_brief(
        _issue93(),
        _overview(ready=True),
        controller_config=Path("C:/controller.yaml"),
        renderer_config=PC_B_CONFIG,
        pc_b_remote_gates=_remote_gates(),
    )
    assert report.pc_b_visit_triage == "pc_a_recheck_first"
    assert set(report.pc_b_deferred_repeat_checks) == {
        "pc-b-field-preflight",
    }
    # Full conservative plan remains visible; the action isn't completed.
    assert {
        step.id for step in report.issue_93_field_plan.steps
        if step.role == "pc_b"
    } == set(report.pc_b_deferred_repeat_checks)
    assert report.issue_93_field_plan.commands_executed is False
    assert report.production_qualified is False
    assert report.issue_93_closed is False
    assert report.issue_40_closed is False


@pytest.mark.parametrize(
    ("status", "agreement", "expected"),
    [
        ("needs_evidence", "conflict", "pc_b_read_only_review"),
        ("unavailable", "not_comparable", "pc_a_access_review_first"),
        ("unconfigured", "not_comparable", "pc_a_access_review_first"),
        ("observed_prerequisites", "not_comparable", "pc_b_read_only_review"),
    ],
)
def test_pc_b_visit_triage_never_defers_when_proof_is_missing_or_conflicts(
    status: str, agreement: str, expected: str,
) -> None:
    report = compile_field_brief(
        _issue93(),
        _overview(ready=True),
        controller_config=Path("C:/controller.yaml"),
        renderer_config=PC_B_CONFIG,
        pc_b_remote_gates=_remote_gates(status=status, agreement=agreement),
    )
    assert report.pc_b_visit_triage == expected
    assert report.pc_b_deferred_repeat_checks == ()
    if agreement == "conflict":
        assert "never restart" not in report.next_priority or (
            "do not" in report.next_priority
        )
        assert "do not" in report.next_priority
    assert not report.production_qualified
    assert all(not step.executed for step in report.issue_93_field_plan.steps)


def test_positive_correlated_sources_still_do_not_qualify_real_machine() -> None:
    result = compile_field_brief(
        _issue93("observations_correlated", all_observed=True),
        _overview(
            ready=True, owner="correlated_read_only",
            survival="replayed_and_live_identity_matched",
            session="20261012T010000Z-aabbccdd",
        ),
        controller_config=Path("C:/controller.yaml"),
        renderer_config=PC_B_CONFIG,
    )
    assert result.issue_93_status == "observations_correlated"
    assert result.cross_observation_identity == "match"
    assert result.next_priority == "Review actual production Packs"
    assert result.session_id == "20261012T010000Z-aabbccdd"
    assert result.production_qualified is False
    assert result.issue_93_closed is False
    assert result.issue_40_closed is False
    assert result.live_comfyui_survival_qualified is False
    assert result.fourteen_stages_independently_verified is False
    assert result.issue_93_field_plan.steps[0].id == "pc-a-production-overview"


def test_separate_authenticated_observations_detect_pid_reuse_and_host_drift() -> None:
    base = _overview(
        ready=True, owner="correlated_read_only",
        survival="replayed_and_live_identity_matched",
    )
    assert base.pc_b_survival_review is not None
    base.pc_b_survival_review.current_listener_started_utc = (
        "2026-10-10T01:00:01+00:00"
    )
    result = compile_field_brief(
        _issue93("observations_correlated", all_observed=True), base,
        controller_config=Path("C:/controller.yaml"),
        renderer_config=PC_B_CONFIG,
    )
    assert result.cross_observation_identity == "conflict"
    assert "never restart ComfyUI" in result.next_priority
    assert result.production_qualified is False

    # Missing creation times are never proof of agreement.
    base.pc_b_survival_review.current_listener_started_utc = None
    second = compile_field_brief(
        _issue93("observations_correlated", all_observed=True), base,
        controller_config=Path("C:/controller.yaml"),
        renderer_config=PC_B_CONFIG,
    )
    assert second.cross_observation_identity == "not_comparable"



@pytest.mark.parametrize(
    ("owner_status", "survival_status", "missing"),
    [
        ("unavailable", "replayed_and_live_identity_matched", ("owner",)),
        ("correlated_read_only", "remote_unavailable", ("survival",)),
        ("unavailable", "remote_unavailable", ("owner", "survival")),
    ],
)
def test_partial_field_brief_retains_safe_status_without_fabricating_proof(
    owner_status: str, survival_status: str, missing: tuple[str, ...],
) -> None:
    report = compile_field_brief(
        _issue93("needs_evidence"),
        _overview(
            ready=True, owner=owner_status,
            survival=survival_status,
        ),
        controller_config=Path("C:/controller.yaml"),
        renderer_config=PC_B_CONFIG,
        checked_utc=NOW,
    )
    assert report.failed_live_probes == missing
    assert report.partial_live_diagnostics is True
    assert report.owner_live_status == owner_status
    assert report.survival_live_status == survival_status
    assert report.cross_observation_identity == "not_comparable"
    assert report.recorded_pass_stages == 2
    assert report.recorded_skipped_stages == 1
    assert report.unresolved_stages == 11
    assert "never restart" in report.next_priority
    assert report.commands_executed is False
    assert report.production_qualified is False
    assert report.issue_93_closed is False
    assert report.issue_40_closed is False


def test_field_brief_nonmissing_but_blocked_is_not_a_transport_exception() -> None:
    report = _compiled()
    assert report.failed_live_probes == ()
    assert report.partial_live_diagnostics is False
    assert report.owner_live_status == "blocked"
    assert report.survival_live_status == "blocked"
    assert report.production_qualified is False

def test_conflicting_owner_uses_explicit_safe_investigation() -> None:
    report = _issue93("conflict").model_copy(update={"conflicting_checks": 1})
    result = compile_field_brief(
        report, _overview(ready=True),
        controller_config=Path("C:/controller.yaml"),
        renderer_config=PC_B_CONFIG,
    )
    assert "never restart" in result.next_priority
    assert result.production_qualified is False


@pytest.mark.parametrize("remote", [False, True])
def test_collect_reuses_both_existing_read_only_compilers_without_mutation(
    monkeypatch: pytest.MonkeyPatch, remote: bool,
) -> None:
    settings = ArtifexSettings()
    if remote:
        settings.render_nodes.primary = "gpu-b"
        settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
            base_url="http://192.0.2.10:8188",
            attestation_url="http://192.0.2.10:8190",
        )
    calls: list[tuple[str, object]] = []

    def issue(settings_arg: ArtifexSettings) -> Issue93EvidenceChecklist:
        assert settings_arg is settings
        calls.append(("issue93", None))
        return _issue93()

    def overview(settings_arg: ArtifexSettings, **kwargs: object) -> QualificationOverview:
        assert settings_arg is settings
        calls.append(("overview", kwargs))
        return _overview()

    monkeypatch.setattr(brief_module, "compile_issue93_checklist", issue)
    monkeypatch.setattr(brief_module, "compile_qualification_overview", overview)

    def passive(
        settings_arg: ArtifexSettings, *, current_owner: object | None = None,
    ) -> object:
        assert settings_arg is settings
        calls.append(("passive", None))
        from artifex.qualification.pc_b_remote_field_gates import (
            inspect_remote_pc_b_passive_gates,
        )
        return inspect_remote_pc_b_passive_gates(ArtifexSettings(), now=NOW)

    monkeypatch.setattr(brief_module, "inspect_remote_pc_b_passive_gates", passive)
    result = collect_field_brief(
        settings,
        controller_config=Path("C:/controller.yaml"),
        renderer_config=PC_B_CONFIG,
    )
    assert result.production_qualified is False
    assert len(calls) == 3
    assert calls[0][0] == "issue93"
    assert calls[1][0] == "overview"
    assert calls[2][0] == "passive"
    assert result.pc_b_remote_passive_gates is not None
    assert result.pc_b_remote_passive_gates.status == "unconfigured"
    assert result.pc_b_remote_passive_gates.production_qualified is False
    options = calls[1][1]
    assert isinstance(options, dict)
    assert options["pc_b_owner_live"] is remote
    assert options["pc_b_survival_live"] is remote


def test_cli_is_one_command_and_saves_only_opt_in_local_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "controller.yaml"
    config.write_text("{}\n", encoding="utf-8")
    settings = ArtifexSettings()
    settings.qualification.evidence_dir = tmp_path / "evidence"
    monkeypatch.setattr(cli, "_settings", lambda _: settings)
    calls: list[object] = []

    def collect(*args: object, **kwargs: object) -> QualificationFieldBrief:
        calls.append(kwargs)
        return _compiled(controller=config)

    monkeypatch.setattr(brief_module, "collect_field_brief", collect)
    runner = CliRunner()
    args = [
        "qualify", "field-brief", "--config", str(config),
        "--renderer-config", str(PC_B_CONFIG), "--json",
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    one = json.loads(first.stdout)
    assert one["issue_93_status"] == "needs_evidence"
    assert one["production_qualified"] is False
    assert one["session_id"] is None
    assert not settings.qualification.evidence_dir.exists()

    saved = runner.invoke(app, [*args, "--save"])
    assert saved.exit_code == 0, saved.output
    two = json.loads(saved.stdout)
    target = Path(two["saved_file"])
    assert target.parent == settings.qualification.evidence_dir / "field-brief"
    assert target.is_file()
    disk = json.loads(target.read_text(encoding="utf-8"))
    assert "saved_file" not in disk
    assert disk["production_qualified"] is False
    assert disk["commands_executed"] is False
    assert any(
        str(PC_B_CONFIG) in step["argv"]
        for step in disk["issue_93_field_plan"]["steps"]
        if step["role"] == "pc_b"
    )
    assert len(calls) == 2


def test_cli_rejects_symlink_destination_and_conflicting_output_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "controller.yaml"
    config.write_text("{}\n", encoding="utf-8")
    settings = ArtifexSettings()
    settings.qualification.evidence_dir = tmp_path / "evidence"
    monkeypatch.setattr(cli, "_settings", lambda _: settings)
    monkeypatch.setattr(
        brief_module, "collect_field_brief",
        lambda *a, **k: _compiled(controller=config),
    )
    runner = CliRunner()
    base = ["qualify", "field-brief", "--config", str(config)]
    conflict = runner.invoke(app, [
        *base, "--save", "--output", str(tmp_path / "named.json"),
    ])
    assert conflict.exit_code == 1
    assert "ValueError" in conflict.output
    assert not (tmp_path / "named.json").exists()

    dest = tmp_path / "existing.json"
    dest.write_text("original", encoding="utf-8")
    existing = runner.invoke(app, [*base, "--output", str(dest)])
    assert existing.exit_code == 1
    assert dest.read_text(encoding="utf-8") == "original"

    linked = tmp_path / "linked.json"
    try:
        linked.symlink_to(dest)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this Windows host")
    reject = runner.invoke(app, [*base, "--output", str(linked)])
    assert reject.exit_code == 1
    assert dest.read_text(encoding="utf-8") == "original"
