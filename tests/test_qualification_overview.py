from __future__ import annotations

import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.qualification.models import (
    REQUIRED_STAGES,
    QualificationSession,
    QualificationStage,
    QualificationStageEvidence,
    QualificationStatus,
)
from artifex.qualification.overview import (
    QualificationOverview,
    compile_qualification_overview,
    latest_saved_session_id,
)
from artifex.qualification.readiness_diagnostics import (
    QualificationReadiness,
    ReadinessCheck,
)

NOW = datetime(2026, 10, 9, 13, 0, tzinfo=UTC)


def _session(
    root: Path, ident: str, *,
    hostname: str | None = None,
    passed: tuple[QualificationStage, ...] = (),
    skipped: tuple[QualificationStage, ...] = (),
) -> None:
    folder = root / ident
    folder.mkdir(parents=True, exist_ok=True)
    stages = {
        stage.value: QualificationStageEvidence(
            stage=stage,
            status=(
                QualificationStatus.PASS if stage in passed
                else QualificationStatus.SKIPPED if stage in skipped
                else QualificationStatus.PENDING
            ),
        ) for stage in REQUIRED_STAGES
    }
    session = QualificationSession(
        session_id=ident, created_at=NOW - timedelta(minutes=30),
        updated_at=NOW, hostname=hostname or socket.gethostname(),
        environment={}, configuration={}, workflow={}, assets=(),
        loras=(), doctor_ready=True, doctor={}, stages=stages,
    )
    (folder / "qualification.json").write_text(
        session.model_dump_json(), encoding="utf-8",
    )


def _report(
    *,
    ready: bool,
    stages: dict[str, str] | None = None,
) -> QualificationReadiness:
    checks = (
        ReadinessCheck(
            target="pc_a",
            name="preflight:llm_lan" if not ready else "native_windows",
            status="unknown" if not ready else "pass",
            reason="read-only simulated observation",
            next_action="Recheck PC-A endpoint" if not ready else None,
        ),
    )
    recorded = stages or {}
    return QualificationReadiness(
        captured_utc=NOW,
        primary_node_id="gpu-b",
        environment_ready=ready,
        checks=checks,
        recorded_stages=recorded,
        pending_stages=tuple(
            item for item in recorded if recorded[item] != "pass"
        ),
    )


def _settings(root: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.qualification.evidence_dir = root
    return settings


def test_newest_local_session_is_selected_without_calling_production_verification(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    older = "20261008T110000Z-aaaaaaaa"
    newer = "20261009T120000Z-bbbbbbbb"
    _session(root, older, passed=(QualificationStage.DOCTOR,))
    _session(root, newer, passed=(
        QualificationStage.DOCTOR, QualificationStage.SINGLE_CHARACTER,
    ), skipped=(QualificationStage.DISCORD_CONTROLS,))
    assert latest_saved_session_id(root) == newer
    statuses = {stage.value: "pending" for stage in REQUIRED_STAGES}
    statuses["doctor"] = "pass"
    statuses["single_character"] = "pass"
    statuses["discord_controls"] = "skipped"
    report = compile_qualification_overview(
        _settings(root), readiness=_report(ready=True, stages=statuses),
        now=NOW,
        controller_config=Path("D:/AI/artifex/local.yaml"),
    )
    assert report.session_id == newer
    assert report.session_selection == "latest_saved"
    assert report.auto_collection_binding == "not_checked"
    assert report.recorded_pass_count == 2
    assert report.recorded_skipped_count == 1
    assert report.unresolved_stage_count == 11
    assert len(report.stages) == 14
    assert report.stages[1].recorded_status == "pass"
    assert report.stages[1].independently_revalidated is False
    assert report.stages[-1].evidence_kind == "archive_reproduction"
    assert report.next_safe_command is not None
    assert report.next_safe_command[:5] == (
        "uv", "run", "artifex", "qualify", "collect",
    )
    assert "--apply" not in report.next_safe_command
    assert report.commands_executed is False
    assert report.production_qualified is False
    assert report.actual_eight_hour_soak_verified is False
    assert report.action_plan.production_qualified is False


def test_unready_pc_a_is_prioritized_before_saved_stage_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    ident = "20261009T120000Z-bbbbbbbb"
    _session(root, ident)
    states = {stage.value: "pending" for stage in REQUIRED_STAGES}
    result = compile_qualification_overview(
        _settings(root), session_id=ident,
        readiness=_report(ready=False, stages=states),
        now=NOW, controller_config=Path("C:/my config/controller.yaml"),
    )
    assert result.session_selection == "explicit"
    assert result.environment_ready is False
    assert result.next_safe_command is None
    assert result.already_sampled_read_only == ("pc-a-preflight",)
    assert result.operator_review_required == ()
    assert "already" not in result.next_priority.lower()
    assert "read-only checks" in result.next_priority
    assert result.recorded_pass_count == 0
    assert result.production_qualified is False
    assert result.action_plan.steps[-1].safety == "real_machine_evidence"


def test_no_session_never_becomes_qualification_and_recommends_start(
    tmp_path: Path,
) -> None:
    report = compile_qualification_overview(
        _settings(tmp_path / "absent"),
        readiness=_report(ready=True), now=NOW,
    )
    assert report.session_id is None
    assert report.session_selection == "none"
    assert report.unresolved_stage_count == 14
    assert all(item.recorded_status == "not_inspected" for item in report.stages)
    assert "start" in report.next_priority
    assert report.next_safe_command is None
    assert report.production_qualified is False


def test_latest_foreign_or_corrupt_session_fails_closed_without_old_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    _session(root, "20261008T110000Z-aaaaaaaa")
    latest = "20261009T120000Z-bbbbbbbb"
    _session(root, latest, hostname="another-computer")
    with pytest.raises(ValueError, match="another host"):
        latest_saved_session_id(root)
    file = root / latest / "qualification.json"
    file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        latest_saved_session_id(root)


def test_latest_symlink_and_bounded_scanning_are_never_trusted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    target = root / "real-folder"
    target.mkdir()
    alias = root / "20261009T120000Z-bbbbbbbb"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation unavailable on this host")
    with pytest.raises(ValueError, match="symlink"):
        latest_saved_session_id(root)


def test_overview_cli_is_json_only_no_gpu_no_clobber(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.cli as cli_module
    import artifex.qualification.overview as overview_module

    config = tmp_path / "local.yaml"
    config.write_text("safe: true\n", encoding="utf-8")
    settings = _settings(tmp_path / "no-sessions")
    monkeypatch.setattr(cli_module, "_settings", lambda path: settings)
    monkeypatch.setattr(
        overview_module, "diagnose_qualification_readiness",
        lambda *args, **kwargs: _report(ready=True),
    )
    output = tmp_path / "overview.json"
    args = [
        "qualify", "overview", "--config", str(config),
        "--output", str(output), "--json",
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert QualificationOverview.model_validate(payload).production_qualified is False
    assert payload["session_selection"] == "none"
    assert payload["commands_executed"] is False
    assert payload["mutated_services"] is False
    assert payload["readiness"]["actual_machine_qualification_complete"] is False
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8")) == payload




def test_blocker_triage_avoids_false_remote_sampling_and_pc_b_command_execution(
    tmp_path: Path,
) -> None:
    # An unrelated PC-A preflight check must not count as having sampled the
    # PC-B LAN endpoint or protected renderer inspection.
    root = tmp_path / "evidence"
    ident = "20261009T120000Z-bbbbbbbb"
    _session(root, ident)
    statuses = {stage.value: "pending" for stage in REQUIRED_STAGES}
    report = QualificationReadiness(
        captured_utc=NOW, primary_node_id="gpu-b", environment_ready=False,
        recorded_stages=statuses,
        checks=(
            ReadinessCheck(
                target="pc_a", name="preflight:llm_lan", status="fail",
                reason="No local endpoint",
            ),
            ReadinessCheck(
                target="pc_b", name="owner:remote_probe", status="unknown",
                reason="Missing authenticated endpoint",
            ),
            ReadinessCheck(
                target="pc_b", name="preflight:render_inventory",
                status="fail", reason="Inventory not proven",
            ),
            ReadinessCheck(
                target="pc_b", name="safety:remote_probe",
                status="unknown", reason="Unreachable service",
            ),
        ),
    )
    observed = compile_qualification_overview(
        _settings(root), session_id=ident, readiness=report, now=NOW,
    )
    triage = {item.step_id: item for item in observed.remediation_triage}
    assert triage["pc-a-preflight"].state == "observed_live"
    assert triage["pc-b-connectivity"].state == "unavailable" or (
        "pc-b-connectivity" not in triage
    )
    assert triage["pc-b-owner"].state == "unavailable"
    assert triage["pc-a-remote-renderer-safety"].state == "unavailable"
    assert triage["pc-b-inventory"].state == "requires_local_pc_b"
    assert "pc-b-inventory" in observed.pc_b_local_checks_required
    assert "pc-b-owner" in observed.unavailable_read_only
    assert "pc-a-preflight" in observed.already_sampled_read_only
    assert all(item.commands_executed is False for item in observed.remediation_triage)
    assert observed.next_safe_command is None
    assert not observed.production_qualified


def test_triage_reports_manual_pc_a_repair_before_any_pc_b_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    report = QualificationReadiness(
        captured_utc=NOW, primary_node_id="gpu-b", environment_ready=False,
        checks=(
            ReadinessCheck(
                target="pc_a", name="remote_owner_auth", status="fail",
                reason="PC-A token missing",
            ),
            ReadinessCheck(
                target="pc_b", name="preflight:render_inventory",
                status="fail", reason="Not verified",
            ),
        ),
    )
    overview = compile_qualification_overview(
        _settings(root), readiness=report, now=NOW,
    )
    assert "pc-a-config" in overview.operator_review_required
    assert "pc-b-inventory" in overview.pc_b_local_checks_required
    assert "token environment" in overview.next_priority
    assert overview.next_safe_command is None
    assert overview.commands_executed is False


def test_stale_injected_readiness_never_claims_live_observation(
    tmp_path: Path,
) -> None:
    old = QualificationReadiness(
        captured_utc=NOW - timedelta(hours=2),
        environment_ready=True, primary_node_id="gpu-b",
        checks=(
            ReadinessCheck(
                target="pc_a", name="preflight:llm_lan",
                status="fail", reason="Old report",
            ),
        ),
    )
    overview = compile_qualification_overview(
        _settings(tmp_path / "evidence"), readiness=old, now=NOW,
    )
    assert overview.already_sampled_read_only == ()
    assert "refresh-readiness" in overview.unavailable_read_only
    assert overview.environment_ready is False
    assert overview.production_qualified is False

def test_explicit_session_absence_and_mismatching_injected_stage_are_errors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    with pytest.raises(ValueError):
        compile_qualification_overview(
            _settings(root), session_id="not-existing",
            readiness=_report(ready=True), now=NOW,
        )
    selected = "20261009T120000Z-bbbbbbbb"
    _session(root, selected)
    with pytest.raises(ValueError, match="does not include"):
        compile_qualification_overview(
            _settings(root), session_id=selected,
            readiness=_report(ready=True), now=NOW,
        )
