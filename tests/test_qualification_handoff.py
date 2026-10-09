from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.deployment import DeploymentCheck, DeploymentReport
from artifex.qualification.handoff import (
    QualificationHandoff,
    compile_handoff,
    inspect_qualification_handoff,
)
from artifex.qualification.overview import compile_qualification_overview
from artifex.qualification.readiness_diagnostics import (
    QualificationReadiness,
    ReadinessCheck,
)

NOW = datetime(2026, 10, 9, 14, 0, tzinfo=UTC)
CONTROLLER = Path("C:/AI/artifex/config/local.yaml")
EXPECTED = ("production", "repair")


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.comfyui.default_template = EXPECTED[0]
    settings.production.repair_workflow_template = EXPECTED[1]
    return settings


def _overview(*, ready: bool = True, stale: bool = False) -> Any:
    settings = _settings()
    sample = QualificationReadiness(
        captured_utc=NOW - timedelta(minutes=6) if stale else NOW,
        primary_node_id="gpu-b",
        environment_ready=ready,
        checks=(
            ReadinessCheck(
                target="pc_a", name="native_windows",
                status="pass" if ready else "fail",
                reason="CI fixture: a readiness record is not machine proof",
            ),
        ),
    )
    return compile_qualification_overview(
        settings, readiness=sample, now=NOW, controller_config=CONTROLLER,
    )


def _deployment(
    *,
    ready: bool = True,
    workflows: tuple[str, ...] = EXPECTED,
    role: str = "controller",
    controller_present: bool = True,
) -> DeploymentReport:
    checks: list[DeploymentCheck] = []
    if controller_present:
        checks.append(DeploymentCheck(
            name="controller:comfyui_lan", blocking=True,
            ready=ready, detail="fake authenticated health",
        ))
    checks.extend(
        DeploymentCheck(
            name="workflow:" + name, blocking=True,
            ready=ready, detail="fake node manifest",
        )
        for name in workflows
    )
    return DeploymentReport(
        role=role, ready=ready, checks=tuple(checks),
    )


def test_handoff_requires_two_independent_live_gates_and_suggests_only_session_start(
) -> None:
    report = compile_handoff(
        _overview(), _deployment(),
        controller_config=CONTROLLER,
        expected_workflow_ids=EXPECTED,
    )
    assert report.state == "session_start_candidate"
    assert report.environment_ready
    assert report.deployment_workflows_ready
    assert report.can_suggest_qualification_start
    assert report.saved_session_id is None
    assert report.advisory_argv == (
        "uv", "run", "artifex", "qualify", "start",
        "--config", str(CONTROLLER), "--json",
    )
    assert report.blockers == ()
    assert len(report.remaining_native_evidence) == 2
    assert report.saved_session_not_assumed_active
    assert report.actual_eight_hour_soak_verified is False
    assert report.production_qualified is False
    assert report.renderer_start_authorized is False
    assert report.shell_commands_executed is False
    assert report.gpu_jobs_submitted is False


@pytest.mark.parametrize("reason", [
    "environment",
    "stale",
    "deployment",
    "missing_template",
    "unexpected_template",
    "duplicate_template",
    "missing_controller",
    "inconsistent_flag",
])
def test_handoff_fails_closed_for_partial_or_conflicting_readiness(reason: str) -> None:
    overview = _overview(ready=reason != "environment", stale=reason == "stale")
    deployment = _deployment(
        ready=reason != "deployment",
        workflows=(
            ("production",)
            if reason == "missing_template"
            else ("production", "wrong")
            if reason == "unexpected_template"
            else ("production", "production")
            if reason == "duplicate_template"
            else EXPECTED
        ),
        controller_present=reason != "missing_controller",
    )
    if reason == "inconsistent_flag":
        deployment = deployment.model_copy(update={"ready": False})
    report = compile_handoff(
        overview, deployment, controller_config=CONTROLLER,
        expected_workflow_ids=EXPECTED,
    )
    assert report.state == "blocked"
    assert report.blockers
    assert not report.can_suggest_qualification_start
    assert report.advisory_argv is None
    assert not report.renderer_start_authorized
    assert not report.production_qualified
    if reason in {"missing_template", "unexpected_template", "duplicate_template"}:
        assert "deployment:workflow_requirements_missing_or_failed" in report.blockers


def test_existing_saved_session_is_only_reviewed_not_assumed_active() -> None:
    overview = _overview().model_copy(update={
        "session_id": "20261009T130000Z-deadbeef",
        "session_selection": "latest_saved",
        "next_safe_command": (
            "uv", "run", "artifex", "qualify", "collect", "20261009T130000Z-deadbeef",
            "--config", str(CONTROLLER), "--json",
        ),
    })
    report = compile_handoff(
        overview, _deployment(), controller_config=CONTROLLER,
        expected_workflow_ids=EXPECTED,
    )
    assert report.state == "saved_session_review"
    assert report.saved_session_id == "20261009T130000Z-deadbeef"
    assert report.saved_session_not_assumed_active
    assert report.advisory_argv == overview.next_safe_command
    assert report.advisory_argv is not None and "--apply" not in report.advisory_argv
    assert report.overview.session_stages_are_revalidated is False
    assert report.production_qualified is False


def test_renderer_role_or_any_gpu_smoke_evidence_is_rejected() -> None:
    overview = _overview()
    with pytest.raises(ValueError, match="controller"):
        compile_handoff(
            overview, _deployment(role="renderer"),
            controller_config=CONTROLLER, expected_workflow_ids=EXPECTED,
        )
    with pytest.raises(ValueError, match="GPU smoke"):
        compile_handoff(
            overview,
            _deployment().model_copy(update={
                "checks": (*_deployment().checks, DeploymentCheck(
                    name="render_smoke", ready=True, blocking=True, detail="no",
                )),
            }),
            controller_config=CONTROLLER, expected_workflow_ids=EXPECTED,
        )
    with pytest.raises(ValueError, match="GPU smoke"):
        compile_handoff(
            overview, _deployment().model_copy(update={"smoke": object()}),
            controller_config=CONTROLLER, expected_workflow_ids=EXPECTED,
        )


@pytest.mark.asyncio
async def test_live_orchestrator_only_calls_existing_read_only_entry_points() -> None:
    observed: list[tuple[str, object]] = []

    def inspect(_: ArtifexSettings, **kwargs: object) -> Any:
        observed.append(("overview", kwargs))
        return _overview()

    async def deployment(_: ArtifexSettings, **kwargs: object) -> DeploymentReport:
        observed.append(("deployment", kwargs))
        return _deployment()

    result = await inspect_qualification_handoff(
        _settings(), controller_config=CONTROLLER,
        overview_fn=inspect, deployment_fn=deployment,
    )
    assert result.state == "session_start_candidate"
    assert len(observed) == 2
    assert observed[0][0] == "overview"
    assert observed[1][0] == "deployment"
    assert observed[1][1] == {
        "role": "controller",
        "require_autostart": False,
        "render_smoke": False,
    }


def test_cli_handoff_json_exclusive_report_and_blocked_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.cli as cli
    import artifex.qualification.handoff as handoff

    config = tmp_path / "local.yaml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_settings", lambda _: _settings())

    current = compile_handoff(
        _overview(), _deployment(), controller_config=config,
        expected_workflow_ids=EXPECTED,
    )

    async def fake_inspect(*args: object, **kwargs: object) -> QualificationHandoff:
        return current

    monkeypatch.setattr(handoff, "inspect_qualification_handoff", fake_inspect)
    output = tmp_path / "reports" / "handoff.json"
    args = [
        "qualify", "handoff", "--config", str(config),
        "--output", str(output), "--json",
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["state"] == "session_start_candidate"
    assert payload["gpu_jobs_submitted"] is False
    assert payload["production_qualified"] is False
    assert config.read_text(encoding="utf-8") == "{}\n"
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8")) == payload

    blocked = compile_handoff(
        _overview(ready=False), _deployment(),
        controller_config=config, expected_workflow_ids=EXPECTED,
    )

    async def fake_blocked(*args: object, **kwargs: object) -> QualificationHandoff:
        return blocked

    monkeypatch.setattr(handoff, "inspect_qualification_handoff", fake_blocked)
    third = CliRunner().invoke(
        app, ["qualify", "handoff", "--config", str(config), "--json"],
    )
    assert third.exit_code == 1
    parsed = json.loads(third.stdout)
    assert parsed["state"] == "blocked"
    assert parsed["advisory_argv"] is None
    assert parsed["production_qualified"] is False
