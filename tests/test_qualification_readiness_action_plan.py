from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.qualification.readiness_action_plan import (
    QualificationActionPlan,
    compile_qualification_action_plan,
)
from artifex.qualification.readiness_diagnostics import (
    QualificationReadiness,
    ReadinessCheck,
)

NOW = datetime(2026, 10, 9, 8, 0, tzinfo=UTC)


def _check(
    role: str, name: str, status: str = "fail",
) -> ReadinessCheck:
    return ReadinessCheck.model_validate({
        "target": role, "name": name, "status": status,
        "reason": "unproven",
        "next_action": "manual resolution",
    })


def _report(
    *checks: ReadinessCheck, at: datetime = NOW,
    ready: bool = False,
) -> QualificationReadiness:
    return QualificationReadiness(
        captured_utc=at, primary_node_id="gpu-b",
        environment_ready=ready, checks=checks,
        pending_stages=("overnight_soak", "archive_reproduction"),
    )


def test_plan_groups_pc_a_pc_b_and_never_executes_or_qualifies() -> None:
    report = _report(
        _check("pc_a", "remote_owner_auth"),
        _check("pc_a", "selected_gguf"),
        _check("pc_a", "preflight:llm_lan"),
        _check("pc_b", "owner:tcp_ownership", "unknown"),
        _check("pc_b", "owner:process_identity"),
        _check("pc_b", "preflight:asset_vae"),
        _check("pc_b", "preflight:comfyui_lan"),
        _check("qualification", "session_records", "unknown"),
    )
    plan = compile_qualification_action_plan(
        report,
        controller_config=Path("C:/Artifex folder/controller config.yaml"),
        renderer_config=Path("D:/AI/ComfyUI/render-node.yaml"),
        now=NOW,
    )
    assert not plan.environment_ready_observed
    assert plan.snapshot_fresh
    assert not plan.production_qualified
    assert not plan.gpu_soak_qualified
    assert not plan.mutated_services
    assert plan.outstanding_stages == ("overnight_soak", "archive_reproduction")
    steps = {step.id: step for step in plan.steps}
    assert {
        "pc-a-config", "pc-a-gguf", "pc-a-preflight", "pc-b-owner",
        "pc-b-connectivity", "pc-b-inventory", "14-stage-real-machine-evidence",
    }.issubset(steps)
    assert steps["pc-a-config"].safety == "review_required"
    assert steps["pc-b-owner"].safety == "read_only"
    assert steps["pc-b-owner"].argv == (
        "uv", "run", "artifex", "render-node", "owner-audit",
        "--config", "D:/AI/ComfyUI/render-node.yaml", "--json",
    )
    assert steps["pc-a-preflight"].argv is not None
    assert steps["pc-a-preflight"].argv[-2:] == (
        "C:/Artifex folder/controller config.yaml", "--json",
    )
    assert steps["14-stage-real-machine-evidence"].argv is None
    assert all(step.automatically_executed is False for step in plan.steps)
    assert plan.read_only_steps >= 3
    assert all(x in plan.blocking_checks for x in (
        "pc_b:owner:process_identity", "pc_b:owner:tcp_ownership",
    ))


def test_no_gaps_still_requires_actual_qualification() -> None:
    plan = compile_qualification_action_plan(
        _report(_check("pc_a", "uv_available", "pass"), ready=True),
        now=NOW,
    )
    assert plan.environment_ready_observed
    assert len(plan.steps) == 1
    assert plan.steps[0].id == "14-stage-real-machine-evidence"
    assert plan.steps[0].safety == "real_machine_evidence"
    assert plan.production_qualified is False


@pytest.mark.parametrize("age_seconds", [301, -61])
def test_stale_or_future_snapshot_never_stays_ready(age_seconds: int) -> None:
    report = _report(at=NOW - timedelta(seconds=age_seconds), ready=True)
    plan = compile_qualification_action_plan(
        report, now=NOW, source="saved",
    )
    assert plan.source == "saved"
    assert plan.snapshot_fresh is False
    assert plan.environment_ready_observed is False
    assert any(step.id == "refresh-readiness" for step in plan.steps)
    assert any(step.safety == "read_only" for step in plan.steps)


def test_unclassified_checks_remain_accountable() -> None:
    report = _report(
        _check("pc_a", "new_dependency_A"),
        _check("pc_b", "new_scheduler_policy_z"),
    )
    plan = compile_qualification_action_plan(report, now=NOW)
    assert all(
        check in {b for s in plan.steps for b in s.blocked_checks}
        for check in plan.blocking_checks
    )
    assert any(
        s.safety == "review_required" and s.role == "pc_b"
        for s in plan.steps
    )
    assert all(
        s.argv is None
        for s in plan.steps if s.id.startswith("inspect-")
    )


def test_no_command_built_from_untrusted_check_detail() -> None:
    report = _report(
        ReadinessCheck(
            target="pc_b", name="owner:tcp_ownership", status="unknown",
            reason="'; Remove-Item -Recurse C:/ ;'",
        ),
    )
    plan = compile_qualification_action_plan(report, now=NOW)
    for step in plan.steps:
        if step.argv is not None:
            assert "Remove-Item" not in " ".join(step.argv)


def test_from_saved_report_offline_never_calls_live_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.cli as cli_module
    import artifex.qualification.readiness_diagnostics as diag_module

    path = tmp_path / "readiness.json"
    path.write_text(_report(
        _check("pc_a", "selected_gguf"),
        _check("pc_b", "owner:receipt"),
    ).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        diag_module, "diagnose_qualification_readiness",
        lambda *args, **kwargs: pytest.fail("Must not probe live machines"),
    )
    monkeypatch.setattr(
        cli_module, "_settings",
        lambda *args: pytest.fail("Must not load live config"),
    )
    args = [
        "qualify", "plan", "--from-report", str(path),
        "--config", str(tmp_path / "missing.yaml"),
        "--renderer-config", str(tmp_path / "renderer.yaml"),
        "--json",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    plan = QualificationActionPlan.model_validate_json(result.output)
    assert plan.source == "saved"
    assert plan.mutated_services is False
    assert any(step.id == "pc-b-owner" for step in plan.steps)
    assert all(step.automatically_executed is False for step in plan.steps)


def test_cli_no_clobber_and_session_safe_plan_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "readiness.json"
    path.write_text(_report(
        _check("pc_a", "preflight:llm_lan"),
    ).model_dump_json(), encoding="utf-8")
    output = tmp_path / "plans" / "actions.json"
    args = [
        "qualify", "plan", "--from-report", str(path),
        "--output", str(output), "--json",
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 1, first.output
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["production_qualified"] is False
    assert len(data["steps"]) >= 2
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 1
    assert "FileExistsError" in again.output
    assert json.loads(output.read_text(encoding="utf-8")) == data


def test_cli_rejects_missing_oversized_or_symlink_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.json"
    result = CliRunner().invoke(
        app, ["qualify", "plan", "--from-report", str(path)],
    )
    assert result.exit_code == 1
    assert "missing or too large" in result.output
    path.write_bytes(b" " * (1024 * 1024 + 1))
    result = CliRunner().invoke(
        app, ["qualify", "plan", "--from-report", str(path)],
    )
    assert result.exit_code == 1
    assert "too large" in result.output
    target = tmp_path / "target.json"
    target.write_text("{}")
    path.unlink()
    try:
        path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    result = CliRunner().invoke(
        app, ["qualify", "plan", "--from-report", str(path)],
    )
    assert result.exit_code == 1
    assert "symlinked" in result.output
