"""Safe, configurable one-command PC-A/PC-B diagnostic handoff advice."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.qualification.issue93_checklist import (
    Issue93Check,
    Issue93EvidenceChecklist,
)
from artifex.qualification.issue93_field_plan import compile_issue93_field_plan

PC_A = Path("D:/Artifex/PC A/controller custom.yaml")
PC_B = Path("E:/Artifex/PC B/renderer custom.yaml")
_CHECKS = (
    "isolated_launcher_and_live_owner",
    "two_time_separated_owner_samples",
    "natural_exit_survival_trace",
    "cross_evidence_listener_identity",
)


def _report(
    status: str = "needs_evidence",
    states: tuple[str, str, str, str] = (
        "missing", "missing", "missing", "missing",
    ),
) -> Issue93EvidenceChecklist:
    return Issue93EvidenceChecklist.model_validate({
        "checked_utc": datetime(2026, 10, 11, tzinfo=UTC),
        "node_id": "gpu-b",
        "status": status,
        "checks": [
            {"name": name, "state": state, "reason": "fixture"}
            for name, state in zip(_CHECKS, states, strict=True)
        ],
        "observed_checks": states.count("observed"),
        "missing_checks": states.count("missing"),
        "conflicting_checks": states.count("conflict"),
        "owner_readiness_status": "not_observed",
        "saved_owner_pair_status": "not_observed",
        "natural_exit_trace_status": "not_observed",
        "next_safe_actions": ["On-host acceptance remains outstanding"],
    })


def test_missing_evidence_proposes_exact_per_pc_diagnostics_without_apply() -> None:
    plan = compile_issue93_field_plan(
        _report(), controller_config=PC_A, renderer_config=PC_B,
    )
    assert {step.id for step in plan.steps} == {
        "pc-b-launcher-owner", "pc-a-fresh-owner-pair",
        "pc-b-observer-dry-run", "pc-a-survival-review",
    }
    assert all(step.role in {"pc_a", "pc_b"} for step in plan.steps)
    assert {step.effect for step in plan.steps} == {
        "read_only", "disposable_no_gpu_probe", "local_evidence_write",
    }
    for step in plan.steps:
        path = str(PC_A if step.role == "pc_a" else PC_B)
        assert step.argv[0:3] == ("uv", "run", "artifex")
        assert step.argv[-3:] == ("--config", path, "--json")
        assert "--apply" not in step.argv
        assert "--confirm" not in step.argv
        assert not step.executed
        assert not step.service_mutated
        assert not step.gpu_jobs_submitted
    assert "owner-readiness" in next(
        step.argv for step in plan.steps if step.id == "pc-b-launcher-owner"
    )
    assert "--refresh-owner-pair" in next(
        step.argv for step in plan.steps if step.id == "pc-a-fresh-owner-pair"
    )
    assert "--pc-b-survival-live" in next(
        step.argv for step in plan.steps if step.id == "pc-a-survival-review"
    )
    assert plan.pc_b_config_read_remotely is False
    assert plan.commands_executed is False
    assert plan.production_qualified is False
    assert plan.physical_pc_b_qualified is False


def test_unconfigured_uses_controller_first_run_without_probing_pc_b() -> None:
    result = compile_issue93_field_plan(
        _report("unconfigured"), controller_config=PC_A, renderer_config=PC_B,
    )
    assert len(result.steps) == 1
    assert result.steps[0].id == "pc-a-first-run"
    assert result.steps[0].role == "pc_a"
    assert "first-run" in result.steps[0].argv
    assert str(PC_A) in result.steps[0].argv
    assert str(PC_B) not in result.model_dump_json()


def test_conflict_requires_local_pc_b_owner_audit_not_any_restart() -> None:
    result = compile_issue93_field_plan(
        _report(
            "conflict",
            ("observed", "observed", "observed", "conflict"),
        ),
        controller_config=PC_A, renderer_config=PC_B,
    )
    assert [s.id for s in result.steps] == ["pc-b-owner-conflict"]
    assert "owner-audit" in result.steps[0].argv
    assert result.steps[0].effect == "read_only"
    assert "--apply" not in result.steps[0].argv


def test_all_correlated_still_has_real_host_production_pending() -> None:
    result = compile_issue93_field_plan(
        _report(
            "observations_correlated",
            ("observed", "observed", "observed", "observed"),
        ),
        controller_config=PC_A, renderer_config=PC_B,
    )
    assert len(result.steps) == 1
    assert result.steps[0].id == "pc-a-production-overview"
    assert "--pc-b-owner-live" in result.steps[0].argv
    assert "--pc-b-survival-live" in result.steps[0].argv
    assert not result.production_qualified
    assert not result.historical_supervisor_loss_authenticated


@pytest.mark.parametrize("part", [
    "isolated_launcher_and_live_owner",
    "two_time_separated_owner_samples",
    "natural_exit_survival_trace",
])
def test_only_failed_checks_generate_their_respective_commands(part: str) -> None:
    states = tuple("missing" if n == part else "observed" for n in _CHECKS)
    result = compile_issue93_field_plan(
        _report("needs_evidence", states),
        controller_config=PC_A, renderer_config=PC_B,
    )
    kinds = {s.id for s in result.steps}
    assert ("pc-b-launcher-owner" in kinds) == (
        part == "isolated_launcher_and_live_owner"
    )
    assert ("pc-a-fresh-owner-pair" in kinds) == (
        part == "two_time_separated_owner_samples"
    )
    assert ("pc-b-observer-dry-run" in kinds) == (
        part == "natural_exit_survival_trace"
    )


def test_issue93_cli_includes_custom_paths_without_running_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex import cli
    from artifex.qualification import issue93_checklist

    local = tmp_path / "controller.yaml"
    local.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_settings", lambda _: ArtifexSettings())
    monkeypatch.setattr(
        issue93_checklist, "compile_issue93_checklist",
        lambda settings: _report(),
    )
    output = CliRunner().invoke(app, [
        "qualify", "issue93-status",
        "--config", str(local),
        "--renderer-config", str(PC_B), "--json",
    ])
    assert output.exit_code == 1, output.output
    payload = json.loads(output.stdout)
    assert payload["status"] == "needs_evidence"
    plan = payload["field_plan"]
    assert plan["commands_executed"] is False
    assert plan["pc_b_config_read_remotely"] is False
    assert plan["production_qualified"] is False
    assert any(
        str(PC_B) in s["argv"] for s in plan["steps"] if s["role"] == "pc_b"
    )
    assert any(
        str(local) in s["argv"] for s in plan["steps"] if s["role"] == "pc_a"
    )
    assert all("--apply" not in s["argv"] for s in plan["steps"])
