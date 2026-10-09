from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.qualification.readiness_action_plan import RemediationStep
from artifex.qualification.readiness_diagnostics import (
    QualificationReadiness,
    ReadinessCheck,
    load_saved_readiness,
)
from artifex.qualification.readiness_recheck import (
    ReadOnlyReconciliation,
    _observed,
    reconcile_read_only_checks,
)

NOW = datetime(2026, 10, 9, 8, 0, tzinfo=UTC)


def _check(target: str, name: str, status: str) -> ReadinessCheck:
    return ReadinessCheck.model_validate({
        "target": target, "name": name, "status": status,
        "reason": "observed in test", "next_action": "manual only",
    })


def _report(
    *checks: ReadinessCheck,
    now: datetime = NOW, ready: bool = False,
) -> QualificationReadiness:
    return QualificationReadiness(
        captured_utc=now, primary_node_id="gpu-b",
        environment_ready=ready, checks=checks,
        pending_stages=("overnight_soak",),
    )


def _reconcile(
    prior: QualificationReadiness | None,
    current: QualificationReadiness,
) -> tuple[ReadOnlyReconciliation, list[int]]:
    called: list[int] = []

    def probe(settings: ArtifexSettings, *, session_id: str | None) -> QualificationReadiness:
        called.append(1)
        assert session_id is None
        return current

    result = reconcile_read_only_checks(
        ArtifexSettings(), previous=prior, now=NOW, live_probe=probe,
    )
    return result, called


def test_recheck_executes_one_fixed_live_probe_and_removes_resolved_work() -> None:
    previous = _report(
        _check("pc_a", "preflight:llm_lan", "fail"),
        _check("pc_b", "owner:tcp_ownership", "unknown"),
        _check("pc_b", "owner:overall", "fail"),
    )
    current = _report(
        _check("pc_a", "preflight:llm_lan", "pass"),
        _check("pc_b", "owner:freshness", "pass"),
        _check("pc_b", "owner:overall", "pass"),
        _check("pc_b", "owner:tcp_ownership", "pass"),
        now=NOW, ready=True,
    )
    result, called = _reconcile(previous, current)
    assert len(called) == 1
    assert result.before_source == "saved"
    assert result.latest_environment_ready
    assert result.updated_plan.environment_ready_observed
    assert result.updated_plan.blocking_checks == ()
    assert result.updated_plan.steps[0].id == "14-stage-real-machine-evidence"
    observed = {x.step_id: x for x in result.observations}
    assert observed["pc-a-preflight"].state == "observed_live"
    assert observed["pc-b-owner"].state == "observed_live"
    assert all(x.cli_commands_executed is False for x in result.observations)
    changes = {x.check_name: x for x in result.changes}
    assert changes["pc_a:preflight:llm_lan"].kind == "improved"
    assert changes["pc_b:owner:tcp_ownership"].kind == "improved"
    assert result.production_qualified is False
    assert result.gpu_soak_qualified is False
    assert result.arbitrary_plan_commands_executed is False
    assert result.mutated_services is False


def test_recheck_fails_closed_when_ownership_unavailable_now() -> None:
    previous = _report(
        _check("pc_b", "owner:overall", "pass"),
        _check("pc_b", "owner:freshness", "pass"),
        ready=True,
    )
    current = _report(
        _check("pc_b", "owner:remote_probe", "unknown"),
        ready=False,
    )
    result, _ = _reconcile(previous, current)
    assert not result.latest_environment_ready
    assert any(c.kind == "unavailable" for c in result.changes)
    assert "pc_b:owner:remote_probe" in result.updated_plan.blocking_checks
    assert result.updated_plan.production_qualified is False



def test_pc_a_remote_safety_is_acknowledged_only_when_both_probes_return() -> None:
    previous = _report(
        _check("pc_b", "safety:overall", "fail"),
        _check("pc_b", "safety:three_ports", "fail"),
    )
    current = _report(
        _check("pc_b", "safety:overall", "pass"),
        _check("pc_b", "safety:three_ports", "pass"),
        _check("pc_b", "safety:freshness", "pass"),
        _check("pc_b", "safety:pid_consistency", "pass"),
        ready=True,
    )
    result, calls = _reconcile(previous, current)
    assert len(calls) == 1
    mapped = {item.step_id: item for item in result.observations}
    assert mapped["pc-a-remote-renderer-safety"].state == "observed_live"
    assert mapped["pc-a-remote-renderer-safety"].cli_commands_executed is False
    assert result.arbitrary_plan_commands_executed is False
    assert result.production_qualified is False

    missing = _report(
        _check("pc_b", "safety:remote_probe", "unknown"),
        ready=False,
    )
    blocked, _ = _reconcile(previous, missing)
    item = next(
        x for x in blocked.observations
        if x.step_id == "pc-a-remote-renderer-safety"
    )
    assert item.state == "unavailable"
    assert not blocked.latest_environment_ready


def test_remote_inventory_is_never_called_a_native_pc_b_preflight() -> None:
    previous = _report(
        _check("pc_b", "preflight:render_inventory", "fail"),
    )
    current = _report(
        _check("pc_b", "preflight:render_inventory", "pass"),
        ready=True,
    )
    result, _ = _reconcile(previous, current)
    local = next(o for o in result.observations if o.step_id == "pc-b-inventory")
    assert local.state == "requires_local_pc_b"
    assert local.cli_commands_executed is False
    assert result.updated_plan.environment_ready_observed
    assert result.production_qualified is False


def test_unknown_steps_are_never_executed_even_when_named_read_only() -> None:
    step = RemediationStep(
        id="remote-arbitrary-command",
        role="pc_b",
        safety="read_only",
        blocked_checks=("pc_b:owner:overall",),
        argv=("powershell", "-Command", "Remove-Item C:\\"),
        description="untrusted step name",
    )
    outcome = _observed(
        step, _report(_check("pc_b", "owner:overall", "pass")),
    )
    assert outcome.state == "not_executed"
    assert outcome.cli_commands_executed is False


def test_duplicate_check_names_cannot_hide_failure() -> None:
    previous = _report(
        _check("pc_b", "owner:overall", "pass"),
        _check("pc_b", "owner:overall", "fail"),
    )
    current = _report(_check("pc_b", "owner:overall", "pass"), ready=True)
    result, _ = _reconcile(previous, current)
    assert any(c.kind == "improved" for c in result.changes)
    assert result.production_qualified is False


def test_no_saved_baseline_requires_one_probe_and_no_false_history() -> None:
    result, called = _reconcile(
        None, _report(_check("pc_a", "preflight:llm_lan", "fail")),
    )
    assert len(called) == 1
    assert result.before_source == "live"
    assert result.changes == ()
    assert not result.latest_environment_ready
    assert result.latest_readiness.actual_machine_qualification_complete is False


def test_saved_snapshot_reader_reuses_existing_bounded_policy(tmp_path: Path) -> None:
    target = tmp_path / "readiness.json"
    record = _report(_check("pc_b", "owner:tcp_ownership", "unknown"))
    target.write_text(record.model_dump_json(), encoding="utf-8")
    assert load_saved_readiness(target) == record
    target.write_bytes(b"Z" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="too large"):
        load_saved_readiness(target)
    target.unlink()
    actual = tmp_path / "original.json"
    actual.write_text(record.model_dump_json(), encoding="utf-8")
    try:
        target.symlink_to(actual)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink not supported")
    with pytest.raises(ValueError, match="symlink"):
        load_saved_readiness(target)


def test_cli_saved_recheck_reads_no_action_argv_and_never_clobbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.cli as cli_module
    import artifex.qualification.readiness_recheck as recheck_module

    config = tmp_path / "controller.yaml"
    config.write_text("{}\n", encoding="utf-8")
    saved = tmp_path / "saved.json"
    previous = _report(
        _check("pc_b", "owner:tcp_ownership", "fail"),
    )
    saved.write_text(previous.model_dump_json(), encoding="utf-8")
    expected, _ = _reconcile(
        previous, _report(
            _check("pc_b", "owner:tcp_ownership", "unknown"),
        ),
    )
    seen: list[Any] = []

    def fake_recheck(settings: ArtifexSettings, **kwargs: Any) -> ReadOnlyReconciliation:
        seen.append(kwargs)
        assert kwargs["previous"] == previous
        return expected

    monkeypatch.setattr(cli_module, "_settings", lambda config: ArtifexSettings())
    monkeypatch.setattr(recheck_module, "reconcile_read_only_checks", fake_recheck)
    output = tmp_path / "reports" / "rechecked.json"
    args = [
        "qualify", "recheck", "--config", str(config),
        "--from-report", str(saved), "--output", str(output), "--json",
    ]
    run = CliRunner().invoke(app, args)
    assert run.exit_code == 1, run.output
    parsed = json.loads(run.output)
    assert parsed["production_qualified"] is False
    assert parsed["arbitrary_plan_commands_executed"] is False
    assert len(seen) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == parsed
    rerun = CliRunner().invoke(app, args)
    assert rerun.exit_code == 1
    assert "FileExistsError" in rerun.output


def test_cli_no_config_prevents_probe_and_symlink_report_is_rejected(
    tmp_path: Path,
) -> None:
    cli = CliRunner()
    config = tmp_path / "missing.yaml"
    missing = cli.invoke(app, ["qualify", "recheck", "--config", str(config)])
    assert missing.exit_code == 1
    assert "configuration is required" in missing.output
    config.write_text("{}\n", encoding="utf-8")
    actual = tmp_path / "real.json"
    actual.write_text(_report().model_dump_json(), encoding="utf-8")
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(actual)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink not supported")
    rejected = cli.invoke(
        app, [
            "qualify", "recheck", "--config", str(config),
            "--from-report", str(alias),
        ],
    )
    assert rejected.exit_code == 1
    assert "symlink" in rejected.output
