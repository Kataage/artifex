"""Passive owner survival: never terminate, restart or reparent a GPU process."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.render_node.supervisor_survival import (
    SurvivalSample,
    assess_survival,
    observe_supervisor_survival,
    sample_supervisor_survival,
    supervisor_process_identities,
)
from artifex.windows_tasks import StartupTaskStatus

_BASE = datetime(2026, 10, 10, tzinfo=UTC)
_SUPERVISOR = ((123, "2026-10-10T00:00:00Z"),)
_REQ = (
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
)


def _sample(
    second: float, *, task_state: str = "Running",
    owner_pid: int = 400, owner_creation: str = "2026-10-09T22:00:00Z",
    supervisors: tuple[tuple[int, str], ...] | None = None,
    state: str = "verified",
) -> SurvivalSample:
    if supervisors is None:
        supervisors = _SUPERVISOR if task_state == "Running" else ()
    return SurvivalSample.model_validate({
        "observed_utc": (_BASE + timedelta(seconds=second)).isoformat(),
        "elapsed_seconds": second,
        "host": "pc-b", "node_id": "gpu-b",
        "state": state,
        "task_state": task_state,
        "scheduler_policy_verified": True,
        "supervisor_identities": supervisors,
        "original_supervisor_pids_absent": task_state == "Ready",
        "owner_audit_state": (
            "observed_stable" if task_state == "Running" else "inconclusive"
        ),
        "owner_checks": {
            key: "unknown" if key == "scheduler_running" and task_state != "Running"
            else "pass" for key in _REQ
        },
        "owner_process_verified": True,
        "actual_listener_pid": owner_pid,
        "actual_listener_started_utc": owner_creation,
        "launcher_pid": 399, "receipt_schema": 2,
    })


def test_natural_transition_is_only_observed_not_production_qualified() -> None:
    report = assess_survival((
        _sample(0), _sample(10), _sample(20, task_state="Ready"),
        _sample(30, task_state="Ready"),
    ), min_separation_seconds=10)
    assert report.status == "observed_after_supervisor_absence"
    assert report.same_comfyui_seen_before_and_after
    assert report.supervisor_absence_observed
    assert report.samples[-1].actual_listener_pid == 400
    assert report.child_survival_qualified is False
    assert report.task_actions_executed is False
    assert report.renderer_restart_authorized is False
    assert report.GPU_jobs_submitted is False
    assert report.services_mutated is False
    assert report.issue_93_closure_authorized is False
    assert report.production_qualified is False


@pytest.mark.parametrize("scenario,expected", (
    ("no_initial_supervisor", "blocked"),
    ("initial_ready", "blocked"),
    ("missing_initial_receipt", "blocked"),
    ("no_transition", "inconclusive"),
    ("only_one_absence", "inconclusive"),
    ("pid_reused", "blocked"),
    ("started_changed", "blocked"),
    ("launcher_changed", "blocked"),
    ("unexpected_supervisor", "blocked"),
    ("supervisor_reappeared", "blocked"),
    ("owner_incomplete", "blocked"),
    ("missing_comfy_listener", "blocked"),
    ("fake_task_ready", "blocked"),
    ("original_pid_still_live", "blocked"),
    ("clock_reused", "blocked"),
    ("different_host", "blocked"),
    ("unsupported", "blocked"),
))
def test_rejects_unproven_or_contradictory_lifecycle(
    scenario: str, expected: str,
) -> None:
    first = _sample(0)
    second = _sample(10, task_state="Ready")
    third = _sample(20, task_state="Ready")
    if scenario == "no_initial_supervisor":
        first = first.model_copy(update={"supervisor_identities": ()})
    if scenario == "initial_ready":
        first = _sample(0, task_state="Ready")
    if scenario == "missing_initial_receipt":
        first = first.model_copy(update={"owner_process_verified": False})
    if scenario == "no_transition":
        second, third = _sample(10), _sample(20)
    if scenario == "only_one_absence":
        second = _sample(10)
        third = _sample(20, task_state="Ready")
    if scenario == "pid_reused":
        second = second.model_copy(update={"actual_listener_pid": 401})
    if scenario == "started_changed":
        second = second.model_copy(update={
            "actual_listener_started_utc": "2026-10-10T00:00:01Z",
        })
    if scenario == "launcher_changed":
        second = second.model_copy(update={"launcher_pid": 200})
    if scenario == "unexpected_supervisor":
        second = second.model_copy(update={"supervisor_identities": _SUPERVISOR})
    if scenario == "supervisor_reappeared":
        third = _sample(20)
    if scenario == "owner_incomplete":
        second = second.model_copy(update={"owner_audit_state": "blocked"})
    if scenario == "missing_comfy_listener":
        second = second.model_copy(update={"state": "blocked"})
    if scenario == "original_pid_still_live":
        second = second.model_copy(update={"original_supervisor_pids_absent": False})
    if scenario == "fake_task_ready":
        second = second.model_copy(update={
            "owner_checks": {**second.owner_checks, "scheduler_running": "pass"},
        })
    if scenario == "clock_reused":
        third = _sample(10, task_state="Ready")
    if scenario == "different_host":
        second = second.model_copy(update={"host": "another"})
    if scenario == "unsupported":
        first = first.model_copy(update={"state": "unsupported"})
    report = assess_survival(
        (first, second, third), min_separation_seconds=10,
    )
    assert report.status == expected
    assert report.production_qualified is False
    assert report.child_survival_qualified is False


def test_observation_is_passive_writes_once_and_never_overwrites(
    tmp_path: Path,
) -> None:
    clock = [0.0]
    samples = [_sample(0), _sample(10, task_state="Ready"), _sample(20, task_state="Ready")]
    count: list[int] = []
    def snapshot(
        settings: ArtifexSettings, *, config: Path,
        elapsed_seconds: float, original_supervisor_ids: tuple[int, ...],
        now: Any,
    ) -> SurvivalSample:
        assert config == cfg
        if not count:
            assert original_supervisor_ids == ()
        else:
            assert original_supervisor_ids == (123,)
        count.append(1)
        return samples[len(count) - 1]

    def advance(seconds: float) -> None:
        clock[0] += seconds

    cfg = tmp_path / "render-node.yaml"
    cfg.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "logs" / "passive.json"
    assessment = observe_supervisor_survival(
        ArtifexSettings(), config=cfg, output=output,
        duration_seconds=30, interval_seconds=10,
        sample=snapshot, monotonic=lambda: clock[0],
        sleep=advance, now=lambda: _BASE + timedelta(seconds=clock[0]),
    )
    assert assessment.status == "observed_after_supervisor_absence"
    assert count == [1, 1, 1]
    assert json.loads(output.read_text())["production_qualified"] is False
    with pytest.raises(FileExistsError):
        observe_supervisor_survival(
            ArtifexSettings(), config=cfg, output=output,
            duration_seconds=30, interval_seconds=10,
            sample=snapshot,
        )


def test_observer_without_exit_stays_inconclusive(tmp_path: Path) -> None:
    clock = [0.0]
    cfg = tmp_path / "renderer.yaml"
    cfg.write_text("{}", encoding="utf-8")
    evidence = tmp_path / "inconclusive.json"
    def advance(seconds: float) -> None:
        clock[0] += seconds

    report = observe_supervisor_survival(
        ArtifexSettings(), config=cfg, output=evidence,
        duration_seconds=20, interval_seconds=10,
        sample=lambda *a, **kw: _sample(clock[0]),
        monotonic=lambda: clock[0], sleep=advance,
    )
    assert report.status == "inconclusive"
    assert len(report.samples) == 3
    assert evidence.exists()


def test_sample_probes_are_read_only_and_require_actual_comfy_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.render_node.supervisor_survival as module

    settings = ArtifexSettings()
    cfg = tmp_path / "render-node.yaml"
    cfg.write_text("{}\n", encoding="utf-8")
    task = StartupTaskStatus(
        role="renderer", task_name="Artifex-Renderer",
        installed=True, managed=True, state="Ready",
        arguments="-m artifex.cli render-node serve --config C:\\\\AI\\\\render-node.yaml",
    )
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    events: list[str] = []
    def task_call(role: str) -> StartupTaskStatus:
        events.append("task_status")
        return task

    def matching(role: str, *, config: Path, status: StartupTaskStatus):
        events.append("task_configuration_matches")
        return True, ()

    def discover(status: StartupTaskStatus):
        events.append("supervisor_process_identities")
        return ()

    def audit(settings: ArtifexSettings, *, config: Path) -> dict[str, Any]:
        events.append("observe_renderer_owner")
        return {
            "status": "inconclusive",
            "checks": {key: {
                "status": "unknown" if key == "scheduler_running" else "pass"
            } for key in _REQ},
            "process_observation_verified": True, "receipt_schema": 2,
            "actual_listener_pid": 400,
            "actual_process_started_utc": "2026-10-09T22:00:00Z",
            "launcher_pid": 399, "restart_authorized": False,
            "child_survival_qualified": False,
            "production_qualified": False, "mutated_services": False,
        }

    monkeypatch.setattr(module, "task_status", task_call)
    monkeypatch.setattr(module, "task_configuration_matches", matching)
    monkeypatch.setattr(module, "supervisor_process_identities", discover)
    monkeypatch.setattr(module, "observe_renderer_owner", audit)
    sample = sample_supervisor_survival(settings, config=cfg, now=lambda: _BASE)
    assert sample.state == "verified"
    assert sample.task_state == "Ready"
    assert sample.supervisor_identities == ()
    assert events == [
        "task_status", "task_configuration_matches",
        "supervisor_process_identities", "observe_renderer_owner",
    ]


def test_cim_inventory_matches_full_task_arguments_without_shell_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.render_node.supervisor_survival as module

    args = r'-m artifex.cli render-node serve --config "C:\AI\render-node.yaml"'
    task = StartupTaskStatus(
        role="renderer", task_name="Artifex-Renderer",
        installed=True, managed=True, state="Running",
        arguments=args,
    )
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    calls: list[object] = []
    def run(command: list[str], **kwargs: Any):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{
                "ProcessId": 123,
                "CreationDate": "2026-10-10T00:00:00Z",
                "ExecutablePath": r"C:\Python\python.exe",
                "CommandLine": r'"C:\Python\python.exe" ' + args,
                "ParentProcessId": 1,
            }, {
                "ProcessId": 124,
                "CreationDate": "2026-10-10T00:00:01Z",
                "ExecutablePath": r"C:\Python\python.exe",
                "CommandLine": r'"C:\Python\python.exe" -m wrong serve',
                "ParentProcessId": 1,
            }]),
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    identities = supervisor_process_identities(task)
    assert identities == ((123, "2026-10-10T00:00:00Z"),)
    assert calls[0][0][0] == "powershell.exe"
    assert args not in " ".join(calls[0][0])
    assert calls[0][1]["shell"] is False


def test_cli_missing_config_and_read_only_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from artifex import cli

    missing = tmp_path / "missing.yaml"
    outfile = tmp_path / "evidence.json"
    runner = CliRunner()
    failed = runner.invoke(app, [
        "render-node", "survival-observe", "--config", str(missing),
        "--output", str(outfile), "--duration-seconds", "20",
        "--interval-seconds", "10",
    ])
    assert failed.exit_code == 1
    assert not outfile.exists()

    cfg = tmp_path / "renderer.yaml"
    cfg.write_text("{}\n", encoding="utf-8")
    calls: list[tuple[Path, Path]] = []
    def fake(*args: Any, **kwargs: Any):
        calls.append((kwargs["config"], kwargs["output"]))
        return assess_survival((_sample(0),), min_separation_seconds=10)

    monkeypatch.setattr(cli, "_settings", lambda _: ArtifexSettings())
    monkeypatch.setattr(
        "artifex.render_node.supervisor_survival.observe_supervisor_survival", fake,
    )
    result = runner.invoke(app, [
        "render-node", "survival-observe", "--config", str(cfg),
        "--output", str(outfile), "--duration-seconds", "20",
        "--interval-seconds", "10", "--json",
    ])
    assert result.exit_code == 1
    assert calls == [(cfg, outfile)]
    assert json.loads(result.stdout)["status"] == "inconclusive"
