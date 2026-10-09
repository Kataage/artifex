"""The independent passive PC-B watcher never mutates the renderer."""
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
from artifex.render_node.supervisor_survival import SurvivalSample
from artifex.render_node.survival_watcher import (
    PassiveSurvivalWindow,
    run_passive_survival_watcher,
)
from artifex.windows_tasks import StartupTaskStatus, _command, _install_script

START = datetime(2026, 10, 10, tzinfo=UTC)
KEYS = (
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
)


def _sample(
    elapsed: float,
    *,
    state: str = "Running",
    pid: int = 400,
    original_missing: bool = True,
    listener_start: str = "2026-10-10T00:00:00Z",
) -> SurvivalSample:
    running = state == "Running"
    return SurvivalSample(
        observed_utc=START + timedelta(seconds=elapsed),
        elapsed_seconds=elapsed,
        host="pc-b",
        node_id="gpu-b",
        state="verified",
        task_state=state,
        scheduler_policy_verified=True,
        supervisor_identities=(
            ((100, "2026-10-10T00:00:00Z"),) if running else ()
        ),
        original_supervisor_pids_absent=(
            original_missing if not running else False
        ),
        owner_audit_state="observed_stable" if running else "inconclusive",
        owner_checks={
            key: ("unknown" if key == "scheduler_running" and not running else "pass")
            for key in KEYS
        },
        owner_process_verified=True,
        actual_listener_pid=pid,
        actual_listener_started_utc=listener_start,
        launcher_pid=399,
        receipt_schema=2,
    )


def test_passive_watcher_accumulates_real_identity_and_reports_natural_exit() -> None:
    window = PassiveSurvivalWindow(interval_seconds=10, window_seconds=60)
    assert window.ingest(_sample(0), monotonic_seconds=100) is None
    assert window.original_pids == (100,)
    assert window.ingest(_sample(10), monotonic_seconds=110) is None
    assert window.ingest(_sample(20, state="Ready"), monotonic_seconds=120) is None
    report = window.ingest(_sample(30, state="Ready"), monotonic_seconds=130)
    assert report is not None
    assert report.status == "observed_after_supervisor_absence"
    assert len(report.samples) == 4
    assert report.child_survival_qualified is False
    assert report.renderer_restart_authorized is False
    assert report.production_qualified is False
    assert window.original_pids == ()


def test_watcher_waits_for_valid_running_then_does_not_write_idle_reports() -> None:
    window = PassiveSurvivalWindow(interval_seconds=10, window_seconds=20)
    assert window.ingest(_sample(0, state="Ready"), monotonic_seconds=0) is None
    assert window.original_pids == ()
    assert window.ingest(_sample(10), monotonic_seconds=10) is None
    assert window.ingest(_sample(20), monotonic_seconds=20) is None
    assert window.ingest(_sample(30), monotonic_seconds=30) is None
    assert window.original_pids == ()
    # Running baseline may be re-armed once rollover completes.
    assert window.ingest(_sample(40), monotonic_seconds=40) is None
    assert window.original_pids == (100,)


@pytest.mark.parametrize("issue", [
    "missing_supervisor", "identity_drift", "original_pid_still_present",
    "sampling_gap", "supervisor_reappears",
])
def test_watcher_rejects_ambiguous_observations(issue: str) -> None:
    window = PassiveSurvivalWindow(interval_seconds=10, window_seconds=100)
    assert window.ingest(_sample(0), monotonic_seconds=0) is None
    if issue == "missing_supervisor":
        broken = _sample(10, state="Ready").model_copy(update={
            "supervisor_identities": ((100, "2026-10-10T00:00:00Z"),),
        })
        at = 10
    elif issue == "identity_drift":
        broken = _sample(10, pid=999)
        at = 10
    elif issue == "original_pid_still_present":
        broken = _sample(10, state="Ready", original_missing=False)
        at = 10
    elif issue == "sampling_gap":
        broken = _sample(60)
        at = 60
    else:
        assert window.ingest(
            _sample(10, state="Ready"), monotonic_seconds=10,
        ) is None
        broken = _sample(20)
        at = 20
    report = window.ingest(broken, monotonic_seconds=at)
    assert report is not None
    assert report.status == "blocked"
    assert report.child_survival_qualified is False
    assert report.renderer_restart_authorized is False
    assert report.production_qualified is False


def test_two_confirmed_ready_samples_can_report_observed_but_not_qualified() -> None:
    window = PassiveSurvivalWindow(interval_seconds=10, window_seconds=20)
    assert window.ingest(_sample(0), monotonic_seconds=0) is None
    assert window.ingest(_sample(10, state="Ready"), monotonic_seconds=10) is None
    report = window.ingest(_sample(20, state="Ready").model_copy(update={
        "original_supervisor_pids_absent": True,
    }), monotonic_seconds=20)
    # At separation exactly 10 seconds, it is a real positive observation.
    assert report is not None
    assert report.status == "observed_after_supervisor_absence"
    assert report.child_survival_qualified is False


def test_service_process_samples_without_influencing_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.render_node.survival_watcher as module

    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    settings = ArtifexSettings()
    settings.render_agent.node_id = "gpu-b"
    cfg = tmp_path / "renderer.yaml"
    cfg.write_text("{}")
    monotonic_time = [0.0]
    observations = [
        _sample(0), _sample(10), _sample(20, state="Ready"),
        _sample(30, state="Ready"),
    ]
    pids: list[tuple[int, ...]] = []
    writes: list[tuple[dict[str, Any], Path]] = []
    heartbeats: list[int] = []
    def read(
        settings: ArtifexSettings, *, config: Path, elapsed_seconds: float,
        original_supervisor_ids: tuple[int, ...], now: Any,
    ) -> SurvivalSample:
        assert config == cfg
        pids.append(original_supervisor_ids)
        return observations[len(pids) - 1]

    def advance(seconds: float) -> None:
        monotonic_time[0] += seconds

    report = run_passive_survival_watcher(
        settings,
        config=cfg,
        poll_seconds=10,
        window_seconds=60,
        max_seconds=30,
        sample=read,
        monotonic=lambda: monotonic_time[0],
        sleep=advance,
        output=lambda settings: tmp_path / "evidence.json",
        persist=lambda data, path: (writes.append((data, path)) or path),
        publish_heartbeat=lambda *_args, **kwargs: heartbeats.append(
            kwargs["sample_count"]
        ),
    )
    assert pids == [(), (100,), (100,), (100,)]
    assert heartbeats == [1, 2, 3, 4]
    assert report["saved_traces"] == 1
    assert report["samples"] == 4
    assert report["last_event"] == "observed_after_supervisor_absence"
    assert report["gpu_jobs_submitted"] is False
    assert report["scheduler_actions_executed"] is False
    assert report["services_mutated"] is False
    assert report["production_qualified"] is False
    assert len(writes) == 1
    assert writes[0][0]["production_qualified"] is False


def test_native_separate_task_command_and_hard_terminate_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.windows_tasks as tasks

    config = tmp_path / "renderer.yaml"
    config.write_text("{}")
    monkeypatch.setattr(tasks.sys, "executable", __import__("sys").executable)
    execute, args, workdir = _command("survival-observer", config)
    assert "survival-watch" in args
    assert "render-node serve" not in args
    assert "daemon" not in args
    script = _install_script(
        "survival-observer",
        execute=execute, arguments=args, working_directory=workdir,
        replace=False, restart_count=10,
    )
    assert "Artifex-Survival-Observer" in script
    assert "DisallowHardTerminate = $true" in script
    assert "IgnoreNew" in script
    assert "Refusing to replace a running Artifex scheduled task" in script
    assert "Start-ScheduledTask" not in script
    assert "Stop-ScheduledTask" not in script
    assert "Artifex-Renderer" not in script


def _task(*, installed: bool) -> StartupTaskStatus:
    return StartupTaskStatus(
        role="survival-observer",
        task_name="Artifex-Survival-Observer",
        installed=installed, managed=installed,
        state="Ready" if installed else None,
    )


def test_enable_dry_run_does_not_register_or_start_any_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from artifex import cli

    cfg = tmp_path / "render-node.yaml"
    cfg.write_text("{}")
    settings = ArtifexSettings()
    settings.render_agent.comfyui_process.enabled = True
    monkeypatch.setattr(cli, "_settings", lambda _: settings)
    monkeypatch.setattr(cli, "task_status", lambda role: _task(installed=False))
    def should_not_run(*args, **kwargs):
        pytest.fail("Dry run must not write or start a Task Scheduler task")

    monkeypatch.setattr(cli, "activate_task", should_not_run)
    result = CliRunner().invoke(app, [
        "startup", "observer-enable", "--config", str(cfg), "--json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "read_only_plan"
    assert payload["action"] == "would_register_and_start_observer_only"
    assert payload["comfyui_process_modified"] is False
    assert payload["renderer_task_modified"] is False


def test_enable_apply_targets_only_observer_and_refuses_foreign_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from artifex import cli

    cfg = tmp_path / "render-node.yaml"
    cfg.write_text("{}")
    settings = ArtifexSettings()
    settings.render_agent.comfyui_process.enabled = True
    monkeypatch.setattr(cli, "_settings", lambda _: settings)
    monkeypatch.setattr(cli, "task_status", lambda role: _task(installed=False))
    roles: list[tuple[str, bool, bool]] = []
    def activate(role: str, *, config: Path, install_missing: bool, replace: bool):
        assert config == cfg
        roles.append((role, install_missing, replace))
        return _task(installed=True), "installed_and_started"

    monkeypatch.setattr(cli, "activate_task", activate)
    result = CliRunner().invoke(app, [
        "startup", "observer-enable", "--config", str(cfg),
        "--apply", "--json",
    ])
    assert result.exit_code == 0, result.output
    assert roles == [("survival-observer", True, False)]
    assert json.loads(result.stdout)["renderer_task_modified"] is False

    unowned = _task(installed=True).model_copy(update={"managed": False})
    monkeypatch.setattr(cli, "task_status", lambda role: unowned)
    denied = CliRunner().invoke(app, [
        "startup", "observer-enable", "--config", str(cfg),
        "--apply", "--json",
    ])
    assert denied.exit_code == 1
    assert roles == [("survival-observer", True, False)]


def test_watcher_cli_rejects_missing_config_before_any_probes(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(app, [
        "render-node", "survival-watch", "--config",
        str(tmp_path / "missing.yaml"), "--max-seconds", "20",
    ])
    assert result.exit_code == 1
