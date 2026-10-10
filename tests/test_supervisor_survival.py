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


@pytest.mark.parametrize("failure", (
    "later_blocked_owner",
    "supervisor_reappeared",
    "listener_replaced",
    "invalid_creation_time",
    "replayed_observation",
))
def test_later_contradiction_clears_prior_survival_success_flags(
    failure: str,
) -> None:
    prefix = (
        _sample(0),
        _sample(10, task_state="Ready"),
        _sample(20, task_state="Ready"),
    )
    accepted = assess_survival(prefix, min_separation_seconds=10)
    assert accepted.status == "observed_after_supervisor_absence"
    assert accepted.same_comfyui_seen_before_and_after
    assert accepted.supervisor_absence_observed

    last = _sample(30, task_state="Ready")
    if failure == "later_blocked_owner":
        last = last.model_copy(update={"owner_process_verified": False})
    elif failure == "supervisor_reappeared":
        last = _sample(30)
    elif failure == "listener_replaced":
        last = last.model_copy(update={"actual_listener_pid": 401})
    elif failure == "invalid_creation_time":
        last = last.model_copy(update={
            "actual_listener_started_utc": "not a datetime",
        })
    else:
        last = last.model_copy(update={
            "observed_utc": _BASE + timedelta(seconds=20),
        })
    invalidated = assess_survival((*prefix, last), min_separation_seconds=10)
    assert invalidated.status == "blocked"
    assert not invalidated.same_comfyui_seen_before_and_after
    assert not invalidated.supervisor_absence_observed
    assert not invalidated.issue_93_closure_authorized
    assert not invalidated.production_qualified


@pytest.mark.parametrize("case", (
    "naive_baseline_observed",
    "naive_later_observed",
    "replayed_wall_clock",
    "reversed_wall_clock",
    "naive_listener_start",
    "malformed_listener_start",
    "future_listener_start",
    "future_supervisor_start",
    "naive_supervisor_start",
    "duplicate_supervisor_pid",
    "zero_supervisor_pid",
    "empty_hostname",
    "empty_node_id",
))
def test_passive_survival_rejects_untrusted_timeline(
    case: str,
) -> None:
    baseline = _sample(0)
    next_sample = _sample(10, task_state="Ready")
    last = _sample(20, task_state="Ready")
    if case == "naive_baseline_observed":
        baseline = baseline.model_copy(update={
            "observed_utc": _BASE.replace(tzinfo=None),
        })
    elif case == "naive_later_observed":
        next_sample = next_sample.model_copy(update={
            "observed_utc": (_BASE + timedelta(seconds=10)).replace(tzinfo=None),
        })
    elif case == "replayed_wall_clock":
        last = last.model_copy(update={
            "observed_utc": _BASE + timedelta(seconds=10),
        })
    elif case == "reversed_wall_clock":
        last = last.model_copy(update={
            "observed_utc": _BASE + timedelta(seconds=5),
        })
    elif case == "naive_listener_start":
        next_sample = next_sample.model_copy(update={
            "actual_listener_started_utc": "2026-10-09T22:00:00",
        })
        last = last.model_copy(update={
            "actual_listener_started_utc": "2026-10-09T22:00:00",
        })
        baseline = baseline.model_copy(update={
            "actual_listener_started_utc": "2026-10-09T22:00:00",
        })
    elif case == "malformed_listener_start":
        baseline = baseline.model_copy(update={
            "actual_listener_started_utc": "unparseable",
        })
    elif case == "future_listener_start":
        baseline = baseline.model_copy(update={
            "actual_listener_started_utc": "2026-10-10T03:00:00Z",
        })
    elif case == "future_supervisor_start":
        baseline = baseline.model_copy(update={
            "supervisor_identities": ((123, "2026-10-11T00:00:00Z"),),
        })
    elif case == "naive_supervisor_start":
        baseline = baseline.model_copy(update={
            "supervisor_identities": ((123, "2026-10-10T00:00:00"),),
        })
    elif case == "duplicate_supervisor_pid":
        baseline = baseline.model_copy(update={
            "supervisor_identities": ( _SUPERVISOR[0], _SUPERVISOR[0] ),
        })
    elif case == "zero_supervisor_pid":
        baseline = baseline.model_copy(update={
            "supervisor_identities": ((0, "2026-10-10T00:00:00Z"),),
        })
    elif case == "empty_hostname":
        baseline = baseline.model_copy(update={"host": ""})
    else:
        baseline = baseline.model_copy(update={"node_id": "  "})
    assessment = assess_survival(
        (baseline, next_sample, last), min_separation_seconds=10,
    )
    assert assessment.status == "blocked"
    assert not assessment.same_comfyui_seen_before_and_after
    assert not assessment.issue_93_closure_authorized
    assert not assessment.production_qualified


@pytest.mark.parametrize("clock_shift", (31.0, 3600.0))
def test_passive_survival_rejects_forward_clock_jump(
    clock_shift: float,
) -> None:
    first = _sample(0)
    second = _sample(10, task_state="Ready").model_copy(update={
        "observed_utc": _BASE + timedelta(seconds=10 + clock_shift),
    })
    third = _sample(20, task_state="Ready").model_copy(update={
        "observed_utc": _BASE + timedelta(seconds=20 + clock_shift),
    })
    report = assess_survival(
        (first, second, third), min_separation_seconds=10,
    )
    assert report.status == "blocked"
    assert report.reason == "sample_wall_clock_elapsed_diverged"
    assert not report.same_comfyui_seen_before_and_after
    assert not report.production_qualified


def test_passive_survival_rejects_slow_wall_clock_with_valid_order() -> None:
    # All UTC sample instants advance; their total duration still differs
    # from elapsed monotonic time by more than the permitted 30 seconds.
    samples = tuple(
        _sample(i * 10, task_state="Ready" if i >= 4 else "Running")
        .model_copy(update={"observed_utc": _BASE + timedelta(seconds=i)})
        for i in range(6)
    )
    report = assess_survival(samples, min_separation_seconds=10)
    assert report.status == "blocked"
    assert report.reason == "sample_wall_clock_elapsed_diverged"
    assert not report.supervisor_absence_observed


def test_passive_survival_permits_bounded_clock_difference() -> None:
    samples = (
        _sample(0),
        _sample(10, task_state="Ready").model_copy(update={
            "observed_utc": _BASE + timedelta(seconds=39),
        }),
        _sample(20, task_state="Ready").model_copy(update={
            "observed_utc": _BASE + timedelta(seconds=49),
        }),
    )
    report = assess_survival(samples, min_separation_seconds=10)
    assert report.status == "observed_after_supervisor_absence"
    assert not report.production_qualified


@pytest.mark.parametrize("separation", (0.0, -1.0, float("inf"), float("nan")))
def test_passive_survival_requires_positive_finite_interval(
    separation: float,
) -> None:
    with pytest.raises(ValueError, match="Positive finite"):
        assess_survival((_sample(0),), min_separation_seconds=separation)


@pytest.mark.parametrize("elapsed", (float("inf"), float("nan")))
def test_passive_survival_rejects_nonfinite_elapsed_even_in_model_copy(
    elapsed: float,
) -> None:
    first = _sample(0).model_copy(update={"elapsed_seconds": elapsed})
    report = assess_survival(
        (first, _sample(10, task_state="Ready"), _sample(20, task_state="Ready")),
        min_separation_seconds=10,
    )
    assert report.status == "blocked"
    assert not report.production_qualified


def test_valid_timezone_offsets_normalize_to_same_native_timeline() -> None:
    from datetime import timezone

    jp = timezone(timedelta(hours=9))
    baseline = _sample(0).model_copy(update={
        "observed_utc": _BASE.astimezone(jp),
        "actual_listener_started_utc": "2026-10-10T07:00:00+09:00",
        "supervisor_identities": ((123, "2026-10-10T09:00:00+09:00"),),
    })
    next_sample = _sample(10, task_state="Ready").model_copy(update={
        "observed_utc": (_BASE + timedelta(seconds=10)).astimezone(jp),
        "actual_listener_started_utc": "2026-10-10T07:00:00+09:00",
    })
    last = _sample(20, task_state="Ready").model_copy(update={
        "observed_utc": (_BASE + timedelta(seconds=20)).astimezone(jp),
        "actual_listener_started_utc": "2026-10-10T07:00:00+09:00",
    })
    # Normalize supervisor and listener creation times to identical identity
    # strings for the existing strict cross-observation identity check.
    assert baseline.observed_utc.astimezone(UTC) == _BASE
    assessment = assess_survival(
        (baseline, next_sample, last), min_separation_seconds=10,
    )
    assert assessment.status == "observed_after_supervisor_absence"
    assert not assessment.production_qualified


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


@pytest.mark.parametrize("timed_out_probe", (
    "scheduler",
    "supervisor_inventory",
    "owner_audit",
    "original_supervisor_pid",
))
def test_native_survival_probe_timeout_is_fail_closed_not_an_observer_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timed_out_probe: str,
) -> None:
    import artifex.render_node.supervisor_survival as module

    cfg = tmp_path / "render-node.yaml"
    cfg.write_text("{}\n", encoding="utf-8")
    task = StartupTaskStatus(
        role="renderer", task_name="Artifex-Renderer",
        installed=True, managed=True, state="Ready",
        arguments=r'-m artifex.cli render-node serve --config "C:\\AI\\renderer.yaml"',
    )
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    probes: list[str] = []

    def failure() -> None:
        raise module.subprocess.TimeoutExpired(
            cmd=["powershell.exe"], timeout=15,
        )

    def scheduler(role: str) -> StartupTaskStatus:
        probes.append("scheduler")
        if timed_out_probe == "scheduler":
            failure()
        return task

    def matching(role: str, *, config: Path, status: StartupTaskStatus):
        probes.append("policy")
        return True, ()

    def inventory(status: StartupTaskStatus):
        probes.append("inventory")
        if timed_out_probe == "supervisor_inventory":
            failure()
        return ()

    def audit(settings: ArtifexSettings, *, config: Path) -> dict[str, Any]:
        probes.append("audit")
        if timed_out_probe == "owner_audit":
            failure()
        return {
            "status": "inconclusive",
            "checks": {
                key: {"status": (
                    "unknown" if key == "scheduler_running" else "pass"
                )}
                for key in _REQ
            },
            "process_observation_verified": True,
            "receipt_schema": 2,
            "actual_listener_pid": 400,
            "actual_process_started_utc": "2026-10-09T22:00:00Z",
            "launcher_pid": 399,
            "restart_authorized": False,
            "child_survival_qualified": False,
            "production_qualified": False,
            "mutated_services": False,
        }

    def pid_identity(pid: int) -> None:
        probes.append("original_pid")
        assert pid == 123
        if timed_out_probe == "original_supervisor_pid":
            failure()
        return None

    monkeypatch.setattr(module, "task_status", scheduler)
    monkeypatch.setattr(module, "task_configuration_matches", matching)
    monkeypatch.setattr(module, "supervisor_process_identities", inventory)
    monkeypatch.setattr(module, "observe_renderer_owner", audit)
    monkeypatch.setattr(module, "windows_process_identity", pid_identity)

    sample = sample_supervisor_survival(
        ArtifexSettings(),
        config=cfg, elapsed_seconds=10.0,
        original_supervisor_ids=(123,),
        now=lambda: _BASE + timedelta(seconds=10),
    )
    assert "audit" in probes
    assert sample.state == (
        "verified" if timed_out_probe == "original_supervisor_pid"
        else "blocked"
    )
    if timed_out_probe == "original_supervisor_pid":
        assert sample.original_supervisor_pids_absent is False
        # An otherwise healthy Ready/ComfyUI snapshot cannot count as
        # original supervisor absence after a CIM lookup timed out.
        after = sample.model_copy(update={"host": "pc-b", "node_id": "gpu-b"})
        report = assess_survival(
            (_sample(0), after, _sample(20, task_state="Ready")),
            min_separation_seconds=10,
        )
        assert report.status == "blocked"
        assert report.reason == "original_supervisor_not_proven_absent"
        assert not report.same_comfyui_seen_before_and_after


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
