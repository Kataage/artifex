"""Independent Windows watcher heartbeat: bounded, atomic, read-only review."""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.config.models import ArtifexSettings
from artifex.render_node.process_identity import WindowsProcessIdentity
from artifex.render_node.observer_heartbeat import (
    _MAX_SIZE,
    SampleState,
    inspect_observer_heartbeat,
    publish_observer_heartbeat,
)

NOW = datetime(2026, 10, 10, 18, 30, tzinfo=UTC)


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_agent.node_id = "gpu-b"
    settings.render_agent.survival_evidence_dir = tmp_path / "survival-spool"
    return settings


def _emit(settings: ArtifexSettings, *, when: datetime = NOW,
          poll: float = 15, count: int = 1,
          state: SampleState = "verified") -> None:
    publish_observer_heartbeat(
        settings, observed_utc=when,
        poll_seconds=poll, sample_count=count, state=state,
    )


def _writer(pid: int, config: Path, *, started: str = "2026-10-09T00:00:00Z",
            exe: str = r"C:\\Python312\\python.exe",
            extra: tuple[str, ...] | None = None) -> WindowsProcessIdentity:
    argv = extra if extra is not None else (
        "-m", "artifex.cli", "render-node", "survival-watch",
        "--config", str(config.resolve()),
    )
    return WindowsProcessIdentity(
        ProcessId=pid, ParentProcessId=200,
        CreationDate=started, ExecutablePath=exe,
        CommandLine=subprocess.list2cmdline([exe, *argv]),
    )


def test_heartbeat_proves_recent_sampling_not_real_gpu(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert inspect_observer_heartbeat(settings, now=NOW) == "missing"
    _emit(settings)
    assert inspect_observer_heartbeat(settings, now=NOW) == "fresh"
    path = settings.render_agent.survival_evidence_dir / "observer-health.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["sample_count"] == 1
    assert data["production_qualified"] is False
    assert data["process_pid"] > 0
    assert "token" not in path.read_text().lower()
    _emit(settings, count=2)
    assert json.loads(path.read_text())["sample_count"] == 2
    assert list(settings.render_agent.survival_evidence_dir.iterdir()) == [path]


@pytest.mark.parametrize(("seconds", "expected"), [
    (30, "fresh"),
    (89, "fresh"),
    (91, "stale"),
    (-31, "stale"),
])
def test_monotonic_wall_time_staleness(
    tmp_path: Path, seconds: int, expected: str,
) -> None:
    settings = _settings(tmp_path)
    _emit(settings)
    assert inspect_observer_heartbeat(
        settings, now=NOW + timedelta(seconds=seconds),
    ) == expected


def test_long_configured_poll_interval_is_respected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _emit(settings, poll=300)
    assert inspect_observer_heartbeat(
        settings, now=NOW + timedelta(minutes=14),
    ) == "fresh"
    assert inspect_observer_heartbeat(
        settings, now=NOW + timedelta(minutes=16),
    ) == "stale"


@pytest.mark.parametrize("change", ["wrong_node", "missing_timezone", "oversized",
                                    "invalid_json", "false_qualification",
                                    "invalid_state", "missing_state"])
def test_invalid_heartbeat_is_not_treated_as_alive(
    tmp_path: Path, change: str,
) -> None:
    settings = _settings(tmp_path)
    _emit(settings)
    path = settings.render_agent.survival_evidence_dir / "observer-health.json"
    payload = json.loads(path.read_text())
    if change == "wrong_node":
        payload["node_id"] = "foreign-pc"
    elif change == "missing_timezone":
        payload["observed_utc"] = "2026-10-10T18:30:00"
    elif change == "false_qualification":
        payload["production_qualified"] = True
    elif change == "invalid_state":
        payload["last_sample_state"] = "running"
    elif change == "missing_state":
        payload.pop("last_sample_state")
    if change == "oversized":
        path.write_bytes(b"x" * (_MAX_SIZE + 1))
    elif change == "invalid_json":
        path.write_text("not JSON")
    else:
        path.write_text(json.dumps(payload))
    assert inspect_observer_heartbeat(settings, now=NOW) == "unsafe"


def test_symlinked_heartbeat_or_root_fails_closed(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    _emit(settings)
    root = settings.render_agent.survival_evidence_dir
    other = root / "safe.json"
    other.write_text("{}")
    target = root / "observer-health.json"
    target.unlink()
    try:
        target.symlink_to(other)
    except (OSError, NotImplementedError):
        pytest.skip("Windows runner may forbid symlink creation")
    assert inspect_observer_heartbeat(settings, now=NOW) == "unsafe"
    with pytest.raises(ValueError, match="symlinked"):
        _emit(settings)
    assert other.read_text() == "{}"


def test_pair_audit_rejects_running_scheduler_without_fresh_heartbeat(
    tmp_path: Path,
) -> None:
    from artifex.render_node.installation_audit import inspect_renderer_installation
    from artifex.windows_tasks import StartupTaskStatus

    settings = _settings(tmp_path)
    settings.render_agent.comfyui_process.enabled = True
    cfg = tmp_path / "renderer.yaml"
    cfg.write_text("{}")

    def probe(role: str) -> StartupTaskStatus:
        return StartupTaskStatus.model_validate({
            "role": role,
            "task_name": "Artifex-" + role,
            "installed": True,
            "managed": True,
            "state": "Running",
        })

    def verify(*args, **kwargs):
        return True, "all protected task settings match"

    initial = inspect_renderer_installation(
        settings, owner_config=cfg, native_windows=True, now=NOW,
        probe=probe, verify=verify,
        observer_process_probe=lambda pid: _writer(pid, cfg),
    )
    assert initial.renderer_task.status == "running"
    assert initial.survival_observer_task.status == "running"
    assert initial.observer_heartbeat == "missing"
    assert not initial.safe_for_passive_observation
    _emit(settings)
    active = inspect_renderer_installation(
        settings, owner_config=cfg, native_windows=True, now=NOW,
        probe=probe, verify=verify,
        observer_process_probe=lambda pid: _writer(pid, cfg),
    )
    assert active.observer_heartbeat == "fresh"
    assert active.safe_for_passive_observation
    older = inspect_renderer_installation(
        settings, owner_config=cfg, native_windows=True,
        now=NOW + timedelta(minutes=4),
        probe=probe, verify=verify,
        observer_process_probe=lambda pid: _writer(pid, cfg),
    )
    assert older.observer_heartbeat == "stale"
    assert not older.safe_for_passive_observation


@pytest.mark.parametrize("state", ["blocked", "unsupported"])
def test_fresh_failed_sample_blocks_readiness_and_next_verified_recovers(
    tmp_path: Path, state: SampleState,
) -> None:
    from artifex.render_node.installation_audit import inspect_renderer_installation
    from artifex.windows_tasks import StartupTaskStatus

    settings = _settings(tmp_path)
    settings.render_agent.comfyui_process.enabled = True
    cfg = tmp_path / "renderer.yaml"
    cfg.write_text("{}")

    def probe(role: str) -> StartupTaskStatus:
        return StartupTaskStatus.model_validate({
            "role": role, "task_name": "Artifex-" + role,
            "installed": True, "managed": True, "state": "Running",
        })

    def verify(*args: object, **kwargs: object) -> tuple[bool, str]:
        return True, "protected policy verified"

    _emit(settings, count=1, state="verified")
    assert inspect_observer_heartbeat(settings, now=NOW) == "fresh"
    _emit(settings, count=2, state=state)
    assert inspect_observer_heartbeat(settings, now=NOW) == state
    report = inspect_renderer_installation(
        settings, owner_config=cfg, native_windows=True, now=NOW,
        probe=probe, verify=verify,
        observer_process_probe=lambda pid: _writer(pid, cfg),
    )
    assert report.renderer_task.status == "running"
    assert report.survival_observer_task.status == "running"
    assert report.observer_heartbeat == state
    assert report.safe_for_passive_observation is False
    assert report.renderer_process_mutated is False
    assert report.production_qualified is False

    _emit(settings, count=3, state="verified")
    recovered = inspect_renderer_installation(
        settings, owner_config=cfg, native_windows=True, now=NOW,
        probe=probe, verify=verify,
        observer_process_probe=lambda pid: _writer(pid, cfg),
    )
    assert recovered.observer_heartbeat == "fresh"
    assert recovered.safe_for_passive_observation is True
    assert recovered.production_qualified is False


@pytest.mark.parametrize(("scenario", "expected"), [
    ("alive", "fresh"), ("exited", "process_missing"),
    ("probe_error", "process_unavailable"),
    ("reused", "process_mismatch"), ("foreign", "process_mismatch"),
    ("other_exe", "process_mismatch"), ("other_pid", "process_mismatch"),
])
def test_observer_writer_pid_and_exact_command_are_checked(
    tmp_path: Path, scenario: str, expected: str,
) -> None:
    settings = _settings(tmp_path)
    config = tmp_path / "render-node.yaml"
    config.write_text("{}")
    _emit(settings)
    def probe(pid: int) -> WindowsProcessIdentity | None:
        if scenario == "exited":
            return None
        if scenario == "probe_error":
            raise OSError("private process details")
        if scenario == "reused":
            return _writer(pid, config, started="2026-10-10T19:00:00Z")
        if scenario == "foreign":
            return _writer(pid, config, extra=("-m", "foreign.module"))
        if scenario == "other_exe":
            return _writer(pid, config, exe=r"C:\\Other\\malware.exe")
        if scenario == "other_pid":
            return _writer(pid + 1, config)
        return _writer(pid, config)
    assert inspect_observer_heartbeat(
        settings, now=NOW, config=config, process_probe=probe,
    ) == expected


def test_blocked_or_stale_heartbeat_never_invokes_process_probe(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    config = tmp_path / "render-node.yaml"
    config.write_text("{}")
    called: list[int] = []
    def probe(pid: int) -> WindowsProcessIdentity | None:
        called.append(pid)
        return _writer(pid, config)
    _emit(settings, state="blocked")
    assert inspect_observer_heartbeat(
        settings, now=NOW, config=config, process_probe=probe,
    ) == "blocked"
    _emit(settings, state="verified")
    assert inspect_observer_heartbeat(
        settings, now=NOW + timedelta(minutes=4),
        config=config, process_probe=probe,
    ) == "stale"
    assert not called
