"""Independent Windows watcher heartbeat: bounded, atomic, read-only review."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.config.models import ArtifexSettings
from artifex.render_node.observer_heartbeat import (
    _MAX_SIZE,
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
          poll: float = 15, count: int = 1) -> None:
    publish_observer_heartbeat(
        settings, observed_utc=when,
        poll_seconds=poll, sample_count=count, state="verified",
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
                                    "invalid_json", "false_qualification"])
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
    )
    assert initial.renderer_task.status == "running"
    assert initial.survival_observer_task.status == "running"
    assert initial.observer_heartbeat == "missing"
    assert not initial.safe_for_passive_observation
    _emit(settings)
    active = inspect_renderer_installation(
        settings, owner_config=cfg, native_windows=True, now=NOW,
        probe=probe, verify=verify,
    )
    assert active.observer_heartbeat == "fresh"
    assert active.safe_for_passive_observation
    older = inspect_renderer_installation(
        settings, owner_config=cfg, native_windows=True,
        now=NOW + timedelta(minutes=4),
        probe=probe, verify=verify,
    )
    assert older.observer_heartbeat == "stale"
    assert not older.safe_for_passive_observation
