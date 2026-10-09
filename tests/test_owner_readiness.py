"""Fail-closed PC-B owner evidence reconciliation tests."""
from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.render_node.native_launcher_probe import LauncherEvidence
from artifex.render_node.owner_readiness import collect_owner_readiness

_REQUIRED = (
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
)


def _owner(state: str = "observed_stable") -> dict[str, Any]:
    return {
        "status": state,
        "checks": {key: {"status": "pass", "reason": "ok"} for key in _REQUIRED},
        "process_observation_verified": True,
        "actual_listener_pid": 4200,
        "restart_authorized": False,
        "production_qualified": False,
    }


def _fixture(state: str = "observed", host: str | None = None) -> LauncherEvidence:
    return LauncherEvidence(
        observed_at_utc=datetime.now(UTC), host=host or socket.gethostname(),
        status=state, reason="fixture_checked",
        original_launcher_verified=True, listener_identity_stable=True,
        tcp_owner_stable=True, verified_runtime_provenance=True,
    )


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_agent.comfyui_process.executable = tmp_path / "python.exe"
    return settings


def test_both_independent_observations_never_authorize_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.render_node.owner_readiness as module

    settings = _settings(tmp_path)
    order: list[str] = []

    def probe(*, python: Path) -> LauncherEvidence:
        assert python == settings.render_agent.comfyui_process.executable
        order.append("fixture")
        return _fixture()

    def owner(_settings: ArtifexSettings, *, config: Path) -> dict[str, Any]:
        assert config == tmp_path / "render-node.yaml"
        order.append("owner")
        return _owner()

    monkeypatch.setattr(module, "inspect_native_python_listener", probe)
    monkeypatch.setattr(module, "observe_renderer_owner", owner)
    report = collect_owner_readiness(settings, config=tmp_path / "render-node.yaml")
    assert order == ["fixture", "owner"]
    assert report["status"] == "observed_independently"
    assert all(item["status"] == "pass" for item in report["checks"].values())
    assert report["remaining_real_machine_evidence"] == list(module._ALWAYS_MISSING)
    assert report["observations_are_independent"]
    assert report["issue_93_closure_authorized"] is False
    assert report["issue_40_closure_authorized"] is False
    assert report["renderer_restart_authorized"] is False
    assert report["gpu_jobs_submitted"] is False
    assert report["production_qualified"] is False
    assert report["mutated_services"] is False


@pytest.mark.parametrize("problem", (
    "fixture_blocked", "fixture_other_host", "owner_blocked",
    "missing_check", "owner_pid_missing", "owner_false_positive",
    "python_invalid", "python_missing", "unsupported",
))
def test_incomplete_evidence_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, problem: str,
) -> None:
    import artifex.render_node.owner_readiness as module

    settings = _settings(tmp_path)
    if problem == "python_missing":
        settings.render_agent.comfyui_process.executable = None
    owner = _owner(
        "unsupported" if problem == "unsupported"
        else "blocked" if problem == "owner_blocked"
        else "observed_stable"
    )
    if problem == "missing_check":
        del owner["checks"]["snapshot_consistency"]
    if problem == "owner_pid_missing":
        owner["actual_listener_pid"] = None
    if problem == "owner_false_positive":
        owner["production_qualified"] = True
    monkeypatch.setattr(module, "observe_renderer_owner", lambda *a, **kw: owner)

    def probe(*, python: Path) -> LauncherEvidence:
        if problem == "python_invalid":
            raise ValueError("test rejection")
        if problem == "fixture_blocked":
            return _fixture(state="blocked")
        if problem == "fixture_other_host":
            return _fixture(host="different-pc-b")
        return _fixture()

    monkeypatch.setattr(module, "inspect_native_python_listener", probe)
    report = collect_owner_readiness(settings, config=tmp_path / "render-node.yaml")
    expected = "unsupported" if problem == "unsupported" else "blocked"
    assert report["status"] == expected
    assert report["remaining_real_machine_evidence"]
    assert report["production_qualified"] is False
    assert report["renderer_restart_authorized"] is False
    assert report["issue_93_closure_authorized"] is False
    assert report["gpu_jobs_submitted"] is False
    if problem == "python_missing":
        assert report["launcher_fixture"] is None
    if problem == "python_invalid":
        assert "ValueError" in report["checks"]["disposable_launcher_fixture"]["reason"]


def test_cli_requires_config_and_never_overwrites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.cli as module

    path, dest = tmp_path / "render-node.yaml", tmp_path / "owner-readiness.json"
    runner = CliRunner()
    missing = runner.invoke(app, [
        "render-node", "owner-readiness", "--config", str(path), "--json",
    ])
    assert missing.exit_code == 1
    assert not dest.exists()

    path.write_text("render_agent: {}\n", encoding="utf-8")
    monkeypatch.setattr(module, "_settings", lambda path: _settings(tmp_path))
    monkeypatch.setattr(
        "artifex.render_node.owner_readiness.collect_owner_readiness",
        lambda *a, **kw: {
            "status": "observed_independently",
            "production_qualified": False,
            "renderer_restart_authorized": False,
        },
    )
    args = [
        "render-node", "owner-readiness", "--config", str(path),
        "--output", str(dest), "--json",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(dest.read_text()) == json.loads(result.stdout)
    repeat = runner.invoke(app, args)
    assert repeat.exit_code == 1
    assert json.loads(dest.read_text())["production_qualified"] is False


def test_cli_blocked_status_is_nonzero_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.cli as module

    path = tmp_path / "render-node.yaml"
    path.write_text("render_agent: {}\n", encoding="utf-8")
    monkeypatch.setattr(module, "_settings", lambda path: _settings(tmp_path))
    monkeypatch.setattr(
        "artifex.render_node.owner_readiness.collect_owner_readiness",
        lambda *a, **kw: {"status": "blocked", "production_qualified": False},
    )
    result = CliRunner().invoke(
        app, ["render-node", "owner-readiness", "--config", str(path), "--json"],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "blocked"
