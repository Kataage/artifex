from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.deployment import DeploymentCheck, DeploymentReport
from artifex.deployment_activation import ActivationReport, activate_deployment
from artifex.windows_tasks import (
    StartupTaskStatus,
    _start_script,
    activate_task,
)


def _task(
    *,
    managed: bool = True,
    installed: bool = True,
    state: str = "Ready",
    arguments: str = "matching",
) -> StartupTaskStatus:
    return StartupTaskStatus(
        role="controller", task_name="Artifex-Controller",
        installed=installed, managed=managed, state=state if installed else None,
        execute="python.exe" if installed else None,
        arguments=arguments if installed else None,
        working_directory="C:/Artifex" if installed else None,
    )


def _deployment(ready: bool) -> DeploymentReport:
    return DeploymentReport(
        role="controller", ready=ready,
        checks=(
            DeploymentCheck(
                name="controller:real_health", ready=ready,
                blocking=True, detail="real service reachable" if ready else "offline",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_activation_preview_never_starts_and_never_claims_qualification(
    tmp_path: Path,
) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("agent: {}\n", encoding="utf-8")
    started: list[object] = []
    verified: list[dict[str, object]] = []

    async def verify(_: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        verified.append(kwargs)
        return _deployment(True)

    report = await activate_deployment(
        ArtifexSettings(), role="controller", config=config,
        status_fn=lambda _: _task(),
        match_fn=lambda *a, **kw: (True, "matches"),
        start_fn=lambda *a, **kw: started.append(kw) or (_task(), "started"),
        verify_fn=verify,
    )
    assert report.mode == "preview" and report.ready
    assert report.action == "preview" and report.production_qualified is False
    assert report.checks_performed == 1 and not started
    assert verified == [{
        "role": "controller", "require_autostart": True, "render_smoke": False
    }]


@pytest.mark.asyncio
async def test_activation_apply_starts_only_local_task_and_waits_for_real_health(
    tmp_path: Path,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.write_text("render_agent: {}\n", encoding="utf-8")
    starts: list[dict[str, object]] = []
    readings: list[bool] = [False, True]
    sleeps: list[float] = []

    async def verify(_: ArtifexSettings, **kw: Any) -> DeploymentReport:
        assert kw["role"] == "renderer"
        assert kw["render_smoke"] is False
        ready = readings.pop(0)
        return DeploymentReport(
            role="renderer", ready=ready, checks=(
                DeploymentCheck(name="renderer:real", ready=ready, blocking=True, detail="ok"),
            ),
        )

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def activate(_: str, **kwargs: Any) -> tuple[StartupTaskStatus, str]:
        starts.append(kwargs)
        return _task(state="Running"), "installed_and_started"

    report = await activate_deployment(
        ArtifexSettings(), role="renderer", config=config,
        apply=True, install_missing=True, wait_seconds=60,
        status_fn=lambda _: _task(installed=False),
        match_fn=lambda *a, **kw: (
            kw["status"].installed, "match" if kw["status"].installed else "missing"
        ),
        start_fn=activate,
        verify_fn=verify,
        sleep_fn=sleep,
    )
    assert report.ready and report.checks_performed == 2
    assert report.action == "installed_and_started"
    assert starts == [{
        "config": config, "install_missing": True, "replace": False,
    }]
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 5


@pytest.mark.asyncio
async def test_activation_fails_closed_on_unhealthy_services_or_stale_task(
    tmp_path: Path,
) -> None:
    config = tmp_path / "local.yaml"
    config.touch()

    async def verify(_: ArtifexSettings, **kw: Any) -> DeploymentReport:
        return _deployment(True)

    bad = await activate_deployment(
        ArtifexSettings(), role="controller", config=config,
        status_fn=lambda _: _task(managed=False),
        match_fn=lambda *a, **kw: (False, "unmanaged"),
        verify_fn=verify,
    )
    assert not bad.ready and not bad.task_compatible
    with pytest.raises(ValueError, match="require --apply"):
        await activate_deployment(
            ArtifexSettings(), role="controller", config=config,
            install_missing=True, verify_fn=verify,
        )
    with pytest.raises(ValueError, match="require --apply"):
        await activate_deployment(
            ArtifexSettings(), role="controller", config=config,
            replace=True, verify_fn=verify,
        )

    async def offline(_: ArtifexSettings, **kw: Any) -> DeploymentReport:
        return _deployment(False)

    stopped = await activate_deployment(
        ArtifexSettings(), role="controller", config=config,
        apply=True, wait_seconds=0,
        status_fn=lambda _: _task(),
        match_fn=lambda *a, **kw: (True, "matches"),
        start_fn=lambda *a, **kw: (_task(state="Running"), "started"),
        verify_fn=offline,
    )
    assert stopped.mode == "apply" and not stopped.ready
    assert stopped.checks_performed == 1


def test_activation_task_refuses_unowned_missing_and_stale_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as win

    config = tmp_path / "local.yaml"
    config.touch()
    monkeypatch.setattr(win.platform, "system", lambda: "Windows")
    monkeypatch.setattr(win, "_command", lambda *args: ("python.exe", "matching", "C:/Artifex"))
    messages: list[str] = []
    monkeypatch.setattr(win, "_run_powershell", lambda script: messages.append(script) or "")
    monkeypatch.setattr(win, "task_status", lambda _: _task(managed=False))
    with pytest.raises(ValueError, match="unowned"):
        activate_task("controller", config=config, replace=True)
    assert not messages

    monkeypatch.setattr(win, "task_status", lambda _: _task(installed=False))
    with pytest.raises(ValueError, match="install-missing"):
        activate_task("controller", config=config)
    assert not messages

    monkeypatch.setattr(win, "task_status", lambda _: _task(arguments="wrong"))
    with pytest.raises(ValueError, match="--replace"):
        activate_task("controller", config=config)
    assert not messages

    monkeypatch.setattr(win, "task_status", lambda _: _task(arguments="wrong", state="Running"))
    with pytest.raises(ValueError, match="running"):
        activate_task("controller", config=config, replace=True)
    assert not messages


def test_activation_start_is_single_instance_and_checks_identity_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as win

    config = tmp_path / "local.yaml"
    config.touch()
    monkeypatch.setattr(win.platform, "system", lambda: "Windows")
    monkeypatch.setattr(win, "_command", lambda *args: ("python.exe", "matching", "C:/Artifex"))
    messages: list[str] = []
    monkeypatch.setattr(win, "_run_powershell", lambda script: messages.append(script) or "")
    monkeypatch.setattr(win, "task_status", lambda _: _task(state="Running"))
    status, action = activate_task("controller", config=config)
    assert status.state == "Running" and action == "already_running"
    assert not messages

    monkeypatch.setattr(win, "task_status", lambda _: _task())
    status, action = activate_task("controller", config=config)
    assert action == "started"
    assert len(messages) == 1
    assert "Start-ScheduledTask" in messages[0]
    assert "task changed after preflight" in messages[0]
    assert "Refusing to start a task not owned by Artifex" in messages[0]
    assert "-cne 'matching'" in messages[0]
    assert status.managed

    script = _start_script(
        "renderer", execute="python.exe",
        arguments="-m artifex.cli render-node serve --config C:/data/my's config.yaml",
        working_directory="C:/data",
    )
    assert "Start-ScheduledTask" in script
    assert "my''s config.yaml" in script


def test_activation_task_explicit_install_and_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as win

    config = tmp_path / "local.yaml"
    config.touch()
    monkeypatch.setattr(win.platform, "system", lambda: "Windows")
    monkeypatch.setattr(win, "_command", lambda *args: ("python.exe", "matching", "C:/Artifex"))
    current: list[StartupTaskStatus] = [_task(installed=False)]
    replacements: list[bool] = []
    starts: list[str] = []
    monkeypatch.setattr(win, "task_status", lambda _: current[0])
    monkeypatch.setattr(win, "_run_powershell", lambda script: starts.append(script) or "")

    def install(_: str, *, config: Path, replace: bool = False) -> StartupTaskStatus:
        replacements.append(replace)
        current[0] = _task()
        return current[0]

    monkeypatch.setattr(win, "install_task", install)
    status, action = activate_task("controller", config=config, install_missing=True)
    assert status.installed and action == "installed_and_started"
    assert replacements == [False] and len(starts) == 1

    current[0] = _task(arguments="outdated")
    status, action = activate_task("controller", config=config, replace=True)
    assert action == "replaced_and_started" and status.managed
    assert replacements == [False, True] and len(starts) == 2


def test_deployment_activate_cli_is_preview_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("agent: {}\n", encoding="utf-8")
    requested: list[dict[str, Any]] = []

    async def fake_activate(_: ArtifexSettings, **kwargs: Any) -> ActivationReport:
        requested.append(kwargs)
        return ActivationReport(
            role="controller", mode="apply" if kwargs["apply"] else "preview",
            action="started" if kwargs["apply"] else "preview",
            task_compatible=True, task_detail="matches",
            task=_task(), deployment=_deployment(True),
            checks_performed=1, ready=True,
        )

    monkeypatch.setattr("artifex.cli.activate_deployment", fake_activate)
    runner = CliRunner()
    preview = runner.invoke(
        app, ["deployment", "activate", "--config", str(config), "--json"]
    )
    assert preview.exit_code == 0, preview.output
    assert requested[-1]["apply"] is False
    assert '"production_qualified": false' in preview.output
    apply = runner.invoke(
        app, ["deployment", "activate", "--config", str(config), "--apply"]
    )
    assert apply.exit_code == 0, apply.output
    assert requested[-1]["apply"] is True


def test_deployment_activate_cli_refuses_missing_or_invalid_configs(tmp_path: Path) -> None:
    runner = CliRunner()
    missing = runner.invoke(
        app, ["deployment", "activate", "--config", str(tmp_path / "missing.yaml"), "--apply"]
    )
    assert missing.exit_code != 0 and "non-symlink config required" in missing.output
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("unknown_config_key: true\n", encoding="utf-8")
    result = runner.invoke(
        app, ["deployment", "activate", "--config", str(invalid), "--apply"]
    )
    assert result.exit_code != 0
