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
    StartupRole,
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
    role: StartupRole = "controller",
) -> StartupTaskStatus:
    return StartupTaskStatus(
        role=role, task_name="Artifex-" + role,
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
    assert report.mode == "preview" and not report.ready
    assert report.deployment.ready and report.task_compatible
    assert report.task_running_observed is False
    assert report.task.state == "Ready"
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

    statuses = iter([
        _task(installed=False),  # initial installation check
        _task(state="Ready"),   # service not yet ready
        _task(state="Running"), # subsequent healthy cycle
    ])
    report = await activate_deployment(
        ArtifexSettings(), role="renderer", config=config,
        apply=True, install_missing=True, wait_seconds=60,
        status_fn=lambda _: next(statuses),
        match_fn=lambda *a, **kw: (
            kw["status"].installed, "match" if kw["status"].installed else "missing"
        ),
        start_fn=activate,
        verify_fn=verify,
        sleep_fn=sleep,
    )
    assert report.ready and report.checks_performed == 2
    assert report.task_running_observed is True
    assert report.task.state == "Running"
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


@pytest.mark.asyncio
async def test_activation_preview_never_claims_ready_from_stale_healthy_endpoint(
    tmp_path: Path,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.touch()
    seen: list[str] = []

    async def healthy(_: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        seen.append("live_health")
        return _deployment(True)

    def stopped_task(_: str) -> StartupTaskStatus:
        seen.append("task_recheck")
        return _task(state="Ready")

    report = await activate_deployment(
        ArtifexSettings(), role="renderer", config=config,
        status_fn=stopped_task,
        match_fn=lambda *a, **kw: (True, "matching"),
        verify_fn=healthy,
    )
    assert report.deployment.ready is True
    assert report.task_compatible is True
    assert report.task_running_observed is False
    assert report.ready is False
    assert report.action == "preview"
    assert seen == ["task_recheck", "live_health", "task_recheck"]
    assert report.production_qualified is False


@pytest.mark.asyncio
async def test_activation_rechecks_task_after_health_even_when_start_reports_running(
    tmp_path: Path,
) -> None:
    config = tmp_path / "local.yaml"
    config.touch()
    latest: list[StartupTaskStatus] = [
        _task(state="Ready"),  # pre-start
        _task(state="Ready"),  # native task exited while health succeeded
    ]
    calls: list[str] = []

    async def healthy(_: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        calls.append("healthy")
        return _deployment(True)

    report = await activate_deployment(
        ArtifexSettings(), role="controller", config=config,
        apply=True, wait_seconds=0,
        status_fn=lambda _: latest.pop(0),
        start_fn=lambda *a, **kw: (
            _task(state="Running"), "started",
        ),
        match_fn=lambda *a, **kw: (True, "matching"),
        verify_fn=healthy,
    )
    assert calls == ["healthy"]
    assert report.task.state == "Ready"
    assert report.task_running_observed is False
    assert report.deployment.ready is True
    assert not report.ready
    assert report.checks_performed == 1


@pytest.mark.asyncio
async def test_activation_waits_for_task_running_even_if_endpoint_is_healthy(
    tmp_path: Path,
) -> None:
    config = tmp_path / "local.yaml"
    config.touch()
    polls = iter([
        _task(state="Ready"), _task(state="Ready"),
        _task(state="Running"),
    ])
    sleeps: list[float] = []
    checks = 0

    async def healthy(_: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        nonlocal checks
        checks += 1
        return _deployment(True)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    report = await activate_deployment(
        ArtifexSettings(), role="controller", config=config,
        apply=True, wait_seconds=30,
        status_fn=lambda _: next(polls),
        start_fn=lambda *a, **kw: (_task(state="Running"), "started"),
        match_fn=lambda *a, **kw: (True, "matches"),
        verify_fn=healthy, sleep_fn=sleep,
    )
    assert checks == report.checks_performed == 2
    assert report.task_running_observed is True
    assert report.ready is True
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_activation_stops_waiting_and_fails_closed_on_task_registration_drift(
    tmp_path: Path,
) -> None:
    config = tmp_path / "local.yaml"
    config.touch()
    polls = iter([
        _task(state="Ready"),
        _task(state="Running", arguments="drifted"),
    ])
    sleeps: list[float] = []

    async def healthy(_: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        return _deployment(True)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    report = await activate_deployment(
        ArtifexSettings(), role="controller", config=config,
        apply=True, wait_seconds=30,
        status_fn=lambda _: next(polls),
        start_fn=lambda *a, **kw: (_task(state="Running"), "started"),
        match_fn=lambda *a, **kw: (
            kw["status"].arguments == "matching",
            "configuration changed" if kw["status"].arguments != "matching" else "ok",
        ),
        verify_fn=healthy, sleep_fn=sleep,
    )
    assert report.task_running_observed is True
    assert report.task_compatible is False
    assert report.task_detail == "configuration changed"
    assert report.ready is False
    assert report.checks_performed == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_combined_renderer_observer_preview_is_read_only_and_cannot_qualify(
    tmp_path: Path,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.touch()
    settings = ArtifexSettings()
    settings.render_agent.comfyui_process.enabled = True
    started: list[str] = []
    inspected: list[str] = []

    def status(role: StartupRole) -> StartupTaskStatus:
        inspected.append(role)
        return _task(
            role=role,
            installed=role != "survival-observer",
            state="Ready",
        )

    async def healthy(_: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        return _deployment(True)

    report = await activate_deployment(
        settings, role="renderer", config=config, with_observer=True,
        status_fn=status,
        match_fn=lambda *a, **kw: (
            kw["status"].installed, "matches" if kw["status"].installed else "missing",
        ),
        start_fn=lambda role, **kw: (
            started.append(role) or _task(role=role, state="Running"), "started"
        ),
        verify_fn=healthy,
        heartbeat_fn=lambda *a, **kw: "missing",
    )
    assert inspected == [
        "renderer", "survival-observer", "renderer", "survival-observer",
    ]
    assert started == []
    assert report.mode == "preview"
    assert report.observer_requested is True
    assert report.observer_action == "preview"
    assert report.observer_task is not None
    assert not report.observer_task.installed
    assert report.observer_heartbeat == "missing"
    assert not report.observer_ready
    assert not report.ready
    assert not report.production_qualified


@pytest.mark.asyncio
async def test_combined_activation_starts_watcher_before_renderer_and_waits_for_heartbeat(
    tmp_path: Path,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.touch()
    settings = ArtifexSettings()
    settings.render_agent.comfyui_process.enabled = True
    activity: list[str] = []
    heartbeats = iter(["blocked", "fresh"])
    sleeps: list[float] = []
    active: set[str] = set()

    def status(role: StartupRole) -> StartupTaskStatus:
        activity.append("status:" + role)
        return _task(
            role=role, state="Running" if role in active else "Ready",
        )

    def start(role: StartupRole, **kwargs: Any) -> tuple[StartupTaskStatus, str]:
        activity.append("start:" + role)
        active.add(role)
        return _task(role=role, state="Running"), "started"

    def heartbeat(*args: Any, **kwargs: Any) -> str:
        activity.append("heartbeat")
        return next(heartbeats)

    async def healthy(_: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        activity.append("health")
        return _deployment(True)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = await activate_deployment(
        settings, role="renderer", config=config,
        apply=True, with_observer=True, wait_seconds=30,
        status_fn=status,
        match_fn=lambda *a, **kw: (True, "verified"),
        start_fn=start, verify_fn=healthy,
        heartbeat_fn=heartbeat, sleep_fn=sleep,
    )
    assert activity.index("start:survival-observer") < activity.index(
        "start:renderer"
    )
    assert activity.index("start:renderer") < activity.index("health")
    assert result.ready
    assert result.observer_requested and result.observer_ready
    assert result.observer_heartbeat == "fresh"
    assert result.observer_action == "started"
    assert result.observer_task is not None
    assert result.observer_task.state == "Running"
    assert result.checks_performed == 2
    assert len(sleeps) == 1
    assert not result.native_child_survival_proven
    assert not result.production_qualified


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "foreign_watcher", "stale_watcher", "missing_watcher",
    "bad_renderer", "missing_renderer",
])
async def test_combined_activation_refuses_unsafe_preflight_before_any_start(
    tmp_path: Path, case: str,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.touch()
    settings = ArtifexSettings()
    settings.render_agent.comfyui_process.enabled = True
    touched: list[str] = []

    def status(role: StartupRole) -> StartupTaskStatus:
        return _task(
            role=role,
            installed=not (
                role == "survival-observer" and case == "missing_watcher"
                or role == "renderer" and case == "missing_renderer"
            ),
            managed=not (
                role == "survival-observer" and case == "foreign_watcher"
            ),
            arguments="drifted" if (
                role == "survival-observer" and case == "stale_watcher"
                or role == "renderer" and case == "bad_renderer"
            ) else "matching",
        )

    async def healthy(_: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        touched.append("health")
        return _deployment(True)

    with pytest.raises(ValueError):
        await activate_deployment(
            settings, role="renderer", config=config,
            with_observer=True, apply=True,
            status_fn=status,
            match_fn=lambda *a, **kw: (
                kw["status"].installed and kw["status"].managed
                and kw["status"].arguments == "matching", "checked",
            ),
            start_fn=lambda role, **kw: (
                touched.append("started") or _task(role=role), "started"
            ),
            verify_fn=healthy,
        )
    assert touched == []


@pytest.mark.asyncio
async def test_combined_activation_does_not_fabricate_ready_without_fresh_heartbeat(
    tmp_path: Path,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.touch()
    settings = ArtifexSettings()
    settings.render_agent.comfyui_process.enabled = True

    async def healthy(_: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        return _deployment(True)

    report = await activate_deployment(
        settings, role="renderer", config=config,
        apply=True, with_observer=True, wait_seconds=0,
        status_fn=lambda role: _task(role=role, state="Running"),
        match_fn=lambda *a, **kw: (True, "verified"),
        start_fn=lambda role, **kw: (_task(role=role, state="Running"), "started"),
        verify_fn=healthy,
        heartbeat_fn=lambda *a, **kw: "stale",
    )
    assert report.task_running_observed
    assert report.deployment.ready
    assert report.observer_task is not None
    assert report.observer_task.state == "Running"
    assert report.observer_heartbeat == "stale"
    assert report.observer_ready is False
    assert report.ready is False
    assert not report.native_child_survival_proven


@pytest.mark.asyncio
async def test_combined_activation_requires_renderer_managed_and_correct_host(
    tmp_path: Path,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.touch()
    with pytest.raises(ValueError, match="PC-B renderer"):
        await activate_deployment(
            ArtifexSettings(), role="controller", config=config,
            with_observer=True,
        )
    with pytest.raises(ValueError, match="managed ComfyUI"):
        await activate_deployment(
            ArtifexSettings(), role="renderer", config=config,
            with_observer=True,
        )


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
            checks_performed=1, task_running_observed=True, ready=True,
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


def test_deployment_activate_cli_with_observer_is_explicit_and_pc_b_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.write_text("render_agent: {}\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    async def fake_activate(_: ArtifexSettings, **kwargs: Any) -> ActivationReport:
        calls.append(kwargs)
        return ActivationReport(
            role="renderer",
            mode="apply" if kwargs["apply"] else "preview",
            action="started" if kwargs["apply"] else "preview",
            task_compatible=True, task_detail="matches",
            task=_task(role="renderer", state="Running"),
            deployment=_deployment(True), checks_performed=1,
            task_running_observed=True, ready=True,
            observer_requested=kwargs["with_observer"],
            observer_ready=True,
        )

    monkeypatch.setattr("artifex.cli.activate_deployment", fake_activate)
    runner = CliRunner()
    preview = runner.invoke(
        app, [
            "deployment", "activate", "--role", "renderer", "--with-observer",
            "--config", str(config), "--json",
        ],
    )
    assert preview.exit_code == 0, preview.output
    assert calls[-1]["with_observer"] is True
    assert calls[-1]["apply"] is False
    applied = runner.invoke(
        app, [
            "deployment", "activate", "--role", "renderer", "--with-observer",
            "--apply", "--config", str(config), "--json",
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert calls[-1]["with_observer"] is True
    assert calls[-1]["apply"] is True
    assert '"production_qualified": false' in applied.output
    before = len(calls)
    invalid = runner.invoke(
        app, [
            "deployment", "activate", "--role", "controller",
            "--with-observer", "--config", str(config),
        ],
    )
    assert invalid.exit_code != 0
    assert "--role renderer" in invalid.output
    assert len(calls) == before


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
