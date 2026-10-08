from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.windows_tasks import (
    StartupTaskStatus,
    _command,
    _install_script,
    _literal,
    install_task,
    task_status,
    uninstall_task,
)


def _registered(
    role: str = "controller",
    *,
    installed: bool = True,
    managed: bool = True,
) -> StartupTaskStatus:
    return StartupTaskStatus(
        role=role,
        task_name=f"Artifex-{role.capitalize()}",
        installed=installed,
        managed=managed,
        state="Ready" if installed else None,
        execute="C:/path with spaces/.venv/Scripts/python.exe" if installed else None,
    )


def test_windows_task_install_generates_safe_launcher_and_owned_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    config = tmp_path / "my user's config" / "local.yaml"
    config.parent.mkdir()
    config.write_text("agent: {}\n", encoding="utf-8")
    executable = tmp_path / "venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"exe")
    monkeypatch.setattr(tasks.sys, "executable", str(executable))
    monkeypatch.setattr(tasks.platform, "system", lambda: "Windows")
    generated: list[str] = []

    def fake_ps(script: str) -> str:
        generated.append(script)
        return "{}"

    monkeypatch.setattr(tasks, "_run_powershell", fake_ps)
    monkeypatch.setattr(tasks, "task_status", lambda role: _registered(role))
    installed = install_task("controller", config=config, restart_count=3)

    assert installed.installed and installed.managed
    script = generated[0]
    assert "New-ScheduledTaskAction" in script
    assert "New-ScheduledTaskTrigger -AtLogOn" in script
    assert "New-ScheduledTaskPrincipal" in script
    assert "-LogonType Interactive -RunLevel Limited" in script
    assert "-MultipleInstances IgnoreNew" not in script
    assert "MultipleInstances = 'IgnoreNew'" in script
    assert "RestartCount = 3" in script
    assert "RestartInterval" in script
    assert "shell" not in script
    assert "-m artifex.cli daemon --config" in script
    assert "my user''s config" in script  # PowerShell single-quote escaping
    assert "Refusing to overwrite a task not owned by Artifex" in script


def test_renderer_launch_arguments_and_no_restart_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    config = tmp_path / "render-node.yaml"
    config.write_text("render_agent: {}\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.touch()
    monkeypatch.setattr(tasks.sys, "executable", str(executable))
    execute, args, workdir = _command("renderer", config)
    assert execute == str(executable.resolve())
    assert "render-node serve --config" in args
    assert workdir == str(Path.cwd().resolve())
    script = _install_script(
        "renderer",
        execute=execute,
        arguments=args,
        working_directory=workdir,
        replace=False,
        restart_count=0,
    )
    assert "if (0 -gt 0)" in script
    assert "Artifex-Renderer" in script
    assert "already exists; use --replace" in script


def test_uninstall_only_artifex_owned_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    monkeypatch.setattr(tasks.platform, "system", lambda: "Windows")
    emitted: list[str] = []
    monkeypatch.setattr(tasks, "_run_powershell", lambda script: emitted.append(script) or "")
    monkeypatch.setattr(tasks, "task_status", lambda role: _registered(role))
    assert uninstall_task("renderer")
    assert "Unregister-ScheduledTask" in emitted[0]
    assert "Refusing to remove a task not owned by Artifex" in emitted[0]


def test_uninstall_missing_task_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    monkeypatch.setattr(tasks.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tasks, "task_status", lambda role: _registered(role, installed=False))
    monkeypatch.setattr(
        tasks, "_run_powershell",
        lambda _: pytest.fail("must not contact scheduler for missing uninstall"),
    )
    assert not uninstall_task("controller")


def test_status_parses_managed_and_unmanaged_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    monkeypatch.setattr(tasks.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        tasks, "_run_powershell",
        lambda script: json.dumps(
            {
                "installed": True,
                "managed": False,
                "state": "Running",
                "execute": "C:/python.exe",
                "arguments": "external process",
                "working_directory": "C:/",
            }
        ),
    )
    result = task_status("controller")
    assert result.installed and not result.managed
    assert result.state == "Running"


def test_powershell_encoded_command_never_uses_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    monkeypatch.setattr(tasks.platform, "system", lambda: "Windows")
    called: dict[str, Any] = {}

    class Completed:
        returncode = 0
        stdout = '{"ok": true}'
        stderr = ""

    def mock_run(args: list[str], **kwargs: Any) -> Completed:
        called.update(args=args, kwargs=kwargs)
        return Completed()

    monkeypatch.setattr(tasks.subprocess, "run", mock_run)
    assert tasks._run_powershell("$ErrorActionPreference = 'Stop'") == '{"ok": true}'
    args = called["args"]
    assert args[:3] == ["powershell.exe", "-NoProfile", "-NonInteractive"]
    assert args[3] == "-EncodedCommand"
    decoded = base64.b64decode(args[4]).decode("utf-16le")
    assert "$ErrorActionPreference = 'Stop'" in decoded
    assert called["kwargs"]["check"] is False
    assert called["kwargs"]["timeout"] == 45


def test_unsafe_powershell_literals_rejected() -> None:
    assert _literal("O'Brien") == "'O''Brien'"
    with pytest.raises(ValueError, match="control characters"):
        _literal("safe\nInvoke-Expression malicious")
    with pytest.raises(ValueError, match="control characters"):
        _literal("embedded\x00nul")


def test_windows_startup_rejects_non_windows_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    monkeypatch.setattr(tasks.platform, "system", lambda: "Linux")
    with pytest.raises(OSError, match="native Windows"):
        install_task("controller", config=tmp_path / "local.yaml")


def test_startup_cli_status_reports_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.cli.task_status", lambda role: _registered(role))
    result = CliRunner().invoke(app, ["startup", "status", "--role", "renderer", "--json"])
    assert result.exit_code == 0, result.output
    assert '"task_name": "Artifex-Renderer"' in result.output
    assert '"managed": true' in result.output


def test_startup_cli_install_does_not_register_on_invalid_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "artifex.cli.install_task",
        lambda *a, **kw: pytest.fail("must not register an invalid config"),
    )
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("unknown_field: true\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["startup", "install", "--config", str(invalid)]
    )
    assert result.exit_code != 0
    assert "install error" in result.output


def test_startup_cli_uninstall_respects_missing_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.cli.uninstall_task", lambda role: False)
    result = CliRunner().invoke(app, ["startup", "uninstall", "--role", "controller"])
    assert result.exit_code == 0, result.output
    assert "not installed" in result.output
