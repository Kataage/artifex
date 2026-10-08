from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.windows_tasks import (
    StartupTaskStatus,
    _install_script,
    _run_powershell,
    task_configuration_matches,
    task_status,
)


def _status(**values: Any) -> StartupTaskStatus:
    expected = {
        "role": "renderer",
        "task_name": "Artifex-Renderer",
        "installed": True,
        "managed": True,
        "state": "Ready",
        "execute": r"C:\Artifex\.venv\Scripts\python.exe",
        "arguments": '-m artifex.cli render-node serve --config "C:\\Artifex\\config\\render-node.yaml"',
        "working_directory": r"C:\Artifex",
        "action_count": 1,
        "allow_hard_terminate": False,
        "multiple_instances": "IgnoreNew",
        "execution_time_limit_seconds": 0,
    }
    expected.update(values)
    return StartupTaskStatus(**expected)


def test_renderer_registration_disallows_hard_termination_and_running_replacement() -> None:
    script = _install_script(
        "renderer", execute="C:/Artifex/python.exe", arguments="-m artifex.cli render-node serve",
        working_directory="C:/Artifex", replace=True, restart_count=10,
    )
    assert "DisallowHardTerminate = $true" in script
    assert "MultipleInstances = 'IgnoreNew'" in script
    assert "ExecutionTimeLimit = (New-TimeSpan -Seconds 0)" in script
    assert "if ([string]$old.State -eq 'Running')" in script
    assert "Refusing to replace a running Artifex scheduled task" in script
    assert script.index("Refusing to replace a running Artifex scheduled task") < script.index("Register-ScheduledTask")
    controller = _install_script(
        "controller", execute="C:/Artifex/python.exe", arguments="-m artifex.cli daemon",
        working_directory="C:/Artifex", replace=False, restart_count=10,
    )
    assert "if ($false) {" in controller


@pytest.mark.parametrize(
    ("changes", "valid"),
    [
        ({}, True),
        ({"action_count": None}, False),
        ({"action_count": 2}, False),
        ({"allow_hard_terminate": None}, False),
        ({"allow_hard_terminate": True}, False),
        ({"multiple_instances": "Parallel"}, False),
        ({"multiple_instances": None}, False),
        ({"execution_time_limit_seconds": None}, False),
        ({"execution_time_limit_seconds": 259200}, False),
        ({"managed": False}, False),
    ],
)
def test_renderer_policy_is_fail_closed_for_unknown_or_unsafe_settings(
    changes: dict[str, Any],
    valid: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    monkeypatch.setattr(tasks, "_require_windows", lambda: None)
    # Isolate expected config command from real host while exercising the
    # complete matcher policy and receipt-independent refusal behavior.
    trusted = _status()
    monkeypatch.setattr(
        tasks, "_command",
        lambda role, config: (
            trusted.execute, trusted.arguments, trusted.working_directory
        ),
    )
    matching, reason = task_configuration_matches(
        "renderer", config=tmp_path / "render-node.yaml", status=_status(**changes),
    )
    assert matching is valid
    if not valid:
        assert "unsafe" in reason or "unowned" in reason


def test_status_reads_real_scheduler_safety_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    monkeypatch.setattr(tasks.platform, "system", lambda: "Windows")
    emitted: list[str] = []

    def shell(script: str) -> str:
        emitted.append(script)
        return json.dumps({
            "installed": True,
            "managed": True,
            "state": "Running",
            "execute": r"C:\Python\python.exe",
            "arguments": "-m artifex.cli render-node serve",
            "working_directory": r"C:\Artifex",
            "action_count": 1,
            "allow_hard_terminate": False,
            "multiple_instances": "IgnoreNew",
            "execution_time_limit_seconds": 0,
        })

    monkeypatch.setattr(tasks, "_run_powershell", shell)
    report = task_status("renderer")
    assert report.action_count == 1
    assert report.allow_hard_terminate is False
    assert report.multiple_instances == "IgnoreNew"
    assert report.execution_time_limit_seconds == 0
    assert "Get-ScheduledTask" in emitted[0]
    assert "AllowHardTerminate" in emitted[0]
    assert "ExecutionTimeLimit.TotalSeconds" in emitted[0]


def test_read_only_task_safety_cli_exits_nonzero_on_unknown_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_tasks as tasks

    cfg = tmp_path / "render-node.yaml"
    cfg.write_text("render_agent: {}\n", encoding="utf-8")
    monkeypatch.setattr("artifex.cli.task_status", lambda role: _status(allow_hard_terminate=True))
    monkeypatch.setattr(tasks, "_require_windows", lambda: None)
    monkeypatch.setattr(
        tasks, "_command",
        lambda role, config: (_status().execute, _status().arguments, _status().working_directory),
    )
    runner = CliRunner()
    rejected = runner.invoke(
        app, ["startup", "audit", "--role", "renderer", "--config", str(cfg)],
    )
    assert rejected.exit_code == 1, rejected.output
    assert '"verified": false' in rejected.output
    assert '"restart_authorized": false' in rejected.output

    monkeypatch.setattr("artifex.cli.task_status", lambda role: _status())
    accepted = runner.invoke(
        app, ["startup", "audit", "--role", "renderer", "--config", str(cfg)],
    )
    assert accepted.exit_code == 0, accepted.output
    assert '"verified": true' in accepted.output
    assert '"child_survival_qualified": false' in accepted.output


@pytest.mark.skipif(platform.system() != "Windows", reason="native Windows ScheduledTasks cmdlets")
def test_native_windows_scheduler_supports_disallow_hard_terminate() -> None:
    # Exercise native Windows cmdlet on CI without registering/stopping a
    # Task Scheduler task or modifying any running application.
    script = """
$ErrorActionPreference='Stop'
$s=New-ScheduledTaskSettingsSet -DisallowHardTerminate -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)
[pscustomobject]@{
  allow_hard_terminate=[bool]$s.AllowHardTerminate
  multiple_instances=[string]$s.MultipleInstances
  execution_time_limit_seconds=[int]$s.ExecutionTimeLimit.TotalSeconds
} | ConvertTo-Json -Compress
"""
    parsed = json.loads(_run_powershell(script))
    assert parsed == {
        "allow_hard_terminate": False,
        "multiple_instances": "IgnoreNew",
        "execution_time_limit_seconds": 0,
    }
