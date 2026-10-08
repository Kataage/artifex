from __future__ import annotations

import base64
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

StartupRole = Literal["controller", "renderer"]
_TASKS: dict[StartupRole, str] = {
    "controller": "Artifex-Controller",
    "renderer": "Artifex-Renderer",
}
_MARKER = "Artifex managed autostart v1"


class StartupTaskStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: StartupRole
    task_name: str
    installed: bool
    managed: bool
    state: str | None = None
    execute: str | None = None
    arguments: str | None = None
    working_directory: str | None = None


def _require_windows() -> None:
    if platform.system() != "Windows":
        raise OSError("Windows Task Scheduler setup is supported on native Windows only")


def _literal(value: str) -> str:
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Windows scheduled-task values cannot include control characters")
    return "'" + value.replace("'", "''") + "'"


def _command(role: StartupRole, config: Path) -> tuple[str, str, str]:
    interpreter = Path(sys.executable).resolve(strict=True)
    config_path = config.expanduser().resolve(strict=True)
    if not config_path.is_file():
        raise FileNotFoundError(f"Artifex startup config is not a file: {config_path}")
    if role == "controller":
        args = ["-m", "artifex.cli", "daemon", "--config", str(config_path)]
    else:
        args = [
            "-m", "artifex.cli", "render-node", "serve", "--config",
            str(config_path),
        ]
    # The task action executes python directly; it never invokes cmd.exe.
    return str(interpreter), subprocess.list2cmdline(args), str(Path.cwd().resolve())


def _run_powershell(script: str) -> str:
    _require_windows()
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    result = subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-NonInteractive",
            "-EncodedCommand", encoded,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "Windows Task Scheduler operation failed: " + detail[:2500]
        )
    return result.stdout.strip()


def _status_script(role: StartupRole) -> str:
    name = _literal(_TASKS[role])
    marker = _literal(f"{_MARKER}/{role}")
    return f"""
$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName {name} -TaskPath '\\' -ErrorAction SilentlyContinue
if ($null -eq $task) {{
  [pscustomobject]@{{ installed=$false; managed=$false; state=$null; execute=$null; arguments=$null; working_directory=$null }} | ConvertTo-Json -Compress
  exit 0
}}
$action = @($task.Actions)[0]
[pscustomobject]@{{
  installed=$true
  managed=($task.Description -eq {marker})
  state=[string]$task.State
  execute=[string]$action.Execute
  arguments=[string]$action.Arguments
  working_directory=[string]$action.WorkingDirectory
}} | ConvertTo-Json -Compress
"""


def task_status(role: StartupRole) -> StartupTaskStatus:
    _require_windows()
    data: Any = json.loads(_run_powershell(_status_script(role)))
    return StartupTaskStatus(
        role=role,
        task_name=_TASKS[role],
        installed=bool(data["installed"]),
        managed=bool(data["managed"]),
        state=data.get("state"),
        execute=data.get("execute"),
        arguments=data.get("arguments"),
        working_directory=data.get("working_directory"),
    )


def _install_script(
    role: StartupRole,
    *,
    execute: str,
    arguments: str,
    working_directory: str,
    replace: bool,
    restart_count: int,
) -> str:
    name = _literal(_TASKS[role])
    marker = _literal(f"{_MARKER}/{role}")
    force = "$true" if replace else "$false"
    return f"""
$ErrorActionPreference = 'Stop'
$taskName = {name}
$ownerMarker = {marker}
$old = Get-ScheduledTask -TaskName $taskName -TaskPath '\\' -ErrorAction SilentlyContinue
if ($null -ne $old) {{
  if ($old.Description -ne $ownerMarker) {{
    throw 'Refusing to overwrite a task not owned by Artifex'
  }}
  if (-not {force}) {{
    throw 'Artifex startup task already exists; use --replace to update it'
  }}
}}
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute {_literal(execute)} -Argument {_literal(arguments)} -WorkingDirectory {_literal(working_directory)}
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settingsArgs = @{{
  ExecutionTimeLimit = (New-TimeSpan -Seconds 0)
  MultipleInstances = 'IgnoreNew'
  StartWhenAvailable = $true
  AllowStartIfOnBatteries = $true
  DontStopIfGoingOnBatteries = $true
}}
if ({restart_count} -gt 0) {{
  $settingsArgs.RestartCount = {restart_count}
  $settingsArgs.RestartInterval = (New-TimeSpan -Minutes 1)
}}
$settings = New-ScheduledTaskSettingsSet @settingsArgs
Register-ScheduledTask -TaskName $taskName -TaskPath '\\' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description $ownerMarker -Force | Out-Null
"""


def install_task(
    role: StartupRole,
    *,
    config: Path,
    replace: bool = False,
    restart_count: int = 10,
) -> StartupTaskStatus:
    _require_windows()
    if not 0 <= restart_count <= 999:
        raise ValueError("restart_count must be between 0 and 999")
    execute, arguments, workdir = _command(role, config)
    _run_powershell(
        _install_script(
            role,
            execute=execute,
            arguments=arguments,
            working_directory=workdir,
            replace=replace,
            restart_count=restart_count,
        )
    )
    result = task_status(role)
    if not result.installed or not result.managed:
        raise RuntimeError("Artifex scheduled task was not registered as expected")
    return result


def uninstall_task(role: StartupRole) -> bool:
    _require_windows()
    name = _literal(_TASKS[role])
    marker = _literal(f"{_MARKER}/{role}")
    script = f"""
$ErrorActionPreference = 'Stop'
$old = Get-ScheduledTask -TaskName {name} -TaskPath '\\' -ErrorAction SilentlyContinue
if ($null -eq $old) {{ exit 0 }}
if ($old.Description -ne {marker}) {{
  throw 'Refusing to remove a task not owned by Artifex'
}}
Unregister-ScheduledTask -TaskName {name} -TaskPath '\\' -Confirm:$false
"""
    installed = task_status(role).installed
    if installed:
        _run_powershell(script)
    return installed

def task_configuration_matches(
    role: StartupRole,
    *,
    config: Path,
    status: StartupTaskStatus,
) -> tuple[bool, str]:
    """Compare an installed task with the *current* Python, config and working dir."""
    _require_windows()
    execute, arguments, working_directory = _command(role, config)
    if not status.installed:
        return False, "Artifex startup task is not installed"
    if not status.managed:
        return False, "refusing an unowned Windows scheduled task"
    if (
        status.execute != execute
        or status.arguments != arguments
        or status.working_directory != working_directory
    ):
        return False, (
            "Artifex task launch configuration differs from current Python, "
            "working directory or selected --config; explicit --replace is required"
        )
    return True, "Artifex-owned task matches current executable and config"


def _start_script(
    role: StartupRole,
    *,
    execute: str,
    arguments: str,
    working_directory: str,
) -> str:
    name = _literal(_TASKS[role])
    marker = _literal(f"{_MARKER}/{role}")
    return f"""
$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName {name} -TaskPath '\\' -ErrorAction SilentlyContinue
if ($null -eq $task) {{ throw 'Artifex scheduled task does not exist' }}
if ($task.Description -ne {marker}) {{
  throw 'Refusing to start a task not owned by Artifex'
}}
$action = @($task.Actions)[0]
if (
  $action.Execute -cne {_literal(execute)} -or
  $action.Arguments -cne {_literal(arguments)} -or
  $action.WorkingDirectory -cne {_literal(working_directory)}
) {{ throw 'Artifex scheduled task changed after preflight; refusing to start' }}
if ([string]$task.State -eq 'Running') {{ exit 0 }}
Start-ScheduledTask -TaskName {name} -TaskPath '\\'
"""


def activate_task(
    role: StartupRole,
    *,
    config: Path,
    install_missing: bool = False,
    replace: bool = False,
) -> tuple[StartupTaskStatus, str]:
    """Explicitly start one verified task without spawning a second daemon.

    Never modifies a task not owned by Artifex. Existing changed launch
    settings require explicit opt-in replacement; running tasks are not
    replaced under a live process. Called only after CLI --apply consent.
    """
    _require_windows()
    execute, arguments, working_directory = _command(role, config)
    current = task_status(role)
    action = "already_running"
    if not current.installed:
        if not install_missing:
            raise ValueError(
                "Artifex startup task missing; pass --install-missing --apply"
            )
        current = install_task(role, config=config)
        action = "installed_and_started"
    compatible, detail = task_configuration_matches(
        role, config=config, status=current
    )
    if not compatible:
        if not current.managed:
            raise ValueError(detail)
        if not replace:
            raise ValueError(detail)
        if (current.state or "").casefold() == "running":
            raise ValueError(
                "Cannot replace a running Artifex task; stop it safely first"
            )
        current = install_task(role, config=config, replace=True)
        action = "replaced_and_started"
        compatible, detail = task_configuration_matches(
            role, config=config, status=current
        )
        if not compatible:
            raise RuntimeError(detail)
    if (current.state or "").casefold() != "running":
        _run_powershell(
            _start_script(
                role,
                execute=execute,
                arguments=arguments,
                working_directory=working_directory,
            )
        )
        if action == "already_running":
            action = "started"
    status = task_status(role)
    compatible, detail = task_configuration_matches(
        role, config=config, status=status
    )
    if not compatible:
        raise RuntimeError("scheduled task drifted during activation: " + detail)
    return status, action
