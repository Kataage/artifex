"""Isolated, opt-in Windows Task Scheduler child-process lifecycle probe.

Only a uniquely named temporary task and expiring Python mock child are used.
No Artifex startup task, user ComfyUI, GPU, or configuration is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal, cast

from artifex.windows_tasks import _literal, _require_windows, _run_powershell

ProbeMode = Literal["protected", "baseline"]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _mock_child(folder: Path, nonce: str, lifetime: float) -> None:
    deadline = time.monotonic() + lifetime
    count = 0
    while time.monotonic() < deadline and not (folder / "release").exists():
        count += 1
        _atomic_json(
            folder / "heartbeat.json",
            {"nonce": nonce, "pid": os.getpid(), "ticks": count, "utc": time.time()},
        )
        time.sleep(0.2)
    _atomic_json(folder / "child-exited.json", {"nonce": nonce, "pid": os.getpid()})


def _mock_supervisor(folder: Path, nonce: str, lifetime: float) -> None:
    child = subprocess.Popen(
        [
            sys.executable, "-m", "artifex.windows_task_lifecycle_probe",
            "--child", str(folder), nonce, str(lifetime),
        ],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        shell=False,
    )
    _atomic_json(
        folder / "supervisor.json",
        {"nonce": nonce, "supervisor_pid": os.getpid(), "child_pid": child.pid},
    )
    deadline = time.monotonic() + lifetime
    while time.monotonic() < deadline and not (folder / "release").exists():
        time.sleep(0.2)


def _valid_report_dir(folder: Path) -> Path:
    candidate = folder.expanduser().absolute()
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError("Refusing symlinked lifecycle probe report directory")
    candidate.mkdir(parents=True, exist_ok=True)
    if not candidate.is_dir():
        raise ValueError("Lifecycle probe report destination must be a directory")
    return candidate


def _script_task_check(name: str, marker: str, execute: str, arguments: str) -> str:
    return f"""
$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName {_literal(name)} -TaskPath '\\' -ErrorAction Stop
if ($task.Description -cne {_literal(marker)}) {{
  throw 'Refusing to touch a scheduler task without this probe ownership marker'
}}
$actions = @($task.Actions)
if ($actions.Count -ne 1 -or
    $actions[0].Execute -cne {_literal(execute)} -or
    $actions[0].Arguments -cne {_literal(arguments)}) {{
  throw 'Refusing to touch scheduler task with modified action'
}}
"""


def _register_script(
    name: str, marker: str, execute: str, arguments: str, working_dir: str,
    mode: ProbeMode,
) -> str:
    protect = "$argsSettings.DisallowHardTerminate = $true" if mode == "protected" else ""
    return f"""
$ErrorActionPreference = 'Stop'
$old = Get-ScheduledTask -TaskName {_literal(name)} -TaskPath '\\' -ErrorAction SilentlyContinue
if ($null -ne $old) {{ throw 'Refusing to overwrite existing scheduled task' }}
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute {_literal(execute)} -Argument {_literal(arguments)} -WorkingDirectory {_literal(working_dir)}
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddDays(1)
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$argsSettings = @{{
  MultipleInstances = 'IgnoreNew'
  ExecutionTimeLimit = (New-TimeSpan -Seconds 0)
  StartWhenAvailable = $false
}}
{protect}
$settings = New-ScheduledTaskSettingsSet @argsSettings
Register-ScheduledTask -TaskName {_literal(name)} -TaskPath '\\' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description {_literal(marker)} -ErrorAction Stop | Out-Null
"""


def _task_runtime_status(name: str, marker: str, execute: str, arguments: str) -> dict[str, Any]:
    """Read Scheduler state and LastTaskResult without changing anything."""
    script = _script_task_check(name, marker, execute, arguments) + f"""
$info = Get-ScheduledTaskInfo -TaskName {_literal(name)} -TaskPath '\\' -ErrorAction Stop
[pscustomobject]@{{
  state=[string]$task.State
  last_task_result=[int64]$info.LastTaskResult
  last_run_time=[string]$info.LastRunTime
}} | ConvertTo-Json -Compress
"""
    return cast(dict[str, Any], json.loads(_run_powershell(script)))


def _identity(pid: int, nonce: str) -> dict[str, Any] | None:
    """Inspect real CIM identity: PID alone is never trustworthy evidence."""
    from artifex.render_node.process_identity import windows_process_identity

    try:
        info = windows_process_identity(pid)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if info is None or nonce not in info.command_line:
        return None
    return {
        "pid": info.pid,
        "started_utc": info.started_utc,
        "executable": info.executable,
        "command_line": info.command_line,
    }


def _unchanged_identity(before: dict[str, Any], nonce: str) -> bool:
    observed = _identity(int(before["pid"]), nonce)
    return observed is not None and (
        observed["pid"] == before["pid"]
        and observed["started_utc"] == before["started_utc"]
        and observed["executable"] == before["executable"]
        and observed["command_line"] == before["command_line"]
    )


def _poll_ready(folder: Path, nonce: str, seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        supervisor = _read_json(folder / "supervisor.json")
        beat = _read_json(folder / "heartbeat.json")
        if (
            supervisor is not None and beat is not None
            and supervisor.get("nonce") == nonce
            and beat.get("nonce") == nonce
            and isinstance(supervisor.get("supervisor_pid"), int)
            and isinstance(supervisor.get("child_pid"), int)
            and supervisor["child_pid"] == beat.get("pid")
            and isinstance(beat.get("ticks"), int) and beat["ticks"] > 0
        ):
            return supervisor, beat
        time.sleep(0.2)
    raise TimeoutError("Temporary scheduled supervisor/child never became ready")


def _check_preview(mode: ProbeMode) -> dict[str, Any]:
    return {
        "status": "preview", "mode": mode, "performed": False,
        "temporary_task_only": True, "production_qualified": False,
        "child_survival_qualified": False, "restart_authorized": False,
        "explanation": "Pass --apply to register and stop only a random test-owned scheduled task.",
    }


def run_lifecycle_probe(
    *, apply: bool = False, mode: ProbeMode = "protected",
    report_dir: Path = Path("data/qualification/scheduler-probe"),
) -> dict[str, Any]:
    """Observe real scheduler Stop-ScheduledTask semantics without touching ComfyUI.

    'protected' mirrors DisallowHardTerminate; 'baseline' allows hard stop.
    The fake child automatically expires, and cleanup only addresses the exact
    nonce-tagged task. Inconclusive != proof of survival.
    """
    if mode not in {"protected", "baseline"}:
        raise ValueError("mode must be protected or baseline")
    if not apply:
        return _check_preview(mode)
    _require_windows()
    folder_root = _valid_report_dir(report_dir)
    nonce = uuid.uuid4().hex
    name = "Artifex-LifecycleProbe-" + nonce[:20]
    marker = "Artifex isolated lifecycle probe/" + nonce
    folder = folder_root / nonce
    folder.mkdir(mode=0o700)
    executable = str(Path(sys.executable).resolve(strict=True))
    working_dir = str(Path.cwd().resolve())
    args = subprocess.list2cmdline([
        "-m", "artifex.windows_task_lifecycle_probe",
        "--supervisor", str(folder), nonce, "28",
    ])
    report: dict[str, Any] = {
        "status": "inconclusive", "mode": mode, "performed": True,
        "temporary_task_only": True, "task": name,
        "child_survival_qualified": False, "production_qualified": False,
        "restart_authorized": False, "supervisor_stopped": None,
        "child_alive_after_stop": None, "heartbeat_advanced_after_stop": None,
        "registration_checked": False, "cleanup_complete": False,
    }
    registered = False
    try:
        _run_powershell(_register_script(name, marker, executable, args, working_dir, mode))
        registered = True
        preflight = _script_task_check(name, marker, executable, args)
        policy = json.loads(_run_powershell(
            preflight + """
[pscustomobject]@{
  allow_hard_terminate=[bool]$task.Settings.AllowHardTerminate
  multiple_instances=[string]$task.Settings.MultipleInstances
} | ConvertTo-Json -Compress
"""
        ))
        if (
            policy["allow_hard_terminate"] is (mode == "baseline")
            and policy["multiple_instances"] == "IgnoreNew"
        ):
            report["registration_checked"] = True
        else:
            raise RuntimeError("Temporary task policy did not match requested probe mode")
        _run_powershell(preflight + f"\nStart-ScheduledTask -TaskName {_literal(name)} -TaskPath '\\'\n")
        supervisor, initial_beat = _poll_ready(folder, nonce, 12)
        supervisor_before = _identity(supervisor["supervisor_pid"], nonce)
        child_before = _identity(supervisor["child_pid"], nonce)
        if supervisor_before is None or child_before is None:
            raise RuntimeError("Mock process CIM identity could not be proven")
        report["supervisor_pid"] = supervisor_before["pid"]
        report["child_pid"] = child_before["pid"]
        report["initial_ticks"] = initial_beat["ticks"]
        _run_powershell(preflight + f"\nStop-ScheduledTask -TaskName {_literal(name)} -TaskPath '\\' -ErrorAction Stop\n")
        time.sleep(3)
        report["supervisor_stopped"] = not _unchanged_identity(supervisor_before, nonce)
        report["child_alive_after_stop"] = _unchanged_identity(child_before, nonce)
        final_beat = _read_json(folder / "heartbeat.json")
        report["heartbeat_advanced_after_stop"] = (
            final_beat is not None and final_beat.get("nonce") == nonce
            and final_beat.get("pid") == child_before["pid"]
            and isinstance(final_beat.get("ticks"), int)
            and final_beat["ticks"] > initial_beat["ticks"]
        )
        report["observed_ticks"] = final_beat.get("ticks") if final_beat else None
        if report["supervisor_stopped"]:
            report["status"] = (
                "child_survived_scheduler_stop"
                if report["child_alive_after_stop"] and report["heartbeat_advanced_after_stop"]
                else "child_did_not_survive_scheduler_stop"
            )
        else:
            report["status"] = "inconclusive_supervisor_still_running"
        report["child_survival_qualified"] = (
            report["status"] == "child_survived_scheduler_stop"
        )
    except (OSError, RuntimeError, ValueError, TypeError, TimeoutError) as exc:
        report["status"] = "blocked"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if registered:
            try:
                report["task_runtime"] = _task_runtime_status(
                    name, marker, executable, args,
                )
            except (OSError, RuntimeError, ValueError, TypeError) as diag_exc:
                report["task_runtime_error"] = f"{type(diag_exc).__name__}: {diag_exc}"
    finally:
        # Both mock processes voluntarily exit when release is created; the
        # scheduled task's unique marker+action is rechecked before cleanup.
        (folder / "release").write_text(nonce, encoding="utf-8")
        if registered:
            try:
                check = _script_task_check(name, marker, executable, args)
                _run_powershell(
                    check
                    + f"\nUnregister-ScheduledTask -TaskName {_literal(name)} -TaskPath '\\' -Confirm:$false -ErrorAction Stop\n"
                )
                report["cleanup_complete"] = True
            except (OSError, RuntimeError, ValueError) as exc:
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "blocked"
                report["child_survival_qualified"] = False
        report["report_path"] = str(folder / "report.json")
        _atomic_json(folder / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--child", nargs=3, metavar=("FOLDER", "NONCE", "LIFETIME"))
    group.add_argument("--supervisor", nargs=3, metavar=("FOLDER", "NONCE", "LIFETIME"))
    args = parser.parse_args()
    params = args.child or args.supervisor
    folder, nonce, lifetime = Path(params[0]), params[1], float(params[2])
    if len(nonce) != 32 or not all(char in "0123456789abcdef" for char in nonce):
        raise ValueError("Invalid isolated probe nonce")
    if not 1 <= lifetime <= 60:
        raise ValueError("Invalid mock process lifetime")
    if args.child:
        _mock_child(folder, nonce, lifetime)
    else:
        _mock_supervisor(folder, nonce, lifetime)


if __name__ == "__main__":
    main()
