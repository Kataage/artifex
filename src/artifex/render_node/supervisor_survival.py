"""Passive native PC-B ComfyUI survival observations; NEVER stop the supervisor.

The observer does not cause a supervisor-loss transition. It can only witness
a naturally occurring transition, so a simulation/CI PASS is never real-host
qualification and the resulting file cannot authorize renderer operations.
"""
from __future__ import annotations

import json
import math
import ntpath
import os
import socket
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings
from artifex.render_node.owner_audit import observe_renderer_owner, save_owner_observation
from artifex.render_node.process_identity import WindowsProcessIdentity, windows_process_identity
from artifex.windows_tasks import StartupTaskStatus, task_configuration_matches, task_status

_REQUIRED = frozenset({
    "native_windows", "protected_configuration", "scheduler_policy",
    "receipt", "process_identity", "launcher_identity", "tcp_ownership",
    "snapshot_consistency",
})
_MAX_SECONDS = 24 * 3600
_MAX_SAMPLES = 4096
# Match the PC-A untrusted-trace replay check; large OS clock corrections
# cannot masquerade as elapsed independent observation time.
MAX_SURVIVAL_CLOCK_DRIFT_SECONDS = 30.0
# PowerShell command is static. Configuration/path input is never interpolated.
_PROCESS_INVENTORY_SCRIPT = (
    "$ErrorActionPreference='Stop'; "
    "$items=@(Get-CimInstance Win32_Process -ErrorAction Stop | "
    "Where-Object { [string]$_.CommandLine -match '(?i)\\s+-m\\s+artifex\\.cli\\s+render-node\\s+serve\\s+--config\\s+' } | "
    "ForEach-Object { [pscustomobject]@{ "
    "ProcessId=[int]$_.ProcessId; "
    "CreationDate=$_.CreationDate.ToUniversalTime().ToString('o'); "
    "ExecutablePath=[string]$_.ExecutablePath; "
    "CommandLine=[string]$_.CommandLine; "
    "ParentProcessId=[int]$_.ParentProcessId } }); "
    "ConvertTo-Json -InputObject $items -Compress -Depth 3"
)


class SurvivalSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed_utc: datetime
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    host: str
    node_id: str
    state: Literal["verified", "blocked", "unsupported"]
    task_state: str | None
    scheduler_policy_verified: bool
    # Exact supervisor PIDs AND CIM creation times, including venv shims.
    supervisor_identities: tuple[tuple[int, str], ...]
    original_supervisor_pids_absent: bool = False
    owner_audit_state: str
    owner_checks: dict[str, str]
    owner_process_verified: bool
    actual_listener_pid: int | None
    actual_listener_started_utc: str | None
    launcher_pid: int | None
    receipt_schema: int | None


class SurvivalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    host: str
    node_id: str
    status: Literal["observed_after_supervisor_absence", "inconclusive", "blocked"]
    reason: str
    samples: tuple[SurvivalSample, ...]
    # This observation does not prove GPU recovery, reattachment on next
    # scheduler startup, or a production child surviving a forced kill.
    same_comfyui_seen_before_and_after: bool = False
    supervisor_absence_observed: bool = False
    child_survival_qualified: Literal[False] = False
    renderer_restart_authorized: Literal[False] = False
    GPU_jobs_submitted: Literal[False] = False
    task_actions_executed: Literal[False] = False
    services_mutated: Literal[False] = False
    issue_93_closure_authorized: Literal[False] = False
    production_qualified: Literal[False] = False


def supervisor_process_identities(status: StartupTaskStatus) -> tuple[tuple[int, str], ...]:
    """Native CIM inventory of task command; include venv forwarding children.

    Return safe PID/time tuples only. A missing/ambiguous CIM result MUST NOT
    be interpreted as proof that the original supervisor disappeared.
    """
    if os.name != "nt":
        raise OSError("Native Windows process inventory required")
    if not status.installed or not status.managed or not status.arguments:
        raise ValueError("The managed renderer task action is unavailable")
    if "-m artifex.cli render-node serve --config " not in status.arguments:
        raise ValueError("Unexpected renderer task action")
    run = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         _PROCESS_INVENTORY_SCRIPT],
        capture_output=True, text=True, encoding="utf-8", timeout=25,
        shell=False, check=False,
    )
    if run.returncode or len(run.stdout) > 65536:
        raise OSError("Could not inspect bounded native supervisor inventory")
    raw: Any = json.loads(run.stdout.lstrip("\ufeff"))
    if not isinstance(raw, list) or len(raw) > 256:
        raise ValueError("Invalid or excessive native process inventory")
    candidates: list[tuple[int, str]] = []
    for entry in raw:
        process = WindowsProcessIdentity.model_validate(entry)
        if (
            ntpath.basename(process.executable).casefold() in {"python.exe", "pythonw.exe"}
            and process.command_line.casefold().endswith(
                (" " + status.arguments).casefold()
            )
        ):
            candidates.append((process.pid, process.started_utc))
    if len({pid for pid, _ in candidates}) != len(candidates):
        raise ValueError("Duplicate supervisor PID identity")
    return tuple(sorted(candidates))


def sample_supervisor_survival(
    settings: ArtifexSettings,
    *,
    config: Path,
    elapsed_seconds: float = 0,
    original_supervisor_ids: tuple[int, ...] = (),
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> SurvivalSample:
    """Only Task Scheduler, receipt, CIM and TCP reads. Never run a task."""
    tick = now()
    host = socket.gethostname()
    node_id = settings.render_agent.node_id
    if os.name != "nt":
        return SurvivalSample(
            observed_utc=tick, elapsed_seconds=elapsed_seconds,
            host=host, node_id=node_id, state="unsupported", task_state=None,
            scheduler_policy_verified=False, supervisor_identities=(),
            owner_audit_state="unsupported", owner_checks={},
            owner_process_verified=False, actual_listener_pid=None,
            actual_listener_started_utc=None, launcher_pid=None,
            receipt_schema=None,
        )
    task: StartupTaskStatus | None = None
    matching = False
    supervisor: tuple[tuple[int, str], ...] = ()
    try:
        task = task_status("renderer")
        matching, _ = task_configuration_matches(
            "renderer", config=config, status=task,
        )
        if matching:
            supervisor = supervisor_process_identities(task)
    except (OSError, RuntimeError, ValueError, TypeError):
        matching = False
    try:
        audit = observe_renderer_owner(settings, config=config)
    except (OSError, RuntimeError, ValueError, TypeError):
        audit = {}
    checks = audit.get("checks")
    values = (
        {key: item.get("status", "unknown") for key, item in checks.items()}
        if isinstance(checks, dict) else {}
    )
    ownership_ok = (
        audit.get("process_observation_verified") is True
        and audit.get("receipt_schema") == 2
        and isinstance(audit.get("actual_listener_pid"), int)
        and isinstance(audit.get("launcher_pid"), int)
        and isinstance(audit.get("actual_process_started_utc"), str)
        and _REQUIRED.issubset(values)
        and all(values[key] == "pass" for key in _REQUIRED)
        and audit.get("restart_authorized") is False
        and audit.get("child_survival_qualified") is False
        and audit.get("production_qualified") is False
        and audit.get("mutated_services") is False
    )
    task_state = task.state if task else None
    original_pids_absent = False
    if matching and task_state == "Ready" and original_supervisor_ids:
        try:
            # An exact task argv is not enough: independently confirm every
            # original supervisor PID is now gone, not merely renamed or
            # hidden by a changed command line. Reuse is ambiguous => block.
            original_pids_absent = all(
                windows_process_identity(pid) is None
                for pid in original_supervisor_ids
            )
        except (OSError, RuntimeError, ValueError, TypeError):
            original_pids_absent = False
    state: Literal["verified", "blocked", "unsupported"] = (
        "verified" if matching and ownership_ok and task_state in {"Running", "Ready"}
        else "blocked"
    )
    return SurvivalSample(
        observed_utc=tick, elapsed_seconds=elapsed_seconds,
        host=host, node_id=node_id, state=state,
        task_state=task_state, scheduler_policy_verified=matching,
        supervisor_identities=supervisor,
        original_supervisor_pids_absent=original_pids_absent,
        owner_audit_state=str(audit.get("status", "unavailable")),
        owner_checks=values,
        owner_process_verified=ownership_ok,
        actual_listener_pid=audit.get("actual_listener_pid"),
        actual_listener_started_utc=audit.get("actual_process_started_utc"),
        launcher_pid=audit.get("launcher_pid"),
        receipt_schema=audit.get("receipt_schema"),
    )


def _aware_native_time(value: str) -> datetime | None:
    """Reject malformed/naive Windows CIM creation timestamps."""
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return stamp.astimezone(UTC) if stamp.tzinfo is not None else None


def _valid_sample_time(sample: SurvivalSample) -> bool:
    """A historical sample must describe a possible native Windows timeline."""
    observed = sample.observed_utc
    if not math.isfinite(sample.elapsed_seconds):
        return False
    if observed.tzinfo is None or observed.utcoffset() is None:
        return False
    if not sample.host.strip() or not sample.node_id.strip():
        return False
    if sample.actual_listener_pid is None or sample.actual_listener_pid <= 0:
        return False
    started = _aware_native_time(sample.actual_listener_started_utc or "")
    if started is None or started > observed.astimezone(UTC):
        return False
    if len({pid for pid, _ in sample.supervisor_identities}) != len(
        sample.supervisor_identities
    ):
        return False
    return all(
        pid > 0 and (process_start := _aware_native_time(utc)) is not None
        and process_start <= observed.astimezone(UTC)
        for pid, utc in sample.supervisor_identities
    )


def assess_survival(samples: tuple[SurvivalSample, ...], *,
                    min_separation_seconds: float) -> SurvivalAssessment:
    """Only naturally witnessed Task Running -> Ready + identical GPU child.

    Require two separate Ready/absent samples and zero unexpected supervisor
    changes. A mere healthy ComfyUI, receipt or Task Scheduler state is not
    evidence. An incomplete observation stays inconclusive or blocked.
    """
    if not samples:
        raise ValueError("No samples")
    if not math.isfinite(min_separation_seconds) or min_separation_seconds <= 0:
        raise ValueError("Positive finite survival sample separation required")
    baseline = samples[0]
    status: Literal["observed_after_supervisor_absence", "inconclusive", "blocked"]
    status = "inconclusive"
    reason = "natural_supervisor_transition_not_observed"
    found = False
    absent = False
    initial_supervisors = set(baseline.supervisor_identities)
    base_identity = (
        baseline.actual_listener_pid, baseline.actual_listener_started_utc,
        baseline.launcher_pid, baseline.receipt_schema,
    )
    if (
        not _valid_sample_time(baseline)
        or baseline.state != "verified" or baseline.task_state != "Running"
        or not initial_supervisors or baseline.owner_audit_state != "observed_stable"
        or baseline.owner_checks.get("scheduler_running") != "pass"
        or baseline.actual_listener_pid in {pid for pid, _ in initial_supervisors}
        or baseline.owner_process_verified is not True
        or not baseline.scheduler_policy_verified
        or not _REQUIRED.issubset(baseline.owner_checks)
        or any(baseline.owner_checks[key] != "pass" for key in _REQUIRED)
    ):
        status, reason = "blocked", "initial_supervisor_or_comfyui_not_verified"
    else:
        last_elapsed = -1.0
        last_observed: datetime | None = None
        baseline_observed = baseline.observed_utc.astimezone(UTC)
        first_absent: float | None = None
        for item in samples:
            if not _valid_sample_time(item):
                status, reason = "blocked", "invalid_native_observation_or_creation_time"
                break
            current_observed = item.observed_utc.astimezone(UTC)
            if last_observed is not None and current_observed <= last_observed:
                status, reason = "blocked", "sample_wall_clock_reversed_or_duplicated"
                break
            last_observed = current_observed
            # The independent monotonic clock and UTC wall clock must describe
            # the same sampling window. A large forward clock correction can
            # otherwise produce an apparent PASS on PC-B that PC-A will reject.
            clock_difference = abs(
                (current_observed - baseline_observed).total_seconds()
                - (item.elapsed_seconds - baseline.elapsed_seconds)
            )
            if clock_difference > MAX_SURVIVAL_CLOCK_DRIFT_SECONDS:
                status, reason = "blocked", "sample_wall_clock_elapsed_diverged"
                break
            same = (
                item.host == baseline.host and item.node_id == baseline.node_id
                and (
                    item.actual_listener_pid, item.actual_listener_started_utc,
                    item.launcher_pid, item.receipt_schema,
                ) == base_identity
            )
            if item.elapsed_seconds <= last_elapsed and last_elapsed >= 0:
                status, reason = "blocked", "sample_clock_reversed_or_duplicated"
                break
            if last_elapsed >= 0 and item.elapsed_seconds - last_elapsed > 2 * min_separation_seconds:
                status, reason = "blocked", "survival_observer_sampling_gap"
                break
            last_elapsed = item.elapsed_seconds
            checks_ok = (
                item.owner_process_verified
                and item.scheduler_policy_verified
                and _REQUIRED.issubset(item.owner_checks)
                and all(item.owner_checks[key] == "pass" for key in _REQUIRED)
            )
            if item.state != "verified" or not same or not checks_ok:
                status, reason = "blocked", "live_comfyui_identity_or_receipt_unverified"
                break
            if item.task_state == "Running":
                if not item.supervisor_identities or set(item.supervisor_identities) != initial_supervisors:
                    status, reason = "blocked", "supervisor_identity_changed"
                    break
                if first_absent is not None:
                    status, reason = "blocked", "supervisor_reappeared"
                    break
                if item.owner_audit_state != "observed_stable":
                    status, reason = "blocked", "running_owner_audit_incomplete"
                    break
            elif item.task_state == "Ready":
                if item.supervisor_identities or not item.original_supervisor_pids_absent:
                    status, reason = "blocked", "original_supervisor_not_proven_absent"
                    break
                if item.owner_audit_state not in {"inconclusive", "observed_stable"}:
                    status, reason = "blocked", "owner_audit_after_exit_unverified"
                    break
                if item.owner_checks.get("scheduler_running") != "unknown":
                    status, reason = "blocked", "scheduler_absence_not_witnessed"
                    break
                if first_absent is None:
                    first_absent = item.elapsed_seconds
                elif item.elapsed_seconds - first_absent >= min_separation_seconds:
                    status, reason = (
                        "observed_after_supervisor_absence",
                        "same_real_comfyui_process_observed_across_natural_task_exit",
                    )
                    found = absent = True
            else:
                status, reason = "blocked", "unsupported_task_state"
                break
    return SurvivalAssessment(
        host=baseline.host, node_id=baseline.node_id,
        status=status, reason=reason, samples=samples,
        same_comfyui_seen_before_and_after=found,
        supervisor_absence_observed=absent,
    )


def observe_supervisor_survival(
    settings: ArtifexSettings,
    *, config: Path, output: Path,
    duration_seconds: float = 300,
    interval_seconds: float = 10,
    sample: Callable[..., SurvivalSample] = sample_supervisor_survival,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> SurvivalAssessment:
    if not 20 <= duration_seconds <= _MAX_SECONDS:
        raise ValueError("Duration must be between 20 seconds and 24 hours")
    if not 5 <= interval_seconds <= 3600 or interval_seconds > duration_seconds / 2:
        raise ValueError("Interval must permit two independent post-exit samples")
    if duration_seconds / interval_seconds > _MAX_SAMPLES - 1:
        raise ValueError("Observer sample count exceeds bound")
    if config.is_symlink() or not config.is_file():
        raise ValueError("Existing non-symlinked native PC-B config required")
    target = output.expanduser().absolute()
    if any(item.is_symlink() for item in (target, *target.parents)) or target.exists():
        raise FileExistsError("Refusing existing or symlinked survival evidence output")
    started = monotonic()
    items: list[SurvivalSample] = []
    while len(items) <= _MAX_SAMPLES:
        elapsed = max(0.0, monotonic() - started)
        original = tuple(pid for pid, _ in items[0].supervisor_identities) if items else ()
        observed = sample(
            settings, config=config, elapsed_seconds=elapsed,
            original_supervisor_ids=original, now=now,
        )
        items.append(observed)
        verdict = assess_survival(tuple(items), min_separation_seconds=interval_seconds)
        if verdict.status != "inconclusive" or elapsed >= duration_seconds:
            break
        sleep(min(interval_seconds, duration_seconds - elapsed))
    # Output the complete observed trace, not just a caller-supplied boolean.
    # Exclusive creation; no change to receipt, task, ComfyUI or GPU.
    save_owner_observation(verdict.model_dump(mode="json"), target)
    return verdict
