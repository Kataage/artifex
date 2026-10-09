"""Read-only, path-redacted native Windows two-PC installation inspection.

This module never creates/starts/stops/replaces tasks or touches ComfyUI.
Reports describe installation and observational prerequisites ONLY.
"""
from __future__ import annotations

import platform
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.render_node.observer_heartbeat import (
    Status as ObserverHeartbeatStatus,
    inspect_observer_heartbeat,
)
from artifex.windows_tasks import (
    StartupRole,
    StartupTaskStatus,
    task_configuration_matches,
    task_status,
)

TaskCondition = Literal[
    "running", "registered_not_running", "missing", "unsafe", "unavailable",
]
SpoolCondition = Literal["has_evidence", "empty", "missing", "unsafe", "unavailable"]


class TaskInstallationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: StartupRole
    status: TaskCondition
    registered: bool
    managed: bool
    configuration_verified: bool
    state: str | None = None
    reason_code: str


class RendererInstallationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    node_id: str
    captured_utc: datetime
    hostname: str
    native_windows: bool
    managed_renderer_configured: bool
    renderer_task: TaskInstallationCheck
    survival_observer_task: TaskInstallationCheck
    evidence_spool: SpoolCondition
    observer_heartbeat: ObserverHeartbeatStatus = "missing"
    safe_for_passive_observation: bool
    task_actions_executed: Literal[False] = False
    renderer_process_mutated: Literal[False] = False
    actual_survival_observed: Literal[False] = False
    real_gpu_qualified: Literal[False] = False
    issue_93_closure_authorized: Literal[False] = False
    issue_40_closure_authorized: Literal[False] = False
    production_qualified: Literal[False] = False


class RemoteRendererInstallationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    audit: RendererInstallationAudit


def inspect_registered_task(
    role: StartupRole,
    *,
    config: Path,
    native_windows: bool,
    probe: Callable[[StartupRole], StartupTaskStatus] = task_status,
    verify: Callable[..., tuple[bool, str]] = task_configuration_matches,
) -> TaskInstallationCheck:
    """Never return executable, arguments, working directory or raw exceptions."""
    if not native_windows:
        return TaskInstallationCheck(
            role=role, status="unavailable", registered=False, managed=False,
            configuration_verified=False, reason_code="native_windows_required",
        )
    try:
        task = probe(role)
    except (OSError, RuntimeError, ValueError, TypeError):
        return TaskInstallationCheck(
            role=role, status="unavailable", registered=False, managed=False,
            configuration_verified=False, reason_code="scheduler_read_failed",
        )
    if not task.installed:
        return TaskInstallationCheck(
            role=role, status="missing", registered=False, managed=False,
            configuration_verified=False, reason_code="task_not_registered",
        )
    if not task.managed:
        return TaskInstallationCheck(
            role=role, status="unsafe", registered=True, managed=False,
            configuration_verified=False, state=task.state,
            reason_code="task_not_artifex_owned",
        )
    try:
        good, _ = verify(role, config=config, status=task)
    except (OSError, RuntimeError, ValueError, TypeError):
        good = False
    if not good:
        return TaskInstallationCheck(
            role=role, status="unsafe", registered=True, managed=True,
            configuration_verified=False, state=task.state,
            reason_code="task_command_or_policy_mismatch",
        )
    return TaskInstallationCheck(
        role=role,
        status="running" if task.state == "Running" else "registered_not_running",
        registered=True, managed=True, configuration_verified=True,
        state=task.state,
        reason_code="verified_running" if task.state == "Running"
        else "verified_but_not_running",
    )


def inspect_spool(path: Path) -> SpoolCondition:
    """Bounded directory metadata check; never read or copy evidence contents."""
    target = path.expanduser().absolute()
    try:
        if any(part.is_symlink() for part in (target, *target.parents)):
            return "unsafe"
        if not target.exists():
            return "missing"
        if not target.is_dir():
            return "unsafe"
        # PR #122's spool filename is fixed; ignore unrelated user files.
        from artifex.render_node.survival_spool import _MAX_ENTRIES, _TRACE_NAME

        matching = 0
        for entry in target.iterdir():
            if _TRACE_NAME.fullmatch(entry.name):
                matching += 1
                if matching > _MAX_ENTRIES or entry.is_symlink() or not entry.is_file():
                    return "unsafe"
        return "has_evidence" if matching else "empty"
    except (OSError, ValueError):
        return "unavailable"


def inspect_renderer_installation(
    settings: ArtifexSettings,
    *,
    owner_config: Path | None,
    native_windows: bool | None = None,
    now: datetime | None = None,
    probe: Callable[[StartupRole], StartupTaskStatus] = task_status,
    verify: Callable[..., tuple[bool, str]] = task_configuration_matches,
) -> RendererInstallationAudit:
    """Observe local PC-B config, Task Scheduler and spool without mutations."""
    native = platform.system() == "Windows" if native_windows is None else native_windows
    config_ok = (
        owner_config is not None
        and owner_config.is_file()
        and not any(
            p.is_symlink() for p in (owner_config, *owner_config.parents)
        )
    )
    task_config = owner_config or Path("config/render-node.yaml")
    def checked(role: StartupRole) -> TaskInstallationCheck:
        return inspect_registered_task(
            role, config=task_config, native_windows=native and config_ok,
            probe=probe, verify=verify,
        )
    renderer = checked("renderer")
    observer = checked("survival-observer")
    managed = bool(settings.render_agent.comfyui_process.enabled)
    spool = inspect_spool(settings.render_agent.survival_evidence_dir)
    heartbeat = inspect_observer_heartbeat(settings, now=now)
    return RendererInstallationAudit(
        node_id=settings.render_agent.node_id,
        captured_utc=now or datetime.now(UTC),
        hostname=socket.gethostname(),
        native_windows=native,
        managed_renderer_configured=managed,
        renderer_task=renderer,
        survival_observer_task=observer,
        evidence_spool=spool,
        observer_heartbeat=heartbeat,
        safe_for_passive_observation=(
            native and config_ok and managed
            and renderer.status == "running"
            and observer.status == "running"
            and heartbeat == "fresh"
            and spool in {"has_evidence", "empty", "missing"}
        ),
    )
