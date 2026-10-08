from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings
from artifex.deployment import DeploymentReport, DeploymentRole, verify_deployment
from artifex.windows_tasks import (
    StartupRole,
    StartupTaskStatus,
    activate_task,
    task_configuration_matches,
    task_status,
)


class ActivationReport(BaseModel):
    """One-host native Windows bring-up status. Not a production qualification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: DeploymentRole
    mode: Literal["preview", "apply"]
    action: str
    task_compatible: bool
    task_detail: str
    task: StartupTaskStatus
    deployment: DeploymentReport
    checks_performed: int = Field(ge=1)
    ready: bool
    production_qualified: bool = False


async def activate_deployment(
    settings: ArtifexSettings,
    *,
    role: DeploymentRole,
    config: Path,
    apply: bool = False,
    install_missing: bool = False,
    replace: bool = False,
    wait_seconds: float = 90,
    poll_seconds: float = 5,
    status_fn: Callable[[StartupRole], StartupTaskStatus] = task_status,
    match_fn: Callable[..., tuple[bool, str]] = task_configuration_matches,
    start_fn: Callable[..., tuple[StartupTaskStatus, str]] = activate_task,
    verify_fn: Callable[..., Awaitable[DeploymentReport]] = verify_deployment,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> ActivationReport:
    """Preview or explicitly launch only this PC's Artifex-owned scheduled task.

    The controller and renderer are deliberately activated *on their own PCs*:
    no remote PowerShell execution, credential delegation, Docker or implied
    GPU render is performed. Deployment verification is live and read-only.
    """
    if wait_seconds < 0 or wait_seconds > 600:
        raise ValueError("wait_seconds must be between 0 and 600")
    if not 0 < poll_seconds <= 60:
        raise ValueError("poll_seconds must be between 0 and 60")
    if (install_missing or replace) and not apply:
        raise ValueError("--install-missing and --replace require --apply")
    if not config.expanduser().is_file() or config.is_symlink():
        raise ValueError("activate requires an existing regular non-symlink config")
    startup_role: StartupRole = role
    task = await asyncio.to_thread(status_fn, startup_role)
    compatible, detail = await asyncio.to_thread(
        match_fn, startup_role, config=config, status=task
    )
    action = "preview"
    if apply:
        # Calling start_fn performs all ownership/config checks again in
        # PowerShell, guarding against scheduler task replacement races.
        task, action = await asyncio.to_thread(
            start_fn,
            startup_role,
            config=config,
            install_missing=install_missing,
            replace=replace,
        )
        compatible, detail = await asyncio.to_thread(
            match_fn, startup_role, config=config, status=task
        )
        if not compatible:
            raise RuntimeError("task activation validation failed: " + detail)
    deadline = monotonic_fn() + (wait_seconds if apply else 0)
    checks = 0
    while True:
        report = await verify_fn(
            settings, role=role, require_autostart=True, render_smoke=False
        )
        checks += 1
        # A reported 'Running' Task Scheduler state is NOT used as a service
        # health signal. Only real deployment checks prove readiness.
        ready = compatible and report.ready
        if ready or not apply or monotonic_fn() >= deadline:
            return ActivationReport(
                role=role,
                mode="apply" if apply else "preview",
                action=action,
                task_compatible=compatible,
                task_detail=detail,
                task=task,
                deployment=report,
                checks_performed=checks,
                ready=ready,
            )
        await sleep_fn(min(poll_seconds, max(0, deadline - monotonic_fn())))
