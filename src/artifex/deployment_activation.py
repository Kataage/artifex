from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings
from artifex.deployment import DeploymentReport, DeploymentRole, verify_deployment
from artifex.render_node.observer_heartbeat import (
    Status as ObserverHeartbeatStatus,
    inspect_observer_heartbeat,
)
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
    # A healthy endpoint alone may belong to an old/foreign worker. This
    # native Task Scheduler state is necessary, but not sufficient, for ready.
    task_running_observed: bool = False
    ready: bool
    observer_requested: bool = False
    observer_action: str | None = None
    observer_task: StartupTaskStatus | None = None
    observer_task_compatible: bool | None = None
    observer_heartbeat: ObserverHeartbeatStatus | None = None
    observer_ready: bool = False
    # Observer liveness is a prerequisite only, never actual survival proof.
    native_child_survival_proven: bool = False
    production_qualified: bool = False


async def activate_deployment(
    settings: ArtifexSettings,
    *,
    role: DeploymentRole,
    config: Path,
    apply: bool = False,
    install_missing: bool = False,
    replace: bool = False,
    with_observer: bool = False,
    wait_seconds: float = 90,
    poll_seconds: float = 5,
    status_fn: Callable[[StartupRole], StartupTaskStatus] = task_status,
    match_fn: Callable[..., tuple[bool, str]] = task_configuration_matches,
    start_fn: Callable[..., tuple[StartupTaskStatus, str]] = activate_task,
    verify_fn: Callable[..., Awaitable[DeploymentReport]] = verify_deployment,
    heartbeat_fn: Callable[..., ObserverHeartbeatStatus] = inspect_observer_heartbeat,
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
    if with_observer and role != "renderer":
        raise ValueError("--with-observer is valid only on the PC-B renderer role")
    if with_observer and not settings.render_agent.comfyui_process.enabled:
        raise ValueError("--with-observer requires managed ComfyUI to be configured")
    if not config.expanduser().is_file() or config.is_symlink():
        raise ValueError("activate requires an existing regular non-symlink config")
    startup_role: StartupRole = role
    task = await asyncio.to_thread(status_fn, startup_role)
    compatible, detail = await asyncio.to_thread(
        match_fn, startup_role, config=config, status=task
    )
    observer: StartupTaskStatus | None = None
    observer_compatible: bool | None = None
    observer_action: str | None = None
    observer_heartbeat: ObserverHeartbeatStatus | None = None
    observer_ready = False
    if with_observer:
        # Inspect both registrations BEFORE any apply. In particular, never
        # start the watcher as a partial side effect when renderer activation
        # would already fail its known ownership/config checks.
        observer = await asyncio.to_thread(status_fn, "survival-observer")
        observer_compatible, observer_detail = await asyncio.to_thread(
            match_fn, "survival-observer", config=config, status=observer
        )
        if apply:
            if observer.installed and not observer_compatible:
                raise ValueError(
                    "Unsafe or stale survival-observer task; inspect it read-only "
                    "and resolve configuration drift before activation"
                )
            if not observer.installed and not install_missing:
                raise ValueError(
                    "Survival observer is not installed; explicit --install-missing "
                    "is required before combined activation"
                )
            if not compatible and (
                (not task.installed and not install_missing)
                or (task.installed and (
                    not task.managed or not replace
                    or (task.state or "").casefold() == "running"
                ))
            ):
                raise ValueError(
                    "Renderer registration is absent, foreign, running with drift, "
                    "or incompatible; combined activation refused before any task start"
                )
        observer_action = "preview"
    action = "preview"
    if apply:
        if with_observer:
            # The independently scheduled read-only observer must start first.
            # Never replace even an Artifex-owned observer automatically.
            observer, observer_action = await asyncio.to_thread(
                start_fn, "survival-observer", config=config,
                install_missing=install_missing, replace=False,
            )
            observer_compatible, _ = await asyncio.to_thread(
                match_fn, "survival-observer", config=config, status=observer
            )
            if not observer_compatible:
                raise RuntimeError(
                    "Survival observer drifted during startup; renderer not started"
                )
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
        # Re-read the native task AFTER the live health checks, not just at
        # the beginning of the activation. A task might have exited or been
        # replaced while the deployment services were being probed.
        task = await asyncio.to_thread(status_fn, startup_role)
        compatible, detail = await asyncio.to_thread(
            match_fn, startup_role, config=config, status=task
        )
        task_running = bool(
            task.installed and task.managed
            and (task.state or "").casefold() == "running"
        )
        if with_observer:
            observer = await asyncio.to_thread(status_fn, "survival-observer")
            observer_compatible, _ = await asyncio.to_thread(
                match_fn, "survival-observer", config=config, status=observer
            )
            try:
                observer_heartbeat = await asyncio.to_thread(
                    heartbeat_fn, settings, config=config,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                observer_heartbeat = "unavailable"
            observer_ready = bool(
                observer.installed and observer.managed
                and (observer.state or "").casefold() == "running"
                and observer_compatible
                and observer_heartbeat == "fresh"
            )
        # A Task Scheduler "Running" label is NOT proof of service health;
        # conversely a stale service endpoint is NOT proof that the intended
        # Artifex Task Scheduler task is actually running. Require both.
        ready = (
            compatible and task_running and report.ready
            and (not with_observer or observer_ready)
        )
        # Registration drift is unsafe, not a transient startup delay.
        # Failed heartbeats can recover; never restart the watcher to force one.
        if (
            ready or not apply or not compatible
            or (with_observer and not observer_compatible)
            or monotonic_fn() >= deadline
        ):
            return ActivationReport(
                role=role,
                mode="apply" if apply else "preview",
                action=action,
                task_compatible=compatible,
                task_detail=detail,
                task=task,
                deployment=report,
                checks_performed=checks,
                task_running_observed=task_running,
                ready=ready,
                observer_requested=with_observer,
                observer_action=observer_action,
                observer_task=observer,
                observer_task_compatible=observer_compatible,
                observer_heartbeat=observer_heartbeat,
                observer_ready=observer_ready,
            )
        await sleep_fn(min(poll_seconds, max(0, deadline - monotonic_fn())))
