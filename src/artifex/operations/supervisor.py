from __future__ import annotations

import time
from typing import Protocol

from artifex.config.models import OperationsConfig
from artifex.discord.models import Notification, NotificationKind
from artifex.domain import AgentState
from artifex.operations.health import ComponentState, HealthChecker, HealthReport
from artifex.runtime import RuntimeStore
from artifex.telemetry import EventSeverity, TelemetryRepository


class NotificationSink(Protocol):
    async def notify(self, notification: Notification) -> bool: ...


class HealthSupervisor:
    def __init__(
        self,
        checker: HealthChecker,
        runtime: RuntimeStore,
        telemetry: TelemetryRepository,
        config: OperationsConfig,
        *,
        notifications: NotificationSink | None = None,
    ) -> None:
        self._checker = checker
        self._runtime = runtime
        self._telemetry = telemetry
        self._config = config
        self._notifications = notifications
        self._failures: dict[str, int] = {}
        self._last_states: dict[str, ComponentState] = {}
        self._last_check_monotonic: float | None = None
        self._last_heartbeat_monotonic: float | None = None

    async def maintain(self, *, force: bool = False) -> HealthReport | None:
        now = time.monotonic()
        if (
            force
            or self._last_heartbeat_monotonic is None
            or now - self._last_heartbeat_monotonic
            >= self._config.heartbeat_interval_seconds
        ):
            self._telemetry.heartbeat()
            self._last_heartbeat_monotonic = now

        if (
            not force
            and self._last_check_monotonic is not None
            and now - self._last_check_monotonic < self._config.health_interval_seconds
        ):
            return None

        report = await self._checker.check_all(include_worker=True)
        self._last_check_monotonic = now
        await self._apply(report)
        self._telemetry.prune(max_rows=self._config.event_retention_rows)
        return report

    async def _apply(self, report: HealthReport) -> None:
        immediate_block = False
        backend_block = False
        degraded = False

        for component in report.components:
            previous = self._last_states.get(component.name)
            self._last_states[component.name] = component.state

            if component.state is ComponentState.UNHEALTHY:
                self._failures[component.name] = self._failures.get(component.name, 0) + 1
            else:
                self._failures[component.name] = 0

            if component.name in {"database", "storage"} and component.state is ComponentState.UNHEALTHY:
                immediate_block = True
            elif component.name in {"llm", "comfyui"} and component.state is ComponentState.UNHEALTHY:
                if self._failures[component.name] >= self._config.backend_failure_threshold:
                    backend_block = True
                else:
                    degraded = True
            elif component.name in {"discord", "worker"} and component.state is ComponentState.UNHEALTHY:
                degraded = True

            if previous is not component.state:
                severity = (
                    EventSeverity.ERROR
                    if component.state is ComponentState.UNHEALTHY
                    else EventSeverity.INFO
                )
                self._telemetry.record(
                    "health.component_changed",
                    severity,
                    {
                        "component": component.name,
                        "state": component.state.value,
                        "detail": component.detail,
                        "failures": self._failures[component.name],
                    },
                )
                if component.state is ComponentState.UNHEALTHY:
                    await self._notify_component(component.name, component.detail)

        if immediate_block or backend_block:
            self._set_health_state(AgentState.BLOCKED, "health:critical_failure")
        elif degraded:
            self._set_health_state(AgentState.DEGRADED, "health:degraded")
        else:
            self._recover_health_state()

    def _set_health_state(self, target: AgentState, reason: str) -> None:
        current = self._runtime.get_agent_state()
        if current in {AgentState.PAUSED, AgentState.STOPPING, AgentState.STOPPED}:
            return
        if current is AgentState.BLOCKED and not (
            self._runtime.get_agent_reason() or ""
        ).startswith("health:"):
            return
        if current is target:
            return
        self._runtime.set_agent_state(target, expected=current, reason=reason)

    def _recover_health_state(self) -> None:
        current = self._runtime.get_agent_state()
        if current not in {AgentState.DEGRADED, AgentState.BLOCKED}:
            return
        reason = self._runtime.get_agent_reason() or ""
        if not reason.startswith("health:"):
            return
        self._runtime.set_agent_state(
            AgentState.RUNNING,
            expected=current,
            reason="health:recovered",
        )

    async def _notify_component(self, component: str, detail: str) -> None:
        if self._notifications is None:
            return
        kind = NotificationKind.DISK if component == "storage" else NotificationKind.BACKEND
        await self._notifications.notify(
            Notification(
                kind=kind,
                title=f"Artifex {component} health alert",
                body=detail,
            )
        )
