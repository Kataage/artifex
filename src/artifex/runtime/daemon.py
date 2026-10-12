from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Protocol
from uuid import uuid4

from artifex.config.models import AgentConfig
from artifex.domain import AgentState
from artifex.runtime.store import RuntimeStore
from artifex.scheduler import Scheduler, SchedulerAction, SchedulerDecision
from artifex.telemetry import EventSeverity, TelemetryRepository


class RuntimeMaintenance(Protocol):
    async def maintain(self) -> object: ...


class RuntimeHandler(Protocol):
    async def replenish_ideas(self) -> None: ...

    async def plan_pack(
        self,
        concept_id: str | None = None,
        editorial_decision_id: str | None = None,
    ) -> None: ...

    async def plan_series_pack(
        self,
        series_id: str,
        editorial_decision_id: str | None = None,
        series_plan_kind: str | None = None,
    ) -> None: ...

    async def run_pack(self, pack_id: str) -> None: ...

    async def recover_pack(self, pack_id: str) -> None: ...


class RuntimeDaemon:
    def __init__(
        self,
        runtime: RuntimeStore,
        scheduler: Scheduler,
        handler: RuntimeHandler,
        config: AgentConfig,
        *,
        maintenance: RuntimeMaintenance | None = None,
        idle_maintenance: RuntimeMaintenance | None = None,
        telemetry: TelemetryRepository | None = None,
        heartbeat_interval_seconds: float | None = None,
    ) -> None:
        self._runtime = runtime
        self._scheduler = scheduler
        self._handler = handler
        self._config = config
        self._maintenance = maintenance
        self._idle_maintenance = idle_maintenance
        self._stop_requested = False
        self._telemetry = telemetry
        if heartbeat_interval_seconds is not None and heartbeat_interval_seconds <= 0:
            raise ValueError("continuous daemon heartbeat interval must be positive")
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        # Present only for work scheduled inside a continuous daemon loop.
        # Manual run_once calls never produce unattended production evidence.
        self._continuous_run_id: str | None = None

    def start(self) -> AgentState:
        state = self._runtime.reconcile_process_start()
        if state is AgentState.STARTING:
            return self._runtime.set_agent_state(
                AgentState.RUNNING,
                expected=AgentState.STARTING,
                reason="daemon_started",
            )
        return state

    def pause(self) -> AgentState:
        current = self._runtime.get_agent_state()
        if current is AgentState.PAUSED:
            return current
        return self._runtime.set_agent_state(
            AgentState.PAUSED,
            expected=current,
            reason="operator_pause",
        )

    def resume(self) -> AgentState:
        current = self._runtime.get_agent_state()
        if current is AgentState.RUNNING:
            return current
        return self._runtime.set_agent_state(
            AgentState.RUNNING,
            expected=current,
            reason="operator_resume",
        )

    def request_stop(self) -> None:
        self._stop_requested = True

    async def run_once(self) -> SchedulerDecision:
        decision = self._scheduler.decide()
        if decision.action is SchedulerAction.REPLENISH_IDEAS:
            await self._handler.replenish_ideas()
        elif decision.action is SchedulerAction.PLAN_PACK:
            if (
                decision.concept_id is None
                and decision.editorial_decision_id is None
            ):
                await self._handler.plan_pack()
            else:
                await self._handler.plan_pack(
                    decision.concept_id,
                    decision.editorial_decision_id,
                )
        elif decision.action is SchedulerAction.PLAN_SERIES_PACK:
            if decision.series_id is None:
                raise RuntimeError(
                    "PLAN_SERIES_PACK decision missing series_id"
                )
            await self._handler.plan_series_pack(
                decision.series_id,
                decision.editorial_decision_id,
                decision.series_plan_kind,
            )
        elif decision.action is SchedulerAction.RUN_PACK:
            if decision.pack_id is None:
                raise RuntimeError("RUN_PACK decision missing pack_id")
            await self._handler.run_pack(decision.pack_id)
            self._record_unattended_completion(decision.pack_id)
        elif decision.action is SchedulerAction.RECOVER_PACK:
            if decision.pack_id is None:
                raise RuntimeError("RECOVER_PACK decision missing pack_id")
            await self._handler.recover_pack(decision.pack_id)
            self._record_unattended_completion(decision.pack_id)
        return decision

    def _record_unattended_completion(self, pack_id: str) -> None:
        if (
            self._telemetry is None
            or self._continuous_run_id is None
            or not self._scheduler.is_finalized_pack(pack_id)
        ):
            return
        self._telemetry.record_daemon_pack_finalized_once(
            pack_id, self._continuous_run_id
        )


    async def _pulse_continuous_heartbeat(self) -> None:
        """Keep heartbeat fresh while the main loop awaits a long GPU Pack."""
        assert self._telemetry is not None
        assert self._heartbeat_interval_seconds is not None
        while True:
            self._telemetry.heartbeat()
            await asyncio.sleep(self._heartbeat_interval_seconds)

    async def run_forever(self) -> None:
        self.start()
        run_id = uuid4().hex
        self._continuous_run_id = run_id
        if self._telemetry is not None:
            self._telemetry.record(
                "daemon.loop_started", EventSeverity.INFO, {"run_id": run_id},
            )
        pulse: asyncio.Task[None] | None = None
        try:
            if self._telemetry is not None and self._heartbeat_interval_seconds is not None:
                pulse = asyncio.create_task(self._pulse_continuous_heartbeat())
            while not self._stop_requested:
                if self._maintenance is not None:
                    await self._maintenance.maintain()
                decision = await self.run_once()
                if (
                    decision.action is SchedulerAction.IDLE
                    and self._idle_maintenance is not None
                ):
                    await self._idle_maintenance.maintain()
                await asyncio.sleep(self._config.poll_interval_seconds)
        finally:
            if pulse is not None:
                pulse.cancel()
                # A failed periodic pulse must never suppress the loop-stopped
                # provenance marker. The independent soak monitor will reject
                # a stale heartbeat without restarting any process.
                with suppress(asyncio.CancelledError, Exception):
                    await pulse
            self._continuous_run_id = None
            if self._telemetry is not None:
                self._telemetry.record(
                    "daemon.loop_stopped", EventSeverity.INFO, {"run_id": run_id},
                )
            current = self._runtime.get_agent_state()
            if current is not AgentState.STOPPED:
                if current is not AgentState.STOPPING:
                    self._runtime.set_agent_state(
                        AgentState.STOPPING,
                        expected=current,
                        reason="daemon_stopping",
                    )
                self._runtime.set_agent_state(
                    AgentState.STOPPED,
                    expected=AgentState.STOPPING,
                    reason="daemon_stopped",
                )
