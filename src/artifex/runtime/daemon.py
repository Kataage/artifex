from __future__ import annotations

import asyncio
from typing import Protocol

from artifex.config.models import AgentConfig
from artifex.domain import AgentState
from artifex.runtime.store import RuntimeStore
from artifex.scheduler import Scheduler, SchedulerAction, SchedulerDecision


class RuntimeHandler(Protocol):
    async def replenish_ideas(self) -> None: ...

    async def plan_pack(self) -> None: ...

    async def run_pack(self, pack_id: str) -> None: ...

    async def recover_pack(self, pack_id: str) -> None: ...


class RuntimeDaemon:
    def __init__(
        self,
        runtime: RuntimeStore,
        scheduler: Scheduler,
        handler: RuntimeHandler,
        config: AgentConfig,
    ) -> None:
        self._runtime = runtime
        self._scheduler = scheduler
        self._handler = handler
        self._config = config
        self._stop_requested = False

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
            await self._handler.plan_pack()
        elif decision.action is SchedulerAction.RUN_PACK:
            if decision.pack_id is None:
                raise RuntimeError("RUN_PACK decision missing pack_id")
            await self._handler.run_pack(decision.pack_id)
        elif decision.action is SchedulerAction.RECOVER_PACK:
            if decision.pack_id is None:
                raise RuntimeError("RECOVER_PACK decision missing pack_id")
            await self._handler.recover_pack(decision.pack_id)
        return decision

    async def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop_requested:
                await self.run_once()
                await asyncio.sleep(self._config.poll_interval_seconds)
        finally:
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
