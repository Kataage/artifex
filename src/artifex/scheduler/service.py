from __future__ import annotations

from sqlalchemy import select

from artifex.config.models import ProductionConfig
from artifex.db import Database
from artifex.db.models import ConceptRow, PackRow
from artifex.domain import AgentState, PackState
from artifex.runtime.store import RuntimeStore
from artifex.scheduler.models import InventorySnapshot, SchedulerAction, SchedulerDecision


class Scheduler:
    def __init__(
        self,
        database: Database,
        runtime: RuntimeStore,
        production: ProductionConfig,
    ) -> None:
        self._database = database
        self._runtime = runtime
        self._production = production

    def inventory(self) -> InventorySnapshot:
        with self._database.session() as session:
            idea_rows = session.scalars(
                select(ConceptRow).where(ConceptRow.status == "idea")
            ).all()
            planned_rows = session.scalars(
                select(PackRow).where(PackRow.state == PackState.PLANNED.value)
            ).all()
            finalized_rows = session.scalars(
                select(PackRow).where(PackRow.state == PackState.FINALIZED.value)
            ).all()

        completed_available = sum(
            1
            for row in finalized_rows
            if not bool(row.payload_json.get("inventory_consumed", False))
        )
        return InventorySnapshot(
            ideas=len(idea_rows),
            planned=len(planned_rows),
            completed_available=completed_available,
        )

    def next_runnable_pack_id(self) -> str | None:
        with self._database.session() as session:
            return session.scalar(
                select(PackRow.id)
                .where(
                    PackRow.state.in_(
                        {
                            PackState.PLANNED.value,
                            PackState.POLICY_CHECK.value,
                        }
                    )
                )
                .order_by(PackRow.created_at.asc(), PackRow.id.asc())
                .limit(1)
            )

    def decide(self) -> SchedulerDecision:
        state = self._runtime.get_agent_state()
        if state is not AgentState.RUNNING:
            return SchedulerDecision(
                SchedulerAction.IDLE,
                f"agent is {state.value}",
            )

        recoverable = self._runtime.recoverable_pack_ids()
        if recoverable:
            return SchedulerDecision(
                SchedulerAction.RECOVER_PACK,
                "persisted in-flight pack requires recovery",
                recoverable[0],
            )

        inventory = self.inventory()
        completed_target = self._production.completed_inventory_target
        if (
            completed_target is not None
            and inventory.completed_available >= completed_target
        ):
            return SchedulerDecision(
                SchedulerAction.IDLE,
                "completed inventory target reached",
            )

        if inventory.ideas < self._production.idea_inventory_target:
            return SchedulerDecision(
                SchedulerAction.REPLENISH_IDEAS,
                "idea inventory below target",
            )

        if inventory.planned < self._production.planned_inventory_target:
            return SchedulerDecision(
                SchedulerAction.PLAN_PACK,
                "planned pack inventory below target",
            )

        pack_id = self.next_runnable_pack_id()
        if pack_id is not None:
            return SchedulerDecision(
                SchedulerAction.RUN_PACK,
                "runnable pack available",
                pack_id,
            )

        return SchedulerDecision(SchedulerAction.IDLE, "no runnable work")
