from __future__ import annotations

from sqlalchemy import select

from artifex.config.models import ProductionConfig
from artifex.db import Database
from artifex.db.models import ConceptRow, PackRow
from artifex.domain import AgentState, PackState
from artifex.editorial import EditorialLane, EditorialService
from artifex.runtime.store import RuntimeStore
from artifex.scheduler.models import InventorySnapshot, SchedulerAction, SchedulerDecision


class Scheduler:
    def __init__(
        self,
        database: Database,
        runtime: RuntimeStore,
        production: ProductionConfig,
        *,
        editorial: EditorialService | None = None,
    ) -> None:
        self._database = database
        self._runtime = runtime
        self._production = production
        self._editorial = editorial

    def inventory(self) -> InventorySnapshot:
        if self._editorial is not None:
            counts = self._editorial.inventory_counts()
            with self._database.session() as session:
                idea_count = len(
                    session.scalars(
                        select(ConceptRow).where(ConceptRow.status == "idea")
                    ).all()
                )
                planned_count = len(
                    session.scalars(
                        select(PackRow).where(
                            PackRow.state == PackState.PLANNED.value
                        )
                    ).all()
                )
            return InventorySnapshot(
                ideas=idea_count,
                planned=planned_count,
                completed_available=counts.available,
                completed_reserved=counts.reserved,
                completed_consumed=counts.consumed,
                completed_expired=counts.expired,
            )

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

        if self._editorial is not None:
            allowed, production_reason = self._editorial.production_allowed()
            if not allowed:
                return SchedulerDecision(
                    SchedulerAction.IDLE,
                    production_reason,
                )

            if inventory.planned < self._production.planned_inventory_target:
                plan = self._editorial.decide_plan(
                    has_ideas=inventory.ideas > 0
                )
                if plan.lane is EditorialLane.SERIES:
                    if plan.series_id is None:
                        raise RuntimeError(
                            "Series editorial decision is missing series_id"
                        )
                    return SchedulerDecision(
                        SchedulerAction.PLAN_SERIES_PACK,
                        plan.reason,
                        series_id=plan.series_id,
                        editorial_decision_id=plan.decision_id,
                        series_plan_kind=(
                            plan.series_plan_kind.value
                            if plan.series_plan_kind is not None
                            else None
                        ),
                    )
                if plan.concept_id is None:
                    return SchedulerDecision(
                        SchedulerAction.REPLENISH_IDEAS,
                        plan.reason,
                    )
                return SchedulerDecision(
                    SchedulerAction.PLAN_PACK,
                    plan.reason,
                    concept_id=plan.concept_id,
                    editorial_decision_id=plan.decision_id,
                )

            pack_id = self.next_runnable_pack_id()
            if pack_id is not None:
                return SchedulerDecision(
                    SchedulerAction.RUN_PACK,
                    "runnable pack available",
                    pack_id,
                )

            if inventory.ideas < self._production.idea_inventory_target:
                return SchedulerDecision(
                    SchedulerAction.REPLENISH_IDEAS,
                    "idea inventory below target",
                )

            return SchedulerDecision(
                SchedulerAction.IDLE,
                "editorial inventory is healthy and no runnable work exists",
            )

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
