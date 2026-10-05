from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from artifex.config.models import ProductionConfig
from artifex.db import Database
from artifex.db.models import ConceptRow, PackRow
from artifex.domain import AgentState, PackState
from artifex.runtime import RuntimeStore
from artifex.scheduler import Scheduler, SchedulerAction


def _setup(tmp_path: Path) -> tuple[Database, RuntimeStore]:
    db = Database(f"sqlite:///{(tmp_path / 'scheduler.sqlite3').as_posix()}")
    db.migrate()
    runtime = RuntimeStore(db)
    runtime.set_agent_state(AgentState.STARTING, expected=AgentState.STOPPED)
    runtime.set_agent_state(AgentState.RUNNING, expected=AgentState.STARTING)
    return db, runtime


def _concept(concept_id: str, now: datetime) -> ConceptRow:
    return ConceptRow(
        id=concept_id,
        status="idea",
        payload_json={},
        created_at=now,
    )


def _pack(
    pack_id: str,
    state: PackState,
    now: datetime,
    *,
    payload: dict[str, object] | None = None,
) -> PackRow:
    return PackRow(
        id=pack_id,
        state=state.value,
        format_type="single_feature",
        payload_json=payload or {},
        checkpoint_json={},
        created_at=now,
        updated_at=now,
    )


def test_scheduler_recovers_inflight_before_refilling_inventory(tmp_path: Path) -> None:
    db, runtime = _setup(tmp_path)
    now = datetime.now(UTC)
    with db.session() as session:
        session.add(_pack("recover-me", PackState.GENERATING, now))

    scheduler = Scheduler(db, runtime, ProductionConfig())
    decision = scheduler.decide()

    assert decision.action is SchedulerAction.RECOVER_PACK
    assert decision.pack_id == "recover-me"
    db.dispose()


def test_scheduler_stops_expensive_work_at_completed_target(tmp_path: Path) -> None:
    db, runtime = _setup(tmp_path)
    now = datetime.now(UTC)
    with db.session() as session:
        session.add(_pack("done", PackState.FINALIZED, now))
        session.add(_pack("old-done", PackState.FINALIZED, now, payload={"inventory_consumed": True}))

    scheduler = Scheduler(
        db,
        runtime,
        ProductionConfig(
            idea_inventory_target=0,
            planned_inventory_target=0,
            completed_inventory_target=1,
        ),
    )
    decision = scheduler.decide()

    assert decision.action is SchedulerAction.IDLE
    assert decision.reason == "completed inventory target reached"
    db.dispose()


def test_scheduler_refills_ideas_then_plans_then_runs(tmp_path: Path) -> None:
    db, runtime = _setup(tmp_path)
    now = datetime.now(UTC)
    production = ProductionConfig(
        idea_inventory_target=1,
        planned_inventory_target=1,
        completed_inventory_target=None,
    )
    scheduler = Scheduler(db, runtime, production)

    assert scheduler.decide().action is SchedulerAction.REPLENISH_IDEAS

    with db.session() as session:
        session.add(_concept("idea-1", now))

    assert scheduler.decide().action is SchedulerAction.PLAN_PACK

    with db.session() as session:
        session.add(_pack("pack-1", PackState.PLANNED, now))

    decision = scheduler.decide()
    assert decision.action is SchedulerAction.RUN_PACK
    assert decision.pack_id == "pack-1"
    db.dispose()


def test_paused_agent_never_schedules_work(tmp_path: Path) -> None:
    db, runtime = _setup(tmp_path)
    runtime.set_agent_state(AgentState.PAUSED, expected=AgentState.RUNNING)

    decision = Scheduler(db, runtime, ProductionConfig()).decide()

    assert decision.action is SchedulerAction.IDLE
    assert "paused" in decision.reason
    db.dispose()
