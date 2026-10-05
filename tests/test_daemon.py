from __future__ import annotations

from pathlib import Path

import pytest

from artifex.config.models import AgentConfig, ProductionConfig
from artifex.db import Database
from artifex.domain import AgentState
from artifex.runtime import RuntimeDaemon, RuntimeStore
from artifex.scheduler import Scheduler, SchedulerAction


class FakeHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def replenish_ideas(self) -> None:
        self.calls.append(("replenish_ideas", None))

    async def plan_pack(self) -> None:
        self.calls.append(("plan_pack", None))

    async def run_pack(self, pack_id: str) -> None:
        self.calls.append(("run_pack", pack_id))

    async def recover_pack(self, pack_id: str) -> None:
        self.calls.append(("recover_pack", pack_id))


@pytest.mark.asyncio
async def test_daemon_start_pause_resume_and_single_iteration(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{(tmp_path / 'daemon.sqlite3').as_posix()}")
    db.migrate()
    runtime = RuntimeStore(db)
    scheduler = Scheduler(
        db,
        runtime,
        ProductionConfig(
            idea_inventory_target=1,
            planned_inventory_target=0,
            completed_inventory_target=None,
        ),
    )
    handler = FakeHandler()
    daemon = RuntimeDaemon(
        runtime,
        scheduler,
        handler,
        AgentConfig(poll_interval_seconds=0.01),
    )

    assert daemon.start() is AgentState.RUNNING
    assert daemon.pause() is AgentState.PAUSED
    assert daemon.resume() is AgentState.RUNNING

    decision = await daemon.run_once()
    assert decision.action is SchedulerAction.REPLENISH_IDEAS
    assert handler.calls == [("replenish_ideas", None)]
    db.dispose()


@pytest.mark.asyncio
async def test_daemon_clean_stop_persists_stopped(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{(tmp_path / 'stop.sqlite3').as_posix()}")
    db.migrate()
    runtime = RuntimeStore(db)
    daemon = RuntimeDaemon(
        runtime,
        Scheduler(
            db,
            runtime,
            ProductionConfig(
                idea_inventory_target=0,
                planned_inventory_target=0,
                completed_inventory_target=0,
            ),
        ),
        FakeHandler(),
        AgentConfig(poll_interval_seconds=0.001),
    )
    daemon.request_stop()

    await daemon.run_forever()

    assert runtime.get_agent_state() is AgentState.STOPPED
    db.dispose()
