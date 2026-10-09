from __future__ import annotations

from pathlib import Path

import pytest

from artifex.config.models import AgentConfig, ProductionConfig
from artifex.db import Database
from artifex.domain import AgentState
from artifex.runtime import RuntimeDaemon, RuntimeStore
from artifex.scheduler import Scheduler, SchedulerAction, SchedulerDecision
from artifex.telemetry import TelemetryRepository


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



@pytest.mark.asyncio
async def test_expensive_idle_maintenance_runs_only_when_scheduler_is_idle(
    tmp_path: Path,
) -> None:
    db = Database(f"sqlite:///{(tmp_path / 'idle-maintenance.sqlite3').as_posix()}")
    db.migrate()
    runtime = RuntimeStore(db)
    calls: list[str] = []

    class IdleMaintenance:
        daemon: RuntimeDaemon | None = None

        async def maintain(self) -> None:
            calls.append("idle")
            assert self.daemon is not None
            self.daemon.request_stop()

    idle = IdleMaintenance()
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
        idle_maintenance=idle,
    )
    idle.daemon = daemon

    await daemon.run_forever()

    assert calls == ["idle"]
    assert runtime.get_agent_state() is AgentState.STOPPED
    db.dispose()


@pytest.mark.asyncio
async def test_unattended_proof_is_emitted_only_by_continuous_daemon_loop(
    tmp_path: Path,
) -> None:
    db = Database(f"sqlite:///{(tmp_path / 'unattended.sqlite3').as_posix()}")
    db.migrate()
    runtime = RuntimeStore(db)
    telemetry = TelemetryRepository(db)

    class SerialScheduler:
        count = 0

        def decide(self) -> SchedulerDecision:
            if self.count >= 3:
                return SchedulerDecision(SchedulerAction.IDLE, "done")
            self.count += 1
            return SchedulerDecision(
                SchedulerAction.RUN_PACK, "real scheduler dispatch",
                f"pack-{self.count}",
            )

        def is_finalized_pack(self, pack_id: str) -> bool:
            # Production Scheduler requires persisted FINALIZED archive state.
            return pack_id != "unfinalized"

    class StopHandler(FakeHandler):
        daemon: RuntimeDaemon | None = None

        async def run_pack(self, pack_id: str) -> None:
            await super().run_pack(pack_id)
            if pack_id == "pack-3":
                assert self.daemon is not None
                self.daemon.request_stop()

    scheduler = SerialScheduler()
    handler = StopHandler()
    daemon = RuntimeDaemon(
        runtime, scheduler, handler,  # type: ignore[arg-type]
        AgentConfig(poll_interval_seconds=0.001),
        telemetry=telemetry,
    )
    handler.daemon = daemon

    # Direct operator-driven single iteration is not unattended evidence.
    await daemon.run_once()
    assert not any(
        event.event_type.startswith("daemon.")
        for event in telemetry.recent()
    )

    await daemon.run_forever()
    events = [event for event in reversed(telemetry.recent())
              if event.event_type.startswith("daemon.")]
    assert [event.event_type for event in events] == [
        "daemon.loop_started",
        "daemon.pack_finalized",
        "daemon.pack_finalized",
        "daemon.loop_stopped",
    ]
    run_ids = {event.payload.get("run_id") for event in events}
    assert len(run_ids) == 1
    run_id = next(iter(run_ids))
    assert isinstance(run_id, str) and len(run_id) == 32
    assert [e.payload["pack_id"] for e in events
            if e.event_type == "daemon.pack_finalized"] == [
        "pack-2", "pack-3",
    ]
    db.dispose()
