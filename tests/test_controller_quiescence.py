from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow, SceneRow
from artifex.domain import AgentState, PackState
from artifex.operations.quiescence import (
    _active_work,
    _queue_counts,
    quiesce_controller,
)
from artifex.runtime import RuntimeStore


class FakeQueue:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls = 0

    async def queue_snapshot(self) -> dict[str, Any]:
        value = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        assert isinstance(value, dict)
        return value


def _setup(tmp_path: Path) -> tuple[Database, RuntimeStore]:
    db = Database(f"sqlite:///{tmp_path / 'artifex.db'}")
    db.migrate()
    runtime = RuntimeStore(db)
    assert runtime.reconcile_process_start() is AgentState.STARTING
    assert runtime.set_agent_state(AgentState.RUNNING) is AgentState.RUNNING
    return db, runtime


def _idle() -> dict[str, Any]:
    return {"queue_running": [], "queue_pending": []}


def test_queue_counts_require_real_arrays_for_both_fields() -> None:
    assert _queue_counts(_idle()) == (0, 0)
    assert _queue_counts({"queue_running": [1], "queue_pending": [2, 3]}) == (1, 2)
    for invalid in (None, [], {}, {"queue_running": []},
                    {"queue_running": 0, "queue_pending": []},
                    {"queue_running": [], "queue_pending": "0"}):
        assert _queue_counts(invalid) == (None, None)


@pytest.mark.asyncio
async def test_drain_preview_does_not_pause_or_fake_safe_restart(
    tmp_path: Path,
) -> None:
    database, runtime = _setup(tmp_path)
    queue = FakeQueue([_idle()])
    report = await quiesce_controller(database, runtime, queue)  # type: ignore[arg-type]
    assert report.status == "not_paused"
    assert not report.ready
    assert not report.paused_by_command
    assert runtime.get_agent_state() is AgentState.RUNNING
    assert report.restart_authorized is False
    assert report.production_qualified is False
    assert queue.calls == 1
    database.dispose()


@pytest.mark.asyncio
async def test_drain_apply_pauses_scheduler_and_requires_two_consecutive_idle_snapshots(
    tmp_path: Path,
) -> None:
    database, runtime = _setup(tmp_path)
    queue = FakeQueue([
        {"queue_running": [[1]], "queue_pending": []},
        _idle(),
        _idle(),
    ])
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    report = await quiesce_controller(
        database, runtime, queue,  # type: ignore[arg-type]
        apply=True, wait_seconds=60, poll_seconds=1,
        sleep_fn=fake_sleep,
    )
    assert report.status == "observed_quiescent"
    assert report.ready is True and report.paused_by_command is True
    assert report.samples == 3 and report.consecutive_idle_samples == 2
    assert report.last.active_packs == 0
    assert len(sleeps) == 2
    assert runtime.get_agent_state() is AgentState.PAUSED
    assert runtime.get_agent_reason() == "controller_quiescence"
    assert "external_comfyui_submitters_remain_unfenced" in report.next_actions
    assert report.restart_authorized is False
    database.dispose()


@pytest.mark.asyncio
async def test_drain_existing_operator_pause_remains_owned_by_operator(
    tmp_path: Path,
) -> None:
    database, runtime = _setup(tmp_path)
    runtime.set_agent_state(AgentState.PAUSED, reason="discord_operator_pause")
    report = await quiesce_controller(
        database, runtime, FakeQueue([_idle()]),  # type: ignore[arg-type]
        apply=True, required_idle_samples=1,
    )
    assert report.status == "observed_quiescent"
    assert not report.paused_by_command
    assert runtime.get_agent_reason() == "discord_operator_pause"
    database.dispose()


@pytest.mark.asyncio
async def test_drain_blocks_pending_and_running_gpu_work_without_interrupting(
    tmp_path: Path,
) -> None:
    database, runtime = _setup(tmp_path)
    queue = FakeQueue([{"queue_running": [[1]], "queue_pending": [[2]]}])
    report = await quiesce_controller(
        database, runtime, queue,  # type: ignore[arg-type]
        apply=True,
    )
    assert report.status == "draining"
    assert report.last.queue_running == 1 and report.last.queue_pending == 1
    assert not report.ready and not report.restart_authorized
    assert queue.calls == 1
    database.dispose()


@pytest.mark.asyncio
async def test_drain_never_assumes_idle_if_queue_unreachable_or_malformed(
    tmp_path: Path,
) -> None:
    database, runtime = _setup(tmp_path)
    runtime.set_agent_state(AgentState.PAUSED)
    for value in ({"queue_pending": []}, {"queue_running": 0}, httpx.ConnectError("offline")):
        report = await quiesce_controller(
            database, runtime, FakeQueue([value]),  # type: ignore[arg-type]
        )
        assert report.status == "queue_unverifiable"
        assert report.ready is False
        assert report.restart_authorized is False
    database.dispose()


@pytest.mark.asyncio
async def test_drain_no_op_for_degraded_state_and_bad_args(
    tmp_path: Path,
) -> None:
    database, runtime = _setup(tmp_path)
    runtime.set_agent_state(AgentState.DEGRADED)
    with pytest.raises(ValueError, match="only running or already paused"):
        await quiesce_controller(
            database, runtime, FakeQueue([_idle()]),  # type: ignore[arg-type]
            apply=True,
        )
    assert runtime.get_agent_state() is AgentState.DEGRADED
    for options, match in (
        ({"wait_seconds": -1}, "wait_seconds"),
        ({"poll_seconds": 0}, "poll_seconds"),
        ({"required_idle_samples": 0}, "required_idle_samples"),
    ):
        with pytest.raises(ValueError, match=match):
            await quiesce_controller(
                database, runtime, FakeQueue([_idle()]),  # type: ignore[arg-type]
                **options,
            )
    database.dispose()


@pytest.mark.asyncio
async def test_drain_reports_persisted_active_pack_even_when_remote_queue_idle(
    tmp_path: Path,
) -> None:
    database, runtime = _setup(tmp_path)
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(PackRow(
            id="active-pack", state=PackState.GENERATING.value,
            format_type="single", payload_json={}, checkpoint_json={},
            created_at=now, updated_at=now,
        ))
    assert _active_work(database) == (1, 0)
    report = await quiesce_controller(
        database, runtime, FakeQueue([_idle()]),  # type: ignore[arg-type]
        apply=True, required_idle_samples=1,
    )
    assert report.status == "draining"
    assert report.last.active_packs == 1
    assert report.last.queue_pending == 0
    assert not report.ready
    database.dispose()


@pytest.mark.asyncio
async def test_drain_reports_submitted_attempt_even_when_pack_state_not_generating(
    tmp_path: Path,
) -> None:
    database, runtime = _setup(tmp_path)
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(PackRow(
            id="pack", state=PackState.PLANNED.value,
            format_type="single", payload_json={}, checkpoint_json={},
            created_at=now, updated_at=now,
        ))
        session.add(SceneRow(
            id="scene", pack_id="pack", ordinal=0, state="ready",
            publication_tier="public", payload_json={},
        ))
        session.add(GenerationAttemptRow(
            id="attempt", scene_id="scene", ordinal=1,
            backend_status="running", seed=1, prompt="test",
            negative_prompt="", provenance_json={}, created_at=now,
        ))
    assert _active_work(database) == (0, 1)
    report = await quiesce_controller(
        database, runtime, FakeQueue([_idle()]),  # type: ignore[arg-type]
        apply=True, required_idle_samples=1,
    )
    assert report.status == "draining"
    assert report.last.unfinished_attempts == 1
    database.dispose()
