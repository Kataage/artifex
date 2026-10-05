from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from artifex.comfy import ComfyExecutionResult, ComfyOutput
from artifex.db import Database
from artifex.db.models import PackRow, SceneRow
from artifex.domain import PackState, SceneState
from artifex.evaluation import GenerationAttemptRepository
from artifex.operations.recovery import RecoveryManager, RecoveryState
from artifex.runtime import RuntimeStore
from artifex.telemetry import TelemetryRepository


class FakeComfy:
    def __init__(self, history, queue=None) -> None:
        self.history = history
        self.queue = queue or {"queue_running": [], "queue_pending": []}

    async def get_history(self, prompt_id: str):
        del prompt_id
        return self.history

    async def queue_snapshot(self):
        return self.queue


def _setup(tmp_path: Path):
    database = Database(f"sqlite:///{(tmp_path / 'recovery.sqlite3').as_posix()}")
    database.migrate()
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            PackRow(
                id="pack-1",
                state=PackState.GENERATING.value,
                format_type="single_feature",
                payload_json={},
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            SceneRow(
                id="scene-1",
                pack_id="pack-1",
                ordinal=1,
                state=SceneState.GENERATING.value,
                publication_tier="public",
                payload_json={},
            )
        )
    attempts = GenerationAttemptRepository(database, id_factory=lambda: "attempt-1")
    attempt = attempts.create(
        scene_id="scene-1",
        prompt="test",
        negative_prompt="",
        seed=1,
        provenance={"comfy_prompt_id": "prompt-1"},
        backend_status="running",
    )
    return database, attempts, attempt


@pytest.mark.asyncio
async def test_recovery_reuses_completed_history_without_resubmit(tmp_path: Path) -> None:
    database, attempts, attempt = _setup(tmp_path)
    result = ComfyExecutionResult(
        prompt_id="prompt-1",
        completed=True,
        status="success",
        outputs=(
            ComfyOutput(node_id="7", filename="image.png"),
        ),
    )
    manager = RecoveryManager(
        database,
        RuntimeStore(database),
        attempts,
        FakeComfy(result),  # type: ignore[arg-type]
        TelemetryRepository(database),
    )

    recovered = await manager.recover_pack("pack-1")

    assert recovered.state is RecoveryState.RECOVERED
    assert recovered.recovered_scenes == ("scene-1",)
    assert attempts.require(attempt.id).backend_status == "completed"
    with database.session() as session:
        scene = session.get(SceneRow, "scene-1")
        assert scene is not None
        assert scene.state == SceneState.EVALUATING.value
    database.dispose()


@pytest.mark.asyncio
async def test_recovery_waits_for_prompt_still_in_queue(tmp_path: Path) -> None:
    database, attempts, _ = _setup(tmp_path)
    manager = RecoveryManager(
        database,
        RuntimeStore(database),
        attempts,
        FakeComfy(None, {"queue_running": [[1, "prompt-1"]]}),  # type: ignore[arg-type]
        TelemetryRepository(database),
    )

    recovered = await manager.recover_pack("pack-1")

    assert recovered.state is RecoveryState.WAITING
    assert recovered.waiting_scenes == ("scene-1",)
    with database.session() as session:
        scene = session.get(SceneRow, "scene-1")
        assert scene is not None
        assert scene.state == SceneState.GENERATING.value
    database.dispose()


@pytest.mark.asyncio
async def test_recovery_resubmits_only_when_history_and_queue_lost(tmp_path: Path) -> None:
    database, attempts, attempt = _setup(tmp_path)
    manager = RecoveryManager(
        database,
        RuntimeStore(database),
        attempts,
        FakeComfy(None),  # type: ignore[arg-type]
        TelemetryRepository(database),
    )

    recovered = await manager.recover_pack("pack-1")

    assert recovered.state is RecoveryState.RESUBMIT
    assert recovered.resubmit_scenes == ("scene-1",)
    assert attempts.require(attempt.id).backend_status == "lost"
    with database.session() as session:
        scene = session.get(SceneRow, "scene-1")
        assert scene is not None
        assert scene.state == SceneState.READY.value
    database.dispose()
