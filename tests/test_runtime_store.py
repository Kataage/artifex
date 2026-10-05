from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import PackRow, SceneRow
from artifex.domain import AgentState, PackState, SceneState
from artifex.runtime import RuntimeStore


def _database(tmp_path: Path) -> Database:
    db = Database(f"sqlite:///{(tmp_path / 'runtime.sqlite3').as_posix()}")
    db.migrate()
    return db


def test_agent_state_is_persisted_and_reconciled(tmp_path: Path) -> None:
    db = _database(tmp_path)
    store = RuntimeStore(db)

    assert store.get_agent_state() is AgentState.STOPPED
    assert store.set_agent_state(
        AgentState.STARTING,
        expected=AgentState.STOPPED,
    ) is AgentState.STARTING
    store.set_agent_state(AgentState.RUNNING, expected=AgentState.STARTING)

    assert RuntimeStore(db).get_agent_state() is AgentState.RUNNING
    assert store.reconcile_process_start() is AgentState.STARTING
    assert store.get_agent_state() is AgentState.STARTING
    db.dispose()


def test_pack_and_scene_transitions_write_checkpoints(tmp_path: Path) -> None:
    db = _database(tmp_path)
    now = datetime.now(UTC)
    with db.session() as session:
        session.add(
            PackRow(
                id="pack-1",
                state=PackState.PLANNED.value,
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
                state=SceneState.PLANNED.value,
                publication_tier="public",
                payload_json={},
            )
        )

    store = RuntimeStore(db)
    store.transition_pack(
        "pack-1",
        PackState.POLICY_CHECK,
        checkpoint_patch={"step": "policy"},
    )
    store.transition_scene(
        "scene-1",
        SceneState.READY,
        payload_patch={"compiled": True},
    )

    with db.session() as session:
        pack = session.get(PackRow, "pack-1")
        scene = session.get(SceneRow, "scene-1")
        assert pack is not None
        assert scene is not None
        assert pack.state == PackState.POLICY_CHECK.value
        assert pack.checkpoint_json["step"] == "policy"
        assert scene.state == SceneState.READY.value
        assert scene.payload_json["compiled"] is True

    db.dispose()


def test_recoverable_pack_order_is_deterministic(tmp_path: Path) -> None:
    db = _database(tmp_path)
    now = datetime.now(UTC)
    with db.session() as session:
        for pack_id in ("pack-b", "pack-a"):
            session.add(
                PackRow(
                    id=pack_id,
                    state=PackState.GENERATING.value,
                    format_type="single_feature",
                    payload_json={},
                    checkpoint_json={},
                    created_at=now,
                    updated_at=now,
                )
            )

    store = RuntimeStore(db)
    assert store.recoverable_pack_ids() == ("pack-a", "pack-b")

    with db.session() as session:
        ids = session.scalars(select(PackRow.id).order_by(PackRow.id)).all()
        assert list(ids) == ["pack-a", "pack-b"]
    db.dispose()
