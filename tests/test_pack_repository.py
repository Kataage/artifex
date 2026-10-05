from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select

from artifex.db import Database
from artifex.db.models import PackRow, SceneRow
from artifex.domain import PackState, SceneState
from artifex.packs import (
    ContentPackPlan,
    PackRepository,
    ScenePlan,
    VisualSpecification,
)


def _plan() -> ContentPackPlan:
    visual = VisualSpecification(
        composition="portrait composition",
        camera="low angle",
        pose="sitting",
        expression="gentle smile",
        clothing="white dress",
        setting="open window",
        lighting="sunset rim light",
        atmosphere="airy",
    )
    return ContentPackPlan(
        format="single_feature",
        character_ids=("char-a",),
        title="Window angel",
        logline="A four-scene window sequence.",
        continuity_bible=("same white dress", "same sunset window"),
        scenes=(
            ScenePlan(
                ordinal=1,
                title="Arrival",
                purpose="establish setting",
                character_ids=("char-a",),
                visual=visual,
                publication_tier="public",
            ),
            ScenePlan(
                ordinal=2,
                title="Closer",
                purpose="develop mood",
                character_ids=("char-a",),
                continuity_constraints=("same dress",),
                visual=visual.model_copy(update={"camera": "medium shot"}),
                publication_tier="member",
            ),
        ),
    )


def test_repository_persists_entire_pack_before_generation(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'packs.sqlite3').as_posix()}")
    database.migrate()
    pack_ids = iter(["pack-1"])
    scene_ids = iter(["scene-1", "scene-2"])
    repository = PackRepository(
        database,
        pack_id_factory=lambda: next(pack_ids),
        scene_id_factory=lambda: next(scene_ids),
    )

    record = repository.create_planned_pack(
        _plan(),
        concept_id="concept-1",
        planning_provenance={"planner": "test"},
    )

    assert record.pack_id == "pack-1"
    assert record.state is PackState.PLANNED
    assert record.scene_states == (SceneState.PLANNED, SceneState.PLANNED)
    assert repository.scene_plans("pack-1") == _plan().scenes

    with database.session() as session:
        pack = session.get(PackRow, "pack-1")
        assert pack is not None
        assert pack.checkpoint_json["planning_complete"] is True
        assert pack.checkpoint_json["scene_count"] == 2
        assert session.scalar(select(func.count()).select_from(SceneRow)) == 2

    database.dispose()


def test_repository_rejects_duplicate_scene_ids_before_writing(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'atomic.sqlite3').as_posix()}")
    database.migrate()
    repository = PackRepository(
        database,
        pack_id_factory=lambda: "pack-1",
        scene_id_factory=lambda: "duplicate",
    )

    with pytest.raises(ValueError, match="duplicate"):
        repository.create_planned_pack(_plan(), concept_id="concept-1")

    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(PackRow)) == 0
        assert session.scalar(select(func.count()).select_from(SceneRow)) == 0
    database.dispose()
