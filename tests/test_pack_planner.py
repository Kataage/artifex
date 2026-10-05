from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from artifex.db import Database
from artifex.llm import ChatMessage, StructuredGenerator
from artifex.packs import PackPlanner, PackRepository
from artifex.planner.models import (
    CandidateScore,
    ConceptCandidate,
    CreativeAssessment,
    SelectedConcept,
)
from artifex.series import SeriesRepository


class ScriptedClient:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        del messages, response_schema, schema_name
        output = self.outputs[self.calls]
        self.calls += 1
        return output


def _selected_concept(*, format_name: str = "single_feature") -> SelectedConcept:
    candidate = ConceptCandidate(
        candidate_key="candidate-a",
        character_ids=("char-a",),
        idea_source="evergreen",
        format=format_name,
        theme="angel at an open window",
        setting="sunset room",
        mood="gentle",
        visual_hook="half outside an open window with sunset light",
        progression="wide establishment to intimate close-up",
        target_scene_count=2,
        assessment=CreativeAssessment(
            character_fit=0.9,
            novelty=0.8,
            visual_strength=0.9,
            series_potential=0.7,
        ),
    )
    return SelectedConcept(
        concept_id="concept-1",
        candidate=candidate,
        score=CandidateScore(
            aggregate=0.8,
            trend=0.0,
            evergreen=1.0,
            character_fit=0.9,
            novelty=0.8,
            visual_strength=0.9,
            historical_performance=0.5,
            seasonality=0.0,
            series_potential=0.7,
            readiness=0.9,
            similarity_penalty=0.0,
            recent_character_penalty=0.0,
        ),
    )


def _scene(ordinal: int) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "title": f"Scene {ordinal}",
        "purpose": "advance the sequence",
        "character_ids": ["char-a"],
        "continuity_constraints": ["same outfit"],
        "visual": {
            "composition": "portrait",
            "camera": "eye level",
            "pose": "sitting on window frame",
            "expression": "soft smile",
            "clothing": "white dress",
            "setting": "open window",
            "lighting": "sunset light",
            "atmosphere": "airy",
            "positive_constraints": [],
            "negative_constraints": [],
        },
        "publication_tier": "public",
        "transition_from_previous": None if ordinal == 1 else "closer continuation",
        "series_state_updates": {},
        "unresolved_hooks_added": [],
        "unresolved_hooks_resolved": [],
    }


def _plan(*, scene_count: int = 2) -> str:
    return json.dumps(
        {
            "format": "single_feature",
            "character_ids": ["char-a"],
            "title": "Window Angel",
            "logline": "A sunset window sequence.",
            "continuity_bible": ["same white dress", "same window"],
            "scenes": [_scene(index) for index in range(1, scene_count + 1)],
            "series_id": None,
            "episode_number": None,
            "prior_pack_ids": [],
            "unresolved_hooks_carried": [],
        }
    )


@pytest.mark.asyncio
async def test_pack_planner_repairs_incomplete_plan_then_persists(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'planner.sqlite3').as_posix()}")
    database.migrate()
    client = ScriptedClient([_plan(scene_count=1), _plan(scene_count=2)])
    repository = PackRepository(
        database,
        pack_id_factory=lambda: "pack-1",
        scene_id_factory=iter(["scene-1", "scene-2"]).__next__,
    )
    planner = PackPlanner(
        StructuredGenerator(client, repair_attempts=1),
        repository,
    )

    record = await planner.plan_and_persist(_selected_concept())

    assert client.calls == 2
    assert record.pack_id == "pack-1"
    assert len(record.plan.scenes) == 2
    assert record.plan.continuity_bible == ("same white dress", "same window")
    database.dispose()


@pytest.mark.asyncio
async def test_series_plan_must_carry_history_and_next_episode(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'series-plan.sqlite3').as_posix()}")
    database.migrate()
    series_repo = SeriesRepository(database, id_factory=lambda: "series-1")
    series_repo.create(
        title="Window Angel",
        character_ids=("char-a",),
        preferred_format="single_feature",
        unresolved_hooks=("feather",),
    )
    series = series_repo.record_completed_pack(
        "series-1",
        pack_id="pack-0",
        episode_number=1,
        continuity_updates={"outfit": "white dress"},
    )

    payload = json.loads(_plan())
    payload.update(
        {
            "series_id": "series-1",
            "episode_number": 2,
            "prior_pack_ids": ["pack-0"],
            "unresolved_hooks_carried": ["feather"],
        }
    )
    client = ScriptedClient([json.dumps(payload)])
    planner = PackPlanner(
        StructuredGenerator(client, repair_attempts=0),
        PackRepository(
            database,
            pack_id_factory=lambda: "pack-1",
            scene_id_factory=iter(["scene-1", "scene-2"]).__next__,
        ),
    )

    record = await planner.plan_and_persist(_selected_concept(), series=series)

    assert record.series_id == "series-1"
    assert record.plan.episode_number == 2
    assert record.plan.prior_pack_ids == ("pack-0",)
    assert record.plan.unresolved_hooks_carried == ("feather",)
    database.dispose()
