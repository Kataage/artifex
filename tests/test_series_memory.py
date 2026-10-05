from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from artifex.config.models import ContextConfig
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


class CapturingClient:
    def __init__(self, output: str) -> None:
        self.output = output
        self.messages: Sequence[ChatMessage] = ()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        del response_schema, schema_name
        self.messages = messages
        return self.output


def _concept() -> SelectedConcept:
    candidate = ConceptCandidate(
        candidate_key="series",
        character_ids=("char-a",),
        idea_source="series",
        format="single_feature",
        theme="continuation",
        setting="window room",
        mood="calm",
        visual_hook="continuing feather motif",
        progression="single scene continuation",
        target_scene_count=1,
        assessment=CreativeAssessment(
            character_fit=0.9,
            novelty=0.6,
            visual_strength=0.8,
            series_potential=1.0,
        ),
    )
    return SelectedConcept(
        concept_id="concept-series",
        candidate=candidate,
        score=CandidateScore(
            aggregate=0.8,
            trend=0.0,
            evergreen=0.5,
            character_fit=0.9,
            novelty=0.6,
            visual_strength=0.8,
            historical_performance=0.5,
            seasonality=0.0,
            series_potential=1.0,
            readiness=1.0,
            similarity_penalty=0.0,
            recent_character_penalty=0.0,
        ),
    )


@pytest.mark.asyncio
async def test_long_series_keeps_full_db_history_but_sends_only_recent_ids(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'series-memory.sqlite3').as_posix()}")
    database.migrate()
    series_repo = SeriesRepository(
        database,
        id_factory=lambda: "series-1",
        rolling_summary_max_chars=240,
    )
    series_repo.create(
        title="Long Series",
        character_ids=("char-a",),
        preferred_format="single_feature",
        continuity_state={"outfit": "white dress"},
    )
    for episode in range(1, 9):
        series_repo.record_completed_pack(
            "series-1",
            pack_id=f"pack-{episode}",
            episode_number=episode,
            continuity_updates={"location": f"room-{episode}-" + ("x" * 40)},
            hooks_added=(f"hook-{episode}",),
            hooks_resolved=((f"hook-{episode - 1}",) if episode > 1 else ()),
        )

    series = series_repo.require("series-1")
    assert len(series.prior_pack_ids) == 8
    assert len(series.rolling_summary) <= 240

    output = json.dumps(
        {
            "format": "single_feature",
            "character_ids": ["char-a"],
            "title": "Episode 9",
            "logline": "Continue the motif.",
            "continuity_bible": ["white dress"],
            "scenes": [
                {
                    "ordinal": 1,
                    "title": "Continuation",
                    "purpose": "continue series",
                    "character_ids": ["char-a"],
                    "continuity_constraints": ["same outfit"],
                    "visual": {
                        "composition": "portrait",
                        "camera": "eye level",
                        "pose": "standing",
                        "expression": "soft smile",
                        "clothing": "white dress",
                        "setting": "window room",
                        "lighting": "soft",
                        "atmosphere": "calm",
                        "positive_constraints": [],
                        "negative_constraints": [],
                    },
                    "publication_tier": "public",
                    "transition_from_previous": None,
                    "series_state_updates": {},
                    "unresolved_hooks_added": [],
                    "unresolved_hooks_resolved": [],
                }
            ],
            "series_id": "series-1",
            "episode_number": 9,
            "prior_pack_ids": ["pack-6", "pack-7", "pack-8"],
            "unresolved_hooks_carried": ["hook-8"],
        }
    )
    client = CapturingClient(output)
    planner = PackPlanner(
        StructuredGenerator(client, repair_attempts=0),
        PackRepository(
            database,
            pack_id_factory=lambda: "pack-9",
            scene_id_factory=lambda: "scene-9",
        ),
        ContextConfig(series_recent_pack_ids=3, series_tokens=700),
    )

    record = await planner.plan_and_persist(_concept(), series=series)

    assert record.plan.prior_pack_ids == ("pack-6", "pack-7", "pack-8")
    user_payload = json.loads(client.messages[1].content)
    series_payload = user_payload["series"]
    assert series_payload["recent_prior_pack_ids"] == ["pack-6", "pack-7", "pack-8"]
    assert series_payload["omitted_prior_pack_count"] == 5
    assert len(json.dumps(series_payload, ensure_ascii=False)) <= 900
    database.dispose()
