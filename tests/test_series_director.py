from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from artifex.config.models import EditorialConfig, PlannerConfig
from artifex.db import Database
from artifex.editorial import SeriesPlanKind
from artifex.llm import ChatMessage, StructuredGenerator
from artifex.planner import ConceptRepository, SeriesIdeaDirector
from artifex.planner.models import (
    CharacterOption,
    IdeaSource,
    PlanningContext,
    ResearchBriefSummary,
    ResearchEvidenceSummary,
)
from artifex.series import SeriesRepository


class FakeClient:
    def __init__(self, output: dict[str, object]) -> None:
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
        return json.dumps(self.output)


@pytest.mark.asyncio
async def test_series_director_requires_exact_series_shape_and_research(
    tmp_path: Path,
) -> None:
    db = Database(f"sqlite:///{(tmp_path / 'series-director.sqlite3').as_posix()}")
    db.migrate()
    series = SeriesRepository(db).create(
        title="Window Arc",
        character_ids=("char-a",),
        bible=("white dress remains consistent",),
        series_id="series-1",
    )
    now = datetime.now(UTC)
    brief = ResearchBriefSummary(
        id="brief-1",
        topic="Series Window Arc",
        research_run_ids=("run-1",),
        evidence_ids=("evidence-1",),
        key_findings=("backlit window compositions are visually strong",),
        items=(
            ResearchEvidenceSummary(
                evidence_id="evidence-1",
                source="web",
                provider="fake",
                title="window composition",
                finding="use asymmetric sunset backlight",
            ),
        ),
        generated_at=now,
        expires_at=now + timedelta(hours=2),
        freshness_confidence=1.0,
        degraded=False,
        adult=False,
    )
    context = PlanningContext(
        as_of=now,
        characters=(
            CharacterOption(
                id="char-a",
                display_name="Character A",
                namespace="hololive",
                branch="JP",
                group="gen0",
                readiness=1.0,
            ),
        ),
        research_brief=brief,
    )
    candidates = {
        "candidates": [
            {
                "candidate_key": "ep1-a",
                "character_ids": ["char-a"],
                "idea_source": "series",
                "format": "continuation",
                "theme": "sunset window continuation",
                "setting": "quiet room by an open window",
                "mood": "calm",
                "visual_hook": "feathers catching sunset light",
                "progression": "single-scene continuation",
                "target_scene_count": 1,
                "continuity_requirements": ["white dress"],
                "source_refs": [],
                "research_run_ids": ["run-1"],
                "research_evidence_ids": ["evidence-1"],
                "research_notes": ["use backlight"],
                "assessment": {
                    "character_fit": 0.95,
                    "novelty": 0.75,
                    "visual_strength": 0.90,
                    "series_potential": 0.95
                }
            },
            {
                "candidate_key": "ep1-b",
                "character_ids": ["char-a"],
                "idea_source": "series",
                "format": "continuation",
                "theme": "morning window continuation",
                "setting": "bright room by an open window",
                "mood": "hopeful",
                "visual_hook": "soft curtain motion and feathers",
                "progression": "single-scene continuation",
                "target_scene_count": 1,
                "continuity_requirements": ["white dress"],
                "source_refs": [],
                "research_run_ids": ["run-1"],
                "research_evidence_ids": ["evidence-1"],
                "research_notes": ["use asymmetry"],
                "assessment": {
                    "character_fit": 0.90,
                    "novelty": 0.70,
                    "visual_strength": 0.80,
                    "series_potential": 0.90
                }
            }
        ]
    }
    client = FakeClient(candidates)
    repository = ConceptRepository(db)
    director = SeriesIdeaDirector(
        StructuredGenerator(client, repair_attempts=0),
        repository,
        PlannerConfig(),
        EditorialConfig(series_candidate_count=2),
        require_research=True,
    )

    selected = await director.create_concept(
        context,
        series,
        plan_kind=SeriesPlanKind.START,
    )

    assert selected.candidate.idea_source is IdeaSource.SERIES
    assert selected.candidate.character_ids == ("char-a",)
    assert selected.candidate.format.value == "continuation"
    assert repository.idea(selected.concept_id).concept_id == selected.concept_id
    assert "Series Window Arc" in client.messages[1].content
    db.dispose()
