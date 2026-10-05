from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from artifex.config.models import PlannerConfig, PlannerMixConfig
from artifex.db import Database
from artifex.llm import ChatMessage, StructuredGenerator
from artifex.planner import ConceptRepository, IdeaDirector
from artifex.planner.director import ResearchRequiredError
from artifex.planner.models import (
    CharacterOption,
    PlanningContext,
    ResearchBriefSummary,
    ResearchEvidenceSummary,
)


class ScriptedClient:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        del messages, response_schema, schema_name
        self.calls += 1
        return self.output


def _context(*, with_research: bool) -> PlanningContext:
    now = datetime(2026, 10, 6, 12, tzinfo=UTC)
    brief = None
    if with_research:
        brief = ResearchBriefSummary(
            id="brief-1",
            topic="test",
            research_run_ids=("run-1",),
            evidence_ids=("evidence-1",),
            key_findings=("Current compositions emphasize strong silhouettes.",),
            items=(
                ResearchEvidenceSummary(
                    evidence_id="evidence-1",
                    source="web",
                    provider="fake",
                    title="Evidence",
                    finding="Current compositions emphasize strong silhouettes.",
                ),
            ),
            generated_at=now,
            expires_at=now + timedelta(hours=1),
            freshness_confidence=0.9,
            degraded=False,
            adult=False,
        )
    return PlanningContext(
        as_of=now,
        characters=(
            CharacterOption(
                id="char-a",
                display_name="Character A",
                readiness=1.0,
            ),
        ),
        research_brief=brief,
    )


@pytest.mark.asyncio
async def test_idea_director_refuses_autonomous_ideation_without_research(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'gate.sqlite3').as_posix()}")
    database.migrate()
    client = ScriptedClient('{"candidates":[]}')
    director = IdeaDirector(
        StructuredGenerator(client, repair_attempts=0),
        ConceptRepository(database),
        PlannerConfig(
            candidate_count=1,
            mix=PlannerMixConfig(
                evergreen=1.0,
                trend=0.0,
                seasonal=0.0,
                exploration=0.0,
            ),
        ),
        require_research=True,
    )

    with pytest.raises(ResearchRequiredError, match="ResearchBrief"):
        await director.create_concept(_context(with_research=False))

    assert client.calls == 0
    database.dispose()


@pytest.mark.asyncio
async def test_idea_director_accepts_exact_research_evidence_citations(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'grounded.sqlite3').as_posix()}")
    database.migrate()
    output = json.dumps(
        {
            "candidates": [{
                "candidate_key": "grounded",
                "character_ids": ["char-a"],
                "idea_source": "evergreen",
                "format": "single_feature",
                "theme": "strong silhouette portrait",
                "setting": "open rooftop",
                "mood": "confident",
                "visual_hook": "clean silhouette against sky",
                "progression": "wide to close portrait",
                "target_scene_count": 2,
                "continuity_requirements": [],
                "source_refs": [],
                "research_run_ids": ["run-1"],
                "research_evidence_ids": ["evidence-1"],
                "research_notes": ["Evidence supports silhouette-focused composition."],
                "assessment": {
                    "character_fit": 0.9,
                    "novelty": 0.8,
                    "visual_strength": 0.9,
                    "series_potential": 0.5
                }
            }]
        }
    )
    client = ScriptedClient(output)
    director = IdeaDirector(
        StructuredGenerator(client, repair_attempts=0),
        ConceptRepository(database),
        PlannerConfig(
            candidate_count=1,
            mix=PlannerMixConfig(
                evergreen=1.0,
                trend=0.0,
                seasonal=0.0,
                exploration=0.0,
            ),
        ),
        require_research=True,
    )

    selected = await director.create_concept(_context(with_research=True))

    assert selected.candidate.research_run_ids == ("run-1",)
    assert selected.candidate.research_evidence_ids == ("evidence-1",)
    assert client.calls == 1
    database.dispose()
