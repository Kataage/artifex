from __future__ import annotations

from datetime import UTC, datetime

import pytest

from artifex.planner.models import (
    CharacterOption,
    ConceptCandidate,
    CreativeAssessment,
    PlanningContext,
    TrendSignalSummary,
)
from artifex.planner.scoring import DefaultSignalProvider


@pytest.mark.asyncio
async def test_trend_score_uses_strength_confidence_and_freshness() -> None:
    context = PlanningContext(
        as_of=datetime(2026, 10, 5, tzinfo=UTC),
        characters=(CharacterOption(id="char-a", display_name="A"),),
        trend_signals=(
            TrendSignalSummary(
                id="trend-1",
                topic="topic",
                strength=0.8,
                confidence=0.5,
                freshness=0.25,
            ),
        ),
    )
    candidate = ConceptCandidate(
        candidate_key="trend",
        character_ids=("char-a",),
        idea_source="trend",
        format="trend",
        theme="theme",
        setting="setting",
        mood="mood",
        visual_hook="hook",
        progression="progression",
        target_scene_count=2,
        source_refs=("trend-1",),
        assessment=CreativeAssessment(
            character_fit=1,
            novelty=1,
            visual_strength=1,
            series_potential=0.5,
        ),
    )

    signals = await DefaultSignalProvider().evaluate(candidate, context)

    assert signals.trend == pytest.approx(0.1)
