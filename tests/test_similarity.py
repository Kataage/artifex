from __future__ import annotations

from pathlib import Path

import pytest

from artifex.evaluation import SimilarityAwareSignalProvider, SimilarityService
from artifex.planner.models import (
    CharacterOption,
    ConceptCandidate,
    CreativeAssessment,
    PlanningContext,
    RecentConceptSummary,
)
from artifex.planner.scoring import DefaultSignalProvider


class FakeEmbeddings:
    async def embed_text(self, text: str) -> tuple[float, ...]:
        if "new theme" in text or "old theme" in text:
            return (1.0, 0.0)
        return (0.0, 1.0)

    async def embed_image(self, path: Path) -> tuple[float, ...]:
        del path
        return (1.0, 0.0)


@pytest.mark.asyncio
async def test_similarity_aware_planner_signal_penalizes_near_duplicate() -> None:
    candidate = ConceptCandidate(
        candidate_key="new",
        character_ids=("char-a",),
        idea_source="evergreen",
        format="single_feature",
        theme="new theme",
        setting="room",
        mood="calm",
        visual_hook="window",
        progression="sequence",
        target_scene_count=2,
        assessment=CreativeAssessment(
            character_fit=1,
            novelty=1,
            visual_strength=1,
            series_potential=0.5,
        ),
    )
    context = PlanningContext(
        as_of="2026-10-05T00:00:00Z",
        characters=(CharacterOption(id="char-a", display_name="A"),),
        recent_concepts=(
            RecentConceptSummary(
                concept_id="old",
                character_ids=("char-a",),
                theme="old theme",
                setting="room",
                visual_hook="window",
            ),
        ),
    )
    provider = SimilarityAwareSignalProvider(
        DefaultSignalProvider(),
        SimilarityService(FakeEmbeddings()),
    )

    signals = await provider.evaluate(candidate, context)

    assert signals.similarity_penalty == pytest.approx(1.0)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    from artifex.evaluation.similarity import cosine_similarity

    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity((1.0,), (1.0, 0.0))
