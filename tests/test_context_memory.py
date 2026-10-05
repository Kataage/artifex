from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.config.models import ContextConfig
from artifex.db import Database
from artifex.db.models import ConceptRow
from artifex.evaluation.local_similarity import LocalSimilarityEmbeddingProvider
from artifex.memory import ConceptMemoryRetriever, ContextMemoryManager
from artifex.planner import ConceptRepository
from artifex.planner.models import (
    CharacterOption,
    PlanningContext,
    RecentConceptSummary,
    ResearchBriefSummary,
    ResearchEvidenceSummary,
)


@pytest.mark.asyncio
async def test_context_manager_shortlists_compacts_and_retrieves_finalized_history(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'context.sqlite3').as_posix()}")
    database.migrate()
    now = datetime(2026, 10, 6, 1, tzinfo=UTC)

    with database.session() as session:
        for index, status in enumerate(("finalized", "planned", "candidate"), start=1):
            session.add(
                ConceptRow(
                    id=f"history-{index}",
                    status=status,
                    payload_json={
                        "selected": status != "candidate",
                        "candidate": {
                            "character_ids": ["char-1"],
                            "theme": f"historical theme {index}",
                            "setting": f"historical setting {index}",
                            "visual_hook": f"historical hook {index}",
                        },
                    },
                    score=0.8,
                    similarity_score=0.1,
                    created_at=now - timedelta(days=90 - index),
                )
            )

    research_items = tuple(
        ResearchEvidenceSummary(
            evidence_id=f"ev-{index}",
            source="web",
            provider="fake",
            title=f"evidence {index}",
            finding="composition reference " + ("x" * 100),
        )
        for index in range(5)
    )
    context = PlanningContext(
        as_of=now,
        characters=tuple(
            CharacterOption(
                id=f"char-{index}",
                display_name=f"Character {index}",
                readiness=1.0 - (index * 0.01),
                recent_use_penalty=index * 0.02,
                notes=("n" * 80,),
            )
            for index in range(1, 11)
        ),
        recent_concepts=(
            RecentConceptSummary(
                concept_id="recent-1",
                character_ids=("char-1",),
                theme="recent theme",
                setting="recent setting",
                visual_hook="recent hook",
            ),
        ),
        evergreen_prompts=("portrait", "daily life", "seasonal"),
        research_brief=ResearchBriefSummary(
            id="brief-1",
            topic="test",
            research_run_ids=("run-1",),
            evidence_ids=tuple(item.evidence_id for item in research_items),
            key_findings=("current composition signal " + ("y" * 120),),
            items=research_items,
            generated_at=now,
            expires_at=now + timedelta(hours=1),
            freshness_confidence=0.9,
            degraded=False,
            adult=False,
        ),
    )

    manager = ContextMemoryManager(
        ContextConfig(
            max_characters=3,
            max_recent_concepts=1,
            max_long_term_concepts=2,
            max_research_items=2,
            character_tokens=900,
            research_tokens=700,
            recent_history_tokens=400,
            long_term_tokens=600,
            trend_tokens=200,
            evergreen_tokens=100,
            operator_tokens=50,
            long_term_candidate_limit=20,
        ),
        ConceptMemoryRetriever(
            ConceptRepository(database),
            LocalSimilarityEmbeddingProvider(),
        ),
    )

    pre = manager.pre_research(context)
    final = await manager.finalize(pre)

    assert len(pre.characters) <= 3
    assert len(final.long_term_concepts) == 2
    assert {item.concept_id for item in final.long_term_concepts} == {
        "history-1",
        "history-2",
    }
    assert final.research_brief is not None
    assert len(final.research_brief.items) <= 2
    assert set(final.research_brief.evidence_ids) == {
        item.evidence_id for item in final.research_brief.items
    }
    assert final.context_provenance is not None
    assert final.context_provenance.version == "context-v1"
    assert final.context_provenance.omitted_counts["characters"] >= 7
    assert final.context_provenance.long_term_retrieval == "embedding-cosine-top-k"
    database.dispose()


@pytest.mark.asyncio
async def test_context_compaction_is_deterministic(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'deterministic.sqlite3').as_posix()}")
    database.migrate()
    manager = ContextMemoryManager(
        ContextConfig(max_characters=2, character_tokens=500),
        ConceptMemoryRetriever(
            ConceptRepository(database),
            LocalSimilarityEmbeddingProvider(),
        ),
    )
    now = datetime(2026, 10, 6, 1, tzinfo=UTC)
    context = PlanningContext(
        as_of=now,
        characters=(
            CharacterOption(
                id="b",
                display_name="B",
                readiness=0.9,
                recent_use_penalty=0.2,
            ),
            CharacterOption(
                id="a",
                display_name="A",
                readiness=0.9,
                recent_use_penalty=0.1,
            ),
            CharacterOption(
                id="c",
                display_name="C",
                readiness=1.0,
                recent_use_penalty=0.3,
            ),
        ),
    )

    first = await manager.finalize(context)
    second = await manager.finalize(context)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert tuple(item.id for item in first.characters) == ("a", "b")
    database.dispose()
