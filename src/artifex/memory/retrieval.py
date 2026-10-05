from __future__ import annotations

from collections.abc import Sequence

from artifex.evaluation.similarity import EmbeddingProvider, cosine_similarity
from artifex.planner.models import PlanningContext, RecentConceptSummary
from artifex.planner.repository import ConceptRepository


def _summary_text(item: RecentConceptSummary) -> str:
    return " | ".join(
        (
            ",".join(item.character_ids),
            item.theme,
            item.setting,
            item.visual_hook,
        )
    )


def _query_text(context: PlanningContext) -> str:
    parts: list[str] = [
        "characters: " + ", ".join(item.display_name for item in context.characters)
    ]
    if context.research_brief is not None:
        parts.extend(context.research_brief.key_findings)
    parts.extend(signal.topic for signal in context.trend_signals)
    parts.extend(event.title for event in context.seasonal_events)
    parts.extend(context.operator_notes)
    return " | ".join(part for part in parts if part.strip())


class ConceptMemoryRetriever:
    def __init__(
        self,
        repository: ConceptRepository,
        embeddings: EmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings

    async def retrieve(
        self,
        context: PlanningContext,
        *,
        limit: int,
        candidate_limit: int,
    ) -> tuple[RecentConceptSummary, ...]:
        if limit <= 0:
            return ()

        recent_ids = {item.concept_id for item in context.recent_concepts}
        candidates = tuple(
            item
            for item in self._repository.historical_summaries(
                limit=candidate_limit
            )
            if item.concept_id not in recent_ids
        )
        if not candidates:
            return ()

        query = await self._embeddings.embed_text(_query_text(context))
        scored: list[tuple[float, str, RecentConceptSummary]] = []
        for item in candidates:
            vector = await self._embeddings.embed_text(_summary_text(item))
            scored.append(
                (
                    cosine_similarity(query, vector),
                    item.concept_id,
                    item,
                )
            )
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return tuple(item for _, _, item in scored[:limit])
