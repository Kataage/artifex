from __future__ import annotations

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
        "characters: "
        + ", ".join(
            f"{item.id} {item.display_name}"
            for item in context.characters
        )
    ]
    if context.research_brief is not None:
        parts.extend(context.research_brief.key_findings[:8])
    parts.extend(signal.topic for signal in context.trend_signals[:8])
    parts.extend(event.title for event in context.seasonal_events[:6])
    parts.extend(context.evergreen_prompts[:6])
    parts.extend(context.operator_notes[:6])
    return " | ".join(part for part in parts if part.strip())


class ConceptMemoryRetriever:
    def __init__(
        self,
        repository: ConceptRepository,
        embeddings: EmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings

    def candidates(
        self,
        context: PlanningContext,
        *,
        candidate_limit: int,
    ) -> tuple[RecentConceptSummary, ...]:
        recent_ids = {item.concept_id for item in context.recent_concepts}
        return tuple(
            item
            for item in self._repository.historical_summaries(
                limit=candidate_limit
            )
            if item.concept_id not in recent_ids
        )

    def candidate_count(
        self,
        context: PlanningContext,
        *,
        candidate_limit: int,
    ) -> int:
        return len(
            self.candidates(
                context,
                candidate_limit=candidate_limit,
            )
        )

    async def retrieve(
        self,
        context: PlanningContext,
        *,
        limit: int,
        candidate_limit: int,
    ) -> tuple[RecentConceptSummary, ...]:
        if limit <= 0:
            return ()

        candidates = self.candidates(
            context,
            candidate_limit=candidate_limit,
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
