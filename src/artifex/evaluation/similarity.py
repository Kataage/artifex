from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from artifex.planner.models import ConceptCandidate, PlanningContext, RecentConceptSummary
from artifex.planner.scoring import CandidateSignals, SignalProvider


class EmbeddingProvider(Protocol):
    async def embed_text(self, text: str) -> Sequence[float]: ...

    async def embed_image(self, path: Path) -> Sequence[float]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    if not left:
        raise ValueError("embeddings must not be empty")

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("zero-norm embeddings are invalid")
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


class SimilarityService:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    async def max_text_similarity(
        self,
        text: str,
        references: Sequence[str],
    ) -> float:
        if not references:
            return 0.0
        query = await self._provider.embed_text(text)
        best = 0.0
        for reference in references:
            vector = await self._provider.embed_text(reference)
            best = max(best, cosine_similarity(query, vector))
        return best

    async def max_image_similarity(
        self,
        path: Path,
        references: Sequence[Path],
    ) -> float:
        if not references:
            return 0.0
        query = await self._provider.embed_image(path)
        best = 0.0
        for reference in references:
            vector = await self._provider.embed_image(reference)
            best = max(best, cosine_similarity(query, vector))
        return best


def _candidate_text(candidate: ConceptCandidate) -> str:
    return " | ".join(
        (
            ",".join(candidate.character_ids),
            candidate.format.value,
            candidate.theme,
            candidate.setting,
            candidate.mood,
            candidate.visual_hook,
            candidate.progression,
        )
    )


def _recent_text(recent: RecentConceptSummary) -> str:
    return " | ".join(
        (
            ",".join(recent.character_ids),
            recent.theme,
            recent.setting,
            recent.visual_hook,
        )
    )


class SimilarityAwareSignalProvider:
    def __init__(
        self,
        base: SignalProvider,
        similarity: SimilarityService,
    ) -> None:
        self._base = base
        self._similarity = similarity

    async def evaluate(
        self,
        candidate: ConceptCandidate,
        context: PlanningContext,
    ) -> CandidateSignals:
        base = await self._base.evaluate(candidate, context)
        similarity_penalty = await self._similarity.max_text_similarity(
            _candidate_text(candidate),
            tuple(
                _recent_text(item)
                for item in (
                    *context.recent_concepts,
                    *context.long_term_concepts,
                )
            ),
        )
        data = base.model_dump()
        data["similarity_penalty"] = similarity_penalty
        return CandidateSignals.model_validate(data)
