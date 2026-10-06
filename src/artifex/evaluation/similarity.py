from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from artifex.evaluation.semantic import SemanticIndex
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


@dataclass(frozen=True, slots=True)
class IdentitySimilarity:
    aggregate: float | None
    by_character: dict[str, float]
    missing_character_ids: tuple[str, ...] = ()


class SimilarityService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        index: SemanticIndex | None = None,
    ) -> None:
        self._provider = provider
        self._index = index

    async def _text_vector(self, text: str) -> Sequence[float]:
        if self._index is None:
            return await self._provider.embed_text(text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return await self._index.embed_text("text", digest, text)

    async def _image_vector(self, path: Path) -> Sequence[float]:
        if self._index is None:
            return await self._provider.embed_image(path)
        subject_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
        return await self._index.embed_image(
            "image",
            subject_id,
            path,
            metadata={"path": str(path)},
        )

    async def max_text_similarity(
        self,
        text: str,
        references: Sequence[str],
    ) -> float:
        if not references:
            return 0.0
        query = await self._text_vector(text)
        best = 0.0
        for reference in references:
            vector = await self._text_vector(reference)
            best = max(best, cosine_similarity(query, vector))
        return best

    async def max_image_similarity(
        self,
        path: Path,
        references: Sequence[Path],
    ) -> float:
        if not references:
            return 0.0
        query = await self._image_vector(path)
        best = 0.0
        for reference in references:
            vector = await self._image_vector(reference)
            best = max(best, cosine_similarity(query, vector))
        return best

    async def identity_similarity(
        self,
        path: Path,
        references: Mapping[str, Sequence[Path]],
    ) -> IdentitySimilarity:
        available = {
            character_id: tuple(item for item in paths if item.exists())
            for character_id, paths in references.items()
            if paths
        }
        missing = tuple(
            character_id
            for character_id, paths in references.items()
            if not paths
        )
        if not available:
            return IdentitySimilarity(
                aggregate=None,
                by_character={},
                missing_character_ids=missing,
            )

        query = await self._image_vector(path)
        by_character: dict[str, float] = {}
        for character_id, paths in available.items():
            best = 0.0
            for reference in paths:
                vector = await self._image_vector(reference)
                best = max(best, cosine_similarity(query, vector))
            by_character[character_id] = best
        return IdentitySimilarity(
            aggregate=min(by_character.values()) if by_character else None,
            by_character=by_character,
            missing_character_ids=missing,
        )


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
