from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from artifex.config.models import ContextConfig
from artifex.evaluation.similarity import cosine_similarity
from artifex.planner.models import (
    CharacterOption,
    ContextProvenance,
    PlanningContext,
    RecentConceptSummary,
    ResearchBriefSummary,
    ResearchEvidenceSummary,
    TrendSignalSummary,
)
from artifex.planner.repository import ConceptRepository


class TextEmbeddingProvider(Protocol):
    async def embed_text(self, text: str) -> Sequence[float]: ...


class ContextMemoryManager:
    """Deterministic context shortlisting, retrieval and section compaction."""

    def __init__(
        self,
        config: ContextConfig,
        concepts: ConceptRepository,
        embeddings: TextEmbeddingProvider,
    ) -> None:
        self._config = config
        self._concepts = concepts
        self._embeddings = embeddings

    def pre_research(self, context: PlanningContext) -> PlanningContext:
        characters = self._compact_characters(context.characters)
        recent = self._fit_models(
            context.recent_concepts[: self._config.max_recent_concepts],
            self._config.recent_history_tokens,
        )
        trends = self._fit_models(
            tuple(
                sorted(
                    context.trend_signals,
                    key=lambda item: (
                        -(item.strength * item.confidence * item.freshness),
                        item.id,
                    ),
                )[: self._config.max_trend_signals]
            ),
            self._config.trend_tokens,
        )
        evergreen = self._fit_strings(
            context.evergreen_prompts,
            self._config.evergreen_tokens,
        )
        operator = self._fit_strings(
            context.operator_notes,
            self._config.operator_tokens,
        )
        return context.model_copy(
            update={
                "characters": characters,
                "recent_concepts": recent,
                "trend_signals": trends,
                "evergreen_prompts": evergreen,
                "operator_notes": operator,
                "long_term_concepts": (),
                "context_provenance": None,
            }
        )

    async def finalize(self, context: PlanningContext) -> PlanningContext:
        pre = self.pre_research(context)
        long_term = await self._retrieve_long_term(pre)
        research = self._compact_research(pre.research_brief)

        omitted = {
            "characters": max(0, len(context.characters) - len(pre.characters)),
            "recent_concepts": max(
                0, len(context.recent_concepts) - len(pre.recent_concepts)
            ),
            "long_term_concepts": max(
                0,
                min(
                    self._config.long_term_candidate_limit,
                    len(self._concepts.historical_summaries(
                        limit=self._config.long_term_candidate_limit
                    )),
                )
                - len(long_term),
            ),
            "trend_signals": max(
                0, len(context.trend_signals) - len(pre.trend_signals)
            ),
            "research_items": max(
                0,
                (
                    len(pre.research_brief.items)
                    if pre.research_brief is not None
                    else 0
                )
                - (len(research.items) if research is not None else 0),
            ),
        }

        finalized = pre.model_copy(
            update={
                "long_term_concepts": long_term,
                "research_brief": research,
            }
        )
        usage = {
            "characters": _estimate_models(finalized.characters),
            "recent_concepts": _estimate_models(finalized.recent_concepts),
            "long_term_concepts": _estimate_models(finalized.long_term_concepts),
            "trend_signals": _estimate_models(finalized.trend_signals),
            "research": (
                _estimate_model(finalized.research_brief)
                if finalized.research_brief is not None
                else 0
            ),
            "evergreen": _estimate_strings(finalized.evergreen_prompts),
            "operator": _estimate_strings(finalized.operator_notes),
        }
        return finalized.model_copy(
            update={
                "context_provenance": ContextProvenance(
                    version=self._config.version,
                    estimator="conservative-json-char",
                    section_estimated_tokens=usage,
                    omitted_counts=omitted,
                    long_term_retrieval="embedding-cosine-top-k",
                )
            }
        )

    def _compact_characters(
        self,
        characters: Sequence[CharacterOption],
    ) -> tuple[CharacterOption, ...]:
        ranked = tuple(
            sorted(
                characters,
                key=lambda item: (
                    item.recent_use_penalty,
                    -item.readiness,
                    -item.historical_performance,
                    item.id,
                ),
            )[: self._config.max_characters]
        )
        fitted = self._fit_models(ranked, self._config.character_tokens)
        if fitted:
            return fitted

        first = ranked[0]
        return (
            first.model_copy(update={"notes": ()}),
        )

    async def _retrieve_long_term(
        self,
        context: PlanningContext,
    ) -> tuple[RecentConceptSummary, ...]:
        if self._config.max_long_term_concepts == 0:
            return ()

        recent_ids = {item.concept_id for item in context.recent_concepts}
        candidates = tuple(
            item
            for item in self._concepts.historical_summaries(
                limit=self._config.long_term_candidate_limit
            )
            if item.concept_id not in recent_ids
        )
        if not candidates:
            return ()

        query = self._retrieval_query(context)
        query_vector = await self._embeddings.embed_text(query)
        scored: list[tuple[float, str, RecentConceptSummary]] = []
        for item in candidates:
            vector = await self._embeddings.embed_text(_concept_text(item))
            scored.append(
                (
                    cosine_similarity(query_vector, vector),
                    item.concept_id,
                    item,
                )
            )
        ranked = tuple(
            item
            for _, _, item in sorted(
                scored,
                key=lambda entry: (-entry[0], entry[1]),
            )[: self._config.max_long_term_concepts]
        )
        return self._fit_models(ranked, self._config.long_term_tokens)

    @staticmethod
    def _retrieval_query(context: PlanningContext) -> str:
        parts = [
            "characters: "
            + ", ".join(
                f"{item.id} {item.display_name}"
                for item in context.characters
            )
        ]
        if context.research_brief is not None:
            parts.extend(context.research_brief.key_findings[:6])
        parts.extend(item.topic for item in context.trend_signals[:6])
        parts.extend(item.title for item in context.seasonal_events[:4])
        parts.extend(context.evergreen_prompts[:4])
        return " | ".join(part for part in parts if part)

    def _compact_research(
        self,
        brief: ResearchBriefSummary | None,
    ) -> ResearchBriefSummary | None:
        if brief is None:
            return None
        if self._config.max_research_items == 0:
            return brief.model_copy(
                update={
                    "evidence_ids": (),
                    "key_findings": (),
                    "items": (),
                }
            )

        item_budget = max(
            1,
            self._config.research_tokens
            // max(1, self._config.max_research_items),
        )
        compacted: list[ResearchEvidenceSummary] = []
        used = 0
        for item in brief.items[: self._config.max_research_items]:
            compact = item.model_copy(
                update={
                    "title": _truncate(item.title, max(40, item_budget // 4)),
                    "finding": _truncate(item.finding, max(80, item_budget // 2)),
                }
            )
            cost = _estimate_model(compact)
            if compacted and used + cost > self._config.research_tokens:
                break
            if not compacted and cost > self._config.research_tokens:
                compact = compact.model_copy(
                    update={
                        "title": _truncate(compact.title, 80),
                        "finding": _truncate(
                            compact.finding,
                            max(80, self._config.research_tokens // 2),
                        ),
                    }
                )
                cost = _estimate_model(compact)
            compacted.append(compact)
            used += cost

        evidence_ids = tuple(item.evidence_id for item in compacted)
        key_findings = self._fit_strings(
            brief.key_findings,
            max(0, self._config.research_tokens - used),
        )
        return brief.model_copy(
            update={
                "evidence_ids": evidence_ids,
                "key_findings": key_findings,
                "items": tuple(compacted),
            }
        )

    @staticmethod
    def _fit_models[ModelT](
        items: Sequence[ModelT],
        budget: int,
    ) -> tuple[ModelT, ...]:
        if budget <= 0:
            return ()
        fitted: list[ModelT] = []
        used = 0
        for item in items:
            cost = _estimate_model(item)
            if fitted and used + cost > budget:
                break
            if not fitted and cost > budget:
                break
            fitted.append(item)
            used += cost
        return tuple(fitted)

    @staticmethod
    def _fit_strings(
        items: Sequence[str],
        budget: int,
    ) -> tuple[str, ...]:
        if budget <= 0:
            return ()
        fitted: list[str] = []
        used = 0
        for item in items:
            remaining = budget - used
            if remaining <= 0:
                break
            compact = _truncate(item, remaining)
            if not compact:
                continue
            fitted.append(compact)
            used += len(compact)
        return tuple(fitted)


def _concept_text(item: RecentConceptSummary) -> str:
    return " | ".join(
        (
            ",".join(item.character_ids),
            item.theme,
            item.setting,
            item.visual_hook,
        )
    )


def _estimate_model(item: object) -> int:
    if hasattr(item, "model_dump"):
        payload = item.model_dump(mode="json")  # type: ignore[attr-defined]
    else:
        payload = item
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _estimate_models(items: Sequence[object]) -> int:
    return sum(_estimate_model(item) for item in items)


def _estimate_strings(items: Sequence[str]) -> int:
    return sum(len(item) for item in items)


def _truncate(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1].rstrip() + "…"
