from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import TypeVar

from artifex.config.models import ContextConfig
from artifex.memory.retrieval import ConceptMemoryRetriever
from artifex.planner.models import (
    ContextProvenance,
    PlanningContext,
    ResearchBriefSummary,
    ResearchEvidenceSummary,
)

ItemT = TypeVar("ItemT")


class ContextMemoryManager:
    """Deterministic shortlisting, compaction and semantic long-term retrieval."""

    def __init__(
        self,
        config: ContextConfig,
        memory: ConceptMemoryRetriever,
    ) -> None:
        self._config = config
        self._memory = memory

    def pre_research(self, context: PlanningContext) -> PlanningContext:
        characters = sorted(
            context.characters,
            key=lambda item: (
                item.recent_use_penalty,
                -item.readiness,
                -item.historical_performance,
                item.id,
            ),
        )
        shortlisted = self._fit(
            characters[: self._config.max_characters],
            self._config.character_tokens,
            lambda item: item.model_dump(mode="json"),
        )
        if not shortlisted:
            first = characters[0]
            shortlisted = (first.model_copy(update={"notes": ()}),)

        recent = self._fit(
            context.recent_concepts[: self._config.max_recent_concepts],
            self._config.recent_history_tokens,
            lambda item: item.model_dump(mode="json"),
        )
        ranked_trends = tuple(
            sorted(
                context.trend_signals,
                key=lambda item: (
                    -(item.strength * item.confidence * item.freshness),
                    item.id,
                ),
            )[: self._config.max_trend_signals]
        )
        trends = self._fit(
            ranked_trends,
            self._config.trend_tokens,
            lambda item: item.model_dump(mode="json"),
        )
        evergreen = self._fit(
            context.evergreen_prompts,
            self._config.evergreen_tokens,
            str,
        )
        operator = self._fit(
            context.operator_notes,
            self._config.operator_tokens,
            str,
        )
        return context.model_copy(
            update={
                "characters": tuple(shortlisted),
                "recent_concepts": tuple(recent),
                "long_term_concepts": (),
                "trend_signals": tuple(trends),
                "evergreen_prompts": tuple(evergreen),
                "operator_notes": tuple(operator),
                "context_provenance": None,
            }
        )

    async def finalize(self, context: PlanningContext) -> PlanningContext:
        working = self.pre_research(context)
        candidate_count = self._memory.candidate_count(
            working,
            candidate_limit=self._config.long_term_candidate_limit,
        )
        long_term = await self._memory.retrieve(
            working,
            limit=self._config.max_long_term_concepts,
            candidate_limit=self._config.long_term_candidate_limit,
        )
        long_term = self._fit(
            long_term,
            self._config.long_term_tokens,
            lambda item: item.model_dump(mode="json"),
        )

        brief = self._compact_research(working.research_brief)
        compacted = working.model_copy(
            update={
                "long_term_concepts": tuple(long_term),
                "research_brief": brief,
            }
        )
        usage = {
            "characters": _estimated_tokens(
                [item.model_dump(mode="json") for item in compacted.characters]
            ),
            "recent_concepts": _estimated_tokens(
                [item.model_dump(mode="json") for item in compacted.recent_concepts]
            ),
            "long_term_concepts": _estimated_tokens(
                [item.model_dump(mode="json") for item in compacted.long_term_concepts]
            ),
            "trend_signals": _estimated_tokens(
                [item.model_dump(mode="json") for item in compacted.trend_signals]
            ),
            "research": (
                _estimated_tokens(compacted.research_brief.model_dump(mode="json"))
                if compacted.research_brief is not None
                else 0
            ),
            "evergreen": _estimated_tokens(compacted.evergreen_prompts),
            "operator": _estimated_tokens(compacted.operator_notes),
        }
        omitted = {
            "characters": max(0, len(context.characters) - len(compacted.characters)),
            "recent_concepts": max(
                0, len(context.recent_concepts) - len(compacted.recent_concepts)
            ),
            "long_term_concepts": max(0, candidate_count - len(long_term)),
            "trend_signals": max(
                0, len(context.trend_signals) - len(compacted.trend_signals)
            ),
            "research_items": max(
                0,
                (
                    len(context.research_brief.items)
                    if context.research_brief is not None
                    else 0
                )
                - (
                    len(compacted.research_brief.items)
                    if compacted.research_brief is not None
                    else 0
                ),
            ),
        }
        return compacted.model_copy(
            update={
                "context_provenance": ContextProvenance(
                    version=self._config.version,
                    estimator="conservative-json-char/2",
                    section_estimated_tokens=usage,
                    omitted_counts=omitted,
                    long_term_retrieval="embedding-cosine-top-k",
                )
            }
        )

    def _compact_research(
        self,
        brief: ResearchBriefSummary | None,
    ) -> ResearchBriefSummary | None:
        if brief is None:
            return None
        if self._config.max_research_items <= 0 or self._config.research_tokens <= 0:
            return brief.model_copy(
                update={"items": (), "evidence_ids": (), "key_findings": ()}
            )

        items_budget = max(1, int(self._config.research_tokens * 0.8))
        per_item = max(
            32,
            items_budget // max(1, self._config.max_research_items),
        )
        candidates = tuple(
            _compact_research_item(item, per_item)
            for item in brief.items[: self._config.max_research_items]
        )
        items = self._fit(
            candidates,
            items_budget,
            lambda item: item.model_dump(mode="json"),
        )
        used = _estimated_tokens(
            [item.model_dump(mode="json") for item in items]
        )
        findings_budget = max(0, self._config.research_tokens - used)
        findings = self._fit(
            brief.key_findings,
            findings_budget,
            str,
        )
        return brief.model_copy(
            update={
                "items": tuple(items),
                "evidence_ids": tuple(item.evidence_id for item in items),
                "key_findings": tuple(findings),
            }
        )

    @staticmethod
    def _fit(
        items: Sequence[ItemT],
        budget_tokens: int,
        serializer: Callable[[ItemT], object],
    ) -> tuple[ItemT, ...]:
        if budget_tokens <= 0:
            return ()
        used = 0
        result: list[ItemT] = []
        for item in items:
            tokens = _estimated_tokens(serializer(item))
            if result and used + tokens > budget_tokens:
                break
            if not result and tokens > budget_tokens:
                continue
            result.append(item)
            used += tokens
        return tuple(result)


def _compact_research_item(
    item: ResearchEvidenceSummary,
    budget_tokens: int,
) -> ResearchEvidenceSummary:
    char_budget = max(80, budget_tokens * 2)
    return item.model_copy(
        update={
            "title": _truncate(item.title, max(32, char_budget // 4)),
            "finding": _truncate(item.finding, max(48, char_budget // 2)),
        }
    )


def _truncate(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit == 1:
        return value[:1]
    return value[: limit - 1].rstrip() + "…"


def _estimated_tokens(value: object) -> int:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    # Section-level conservative estimator. The LLM client performs an exact
    # llama.cpp token-count hard gate on the complete chat payload.
    return max(1, math.ceil(len(text) / 2))
