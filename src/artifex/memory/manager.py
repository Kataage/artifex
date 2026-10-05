from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import TypeVar

from artifex.config.models import ContextConfig
from artifex.memory.retrieval import ConceptMemoryRetriever
from artifex.planner.models import (
    CharacterOption,
    PlanningContext,
    RecentConceptSummary,
    ResearchEvidenceSummary,
    TrendSignalSummary,
)

ItemT = TypeVar("ItemT")


class ContextMemoryManager:
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
            shortlisted = tuple(characters[:1])

        recent = self._fit(
            context.recent_concepts[: self._config.max_recent_concepts],
            self._config.recent_history_tokens,
            lambda item: item.model_dump(mode="json"),
        )
        trends = self._fit(
            context.trend_signals[: self._config.max_trend_signals],
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
                "trend_signals": tuple(trends),
                "evergreen_prompts": tuple(evergreen),
                "operator_notes": tuple(operator),
            }
        )

    async def finalize(self, context: PlanningContext) -> PlanningContext:
        long_term = await self._memory.retrieve(
            context,
            limit=self._config.max_long_term_concepts,
            candidate_limit=self._config.long_term_candidate_limit,
        )
        long_term = self._fit(
            long_term,
            self._config.long_term_tokens,
            lambda item: item.model_dump(mode="json"),
        )

        brief = context.research_brief
        if brief is not None:
            items = self._fit(
                brief.items[: self._config.max_research_items],
                self._config.research_tokens,
                lambda item: item.model_dump(mode="json"),
            )
            allowed_ids = {item.evidence_id for item in items}
            findings = self._fit(
                tuple(
                    item.finding
                    for item in items
                    if item.evidence_id in allowed_ids
                ),
                self._config.research_tokens,
                str,
            )
            brief = brief.model_copy(
                update={
                    "items": tuple(items),
                    "evidence_ids": tuple(item.evidence_id for item in items),
                    "key_findings": tuple(findings),
                }
            )

        return context.model_copy(
            update={
                "long_term_concepts": tuple(long_term),
                "research_brief": brief,
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
    # Section-level estimator only. The LLM client performs an exact
    # llama.cpp token-count hard gate on the final chat payload.
    return max(1, math.ceil(len(text) / 2))
