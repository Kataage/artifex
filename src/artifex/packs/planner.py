from __future__ import annotations

import json
from typing import Any

from artifex.config.models import ContextConfig
from artifex.llm import ChatMessage, StructuredGenerator
from artifex.llm.prompts import CONTENT_PACK_PROMPT
from artifex.packs.models import ContentPackPlan, PackRecord
from artifex.packs.repository import PackRepository
from artifex.planner.models import SelectedConcept
from artifex.series.models import SeriesProfile, SeriesPromptContext, SeriesStatus


class PackPlanner:
    def __init__(
        self,
        generator: StructuredGenerator,
        repository: PackRepository,
        context_config: ContextConfig | None = None,
    ) -> None:
        self._generator = generator
        self._repository = repository
        self._context = context_config or ContextConfig()

    async def plan_and_persist(
        self,
        concept: SelectedConcept,
        *,
        series: SeriesProfile | None = None,
    ) -> PackRecord:
        series_context = (
            self._series_context(series) if series is not None else None
        )
        messages = self._messages(concept, series_context)

        def validate(plan: ContentPackPlan) -> None:
            self._validate_plan(plan, concept, series)

        plan = await self._generator.generate(
            ContentPackPlan,
            messages,
            schema_name=CONTENT_PACK_PROMPT.schema_name(
                "artifex_content_pack_plan"
            ),
            post_validator=validate,
        )
        return self._repository.create_planned_pack(
            plan,
            concept_id=concept.concept_id,
            planning_provenance={
                "concept_id": concept.concept_id,
                "candidate_key": concept.candidate.candidate_key,
                "series_id": series.id if series is not None else None,
                "series_episode_before_plan": (
                    series.current_episode if series is not None else None
                ),
                "series_prior_pack_count": (
                    len(series.prior_pack_ids) if series is not None else 0
                ),
                "series_context_prior_pack_ids": (
                    list(series_context.recent_prior_pack_ids)
                    if series_context is not None
                    else []
                ),
            },
        )

    def _validate_plan(
        self,
        plan: ContentPackPlan,
        concept: SelectedConcept,
        series: SeriesProfile | None,
    ) -> None:
        candidate = concept.candidate
        if plan.format is not candidate.format:
            raise ValueError(
                f"pack format mismatch: expected {candidate.format.value}, "
                f"got {plan.format.value}"
            )
        if plan.character_ids != candidate.character_ids:
            raise ValueError("pack character_ids must exactly match selected concept")
        if len(plan.scenes) != candidate.target_scene_count:
            raise ValueError(
                f"pack must contain exactly {candidate.target_scene_count} scenes"
            )

        if series is None:
            if plan.series_id is not None or plan.episode_number is not None:
                raise ValueError("non-series concept produced series metadata")
            if plan.prior_pack_ids:
                raise ValueError("non-series pack cannot claim prior series packs")
            return

        if series.status is not SeriesStatus.ACTIVE:
            raise ValueError(f"series is not active: {series.status.value}")
        if plan.series_id != series.id:
            raise ValueError("pack series_id does not match series")
        if plan.episode_number != series.current_episode + 1:
            raise ValueError("pack episode_number is not the next series episode")
        expected_prior = series.prior_pack_ids[-self._context.series_recent_pack_ids :]
        if self._context.series_recent_pack_ids == 0:
            expected_prior = ()
        if plan.prior_pack_ids != expected_prior:
            raise ValueError("pack prior_pack_ids must match bounded recent series history")
        if set(plan.character_ids) - set(series.character_ids):
            raise ValueError("pack introduces characters outside series configuration")
        if series.preferred_format is not None and plan.format is not series.preferred_format:
            raise ValueError("pack format does not match series preferred_format")
        if plan.unresolved_hooks_carried != series.unresolved_hooks:
            raise ValueError("pack must carry the current unresolved series hooks")

    def _series_context(self, series: SeriesProfile) -> SeriesPromptContext:
        recent = (
            series.prior_pack_ids[-self._context.series_recent_pack_ids :]
            if self._context.series_recent_pack_ids > 0
            else ()
        )
        continuity = {
            str(key): _compact_value(value, 240)
            for key, value in sorted(
                series.continuity_state.items(),
                key=lambda item: str(item[0]),
            )
        }
        summary_budget = max(120, self._context.series_tokens // 2)
        summary = series.rolling_summary[-summary_budget:]
        context = SeriesPromptContext(
            id=series.id,
            title=series.title,
            current_episode=series.current_episode,
            character_ids=series.character_ids,
            recent_prior_pack_ids=recent,
            omitted_prior_pack_count=max(
                0, len(series.prior_pack_ids) - len(recent)
            ),
            rolling_summary=summary,
            continuity_state=continuity,
            unresolved_hooks=series.unresolved_hooks,
            preferred_format=series.preferred_format,
        )
        while _estimate_context(context) > self._context.series_tokens and continuity:
            key = sorted(continuity)[-1]
            continuity.pop(key)
            context = context.model_copy(update={"continuity_state": dict(continuity)})
        if _estimate_context(context) > self._context.series_tokens:
            context = context.model_copy(
                update={
                    "rolling_summary": context.rolling_summary[
                        -max(80, self._context.series_tokens // 4) :
                    ]
                }
            )
        return context

    @staticmethod
    def _messages(
        concept: SelectedConcept,
        series: SeriesPromptContext | None,
    ) -> tuple[ChatMessage, ...]:
        payload = {
            "selected_concept": concept.model_dump(mode="json"),
            "series": series.model_dump(mode="json") if series is not None else None,
            "requirements": {
                "plan_entire_pack_before_generation": True,
                "scene_count": concept.candidate.target_scene_count,
                "publication_tier_per_scene": True,
                "continuity_bible": True,
                "scene_ordinals": "contiguous starting from 1",
                "series_rule": (
                    "If series is supplied, preserve its character set constraints, "
                    "history, continuity state and unresolved hooks."
                ),
            },
        }
        return (
            ChatMessage(
                role="system",
                content=CONTENT_PACK_PROMPT.text,
            ),
            ChatMessage(
                role="user",
                content=json.dumps(payload, ensure_ascii=False),
            ),
        )



def _compact_value(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _estimate_context(context: SeriesPromptContext) -> int:
    return len(
        json.dumps(
            context.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
