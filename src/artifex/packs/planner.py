from __future__ import annotations

import json
from typing import Any

from artifex.config.models import ContextConfig, PatreonConfig
from artifex.llm import ChatMessage, StructuredGenerator
from artifex.llm.prompts import CONTENT_PACK_PROMPT
from artifex.packs.models import (
    ContentPackPlan,
    EditorialArchetype,
    PackRecord,
)
from artifex.packs.repository import PackRepository
from artifex.planner.models import SelectedConcept
from artifex.series.models import SeriesProfile, SeriesPromptContext, SeriesStatus


class PackPlanner:
    def __init__(
        self,
        generator: StructuredGenerator,
        repository: PackRepository,
        context_config: ContextConfig | None = None,
        patreon_config: PatreonConfig | None = None,
    ) -> None:
        self._generator = generator
        self._repository = repository
        self._context = context_config or ContextConfig()
        self._patreon = patreon_config or PatreonConfig(enabled=False)

    async def plan_and_persist(
        self,
        concept: SelectedConcept,
        *,
        series: SeriesProfile | None = None,
    ) -> PackRecord:
        series_context = (
            self._series_context(series) if series is not None else None
        )
        archetype = self._archetype_for(concept)
        messages = self._messages(concept, series_context, archetype)

        def validate(plan: ContentPackPlan) -> None:
            self._validate_plan(
                plan,
                concept,
                series,
                series_context,
                archetype,
            )

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
                "idea_source": concept.candidate.idea_source.value,
                "theme": concept.candidate.theme,
                "setting": concept.candidate.setting,
                "format": concept.candidate.format.value,
                "character_ids": list(concept.candidate.character_ids),
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
        series_context: SeriesPromptContext | None,
        archetype: EditorialArchetype,
    ) -> None:
        candidate = concept.candidate
        if plan.format is not candidate.format:
            raise ValueError(
                f"pack format mismatch: expected {candidate.format.value}, "
                f"got {plan.format.value}"
            )
        if plan.character_ids != candidate.character_ids:
            raise ValueError("pack character_ids must exactly match selected concept")
        if plan.editorial_archetype is not archetype:
            raise ValueError(
                "pack editorial_archetype must match the selected editorial policy"
            )
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

        if series_context is None:
            raise ValueError("series prompt context is missing")
        if series.status is not SeriesStatus.ACTIVE:
            raise ValueError(f"series is not active: {series.status.value}")
        if plan.series_id != series.id:
            raise ValueError("pack series_id does not match series")
        if plan.episode_number != series.current_episode + 1:
            raise ValueError("pack episode_number is not the next series episode")
        expected_prior = series_context.recent_prior_pack_ids
        if plan.prior_pack_ids != expected_prior:
            raise ValueError("pack prior_pack_ids must match bounded recent series history")
        if set(plan.character_ids) - set(series.character_ids):
            raise ValueError("pack introduces characters outside series configuration")
        if series.preferred_format is not None and plan.format is not series.preferred_format:
            raise ValueError("pack format does not match series preferred_format")
        if plan.unresolved_hooks_carried != series_context.unresolved_hooks:
            raise ValueError("pack must carry the bounded unresolved series hooks")

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
        bible = _fit_strings(
            series.bible,
            max(80, self._context.series_tokens // 5),
            item_limit=180,
        )
        recent_summaries = _fit_strings(
            series.recent_episode_summaries[-5:],
            max(100, self._context.series_tokens // 4),
            item_limit=240,
        )
        rolling_summary = _truncate(
            series.rolling_summary,
            max(120, self._context.series_tokens // 4),
        )
        hooks = _fit_strings(
            series.unresolved_hooks,
            max(80, self._context.series_tokens // 6),
            item_limit=160,
        )
        context = SeriesPromptContext(
            id=series.id,
            title=_truncate(series.title, 160),
            current_episode=series.current_episode,
            character_ids=series.character_ids,
            bible=bible,
            recent_prior_pack_ids=recent,
            omitted_prior_pack_count=max(
                0, len(series.prior_pack_ids) - len(recent)
            ),
            rolling_summary=rolling_summary,
            recent_episode_summaries=recent_summaries,
            continuity_state=continuity,
            unresolved_hooks=hooks,
            preferred_format=series.preferred_format,
        )

        while (
            _estimate_context(context) > self._context.series_tokens
            and len(context.recent_episode_summaries) > 1
        ):
            context = context.model_copy(
                update={
                    "recent_episode_summaries": context.recent_episode_summaries[1:]
                }
            )
        if _estimate_context(context) > self._context.series_tokens:
            context = context.model_copy(
                update={
                    "rolling_summary": _truncate(
                        context.rolling_summary,
                        max(60, self._context.series_tokens // 8),
                    )
                }
            )
        while _estimate_context(context) > self._context.series_tokens and continuity:
            key = max(continuity)
            continuity.pop(key)
            context = context.model_copy(update={"continuity_state": dict(continuity)})
        while (
            _estimate_context(context) > self._context.series_tokens
            and len(context.unresolved_hooks) > 1
        ):
            context = context.model_copy(
                update={"unresolved_hooks": context.unresolved_hooks[1:]}
            )
        while (
            _estimate_context(context) > self._context.series_tokens
            and len(context.bible) > 1
        ):
            context = context.model_copy(update={"bible": context.bible[:-1]})

        if _estimate_context(context) > self._context.series_tokens:
            raise RuntimeError(
                "Series prompt context cannot fit configured series_tokens budget"
            )
        return context

    def _archetype_for(self, concept: SelectedConcept) -> EditorialArchetype:
        candidate = concept.candidate
        if not self._patreon.enabled:
            return EditorialArchetype.PUBLIC_ONLY
        if candidate.target_scene_count == 1:
            return EditorialArchetype.PUBLIC_ONLY
        if candidate.format.value == "mini_story":
            return EditorialArchetype.MINI_STORY
        if candidate.format.value == "variation_pack":
            return EditorialArchetype.VARIATION_PACK
        if candidate.format.value in {"seasonal", "trend"}:
            return EditorialArchetype.SEASONAL_TREND_PACK
        if candidate.format.value == "outfit_feature":
            return EditorialArchetype.SFW_COMPLETE_MEMBER_ALTERNATE
        return EditorialArchetype(self._patreon.default_archetype)

    @staticmethod
    def _messages(
        concept: SelectedConcept,
        series: SeriesPromptContext | None,
        archetype: EditorialArchetype,
    ) -> tuple[ChatMessage, ...]:
        payload = {
            "selected_concept": concept.model_dump(mode="json"),
            "series": series.model_dump(mode="json") if series is not None else None,
            "requirements": {
                "plan_entire_pack_before_generation": True,
                "scene_count": concept.candidate.target_scene_count,
                "editorial_archetype": archetype.value,
                "publication_tier_per_scene": True,
                "planned_content_rating_per_scene": True,
                "publication_tier_and_content_rating_are_independent": True,
                "scene_role_contract": _role_guidance(
                    concept.candidate.format.value
                ),
                "continuity_bible": True,
                "scene_ordinals": "contiguous starting from 1",
                "series_rule": (
                    "If series is supplied, preserve its character set constraints, "
                    "bible, rolling/recent continuity context and unresolved hooks. "
                    "Output prior_pack_ids must equal recent_prior_pack_ids exactly."
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

def _fit_strings(
    values: tuple[str, ...],
    budget: int,
    *,
    item_limit: int,
) -> tuple[str, ...]:
    if budget <= 0:
        return ()
    fitted: list[str] = []
    used = 0
    for value in reversed(values):
        compact = _truncate(value, item_limit)
        if fitted and used + len(compact) > budget:
            break
        if not fitted and len(compact) > budget:
            compact = _truncate(compact, budget)
        fitted.append(compact)
        used += len(compact)
    return tuple(reversed(fitted))


def _truncate(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit == 1:
        return value[:1]
    return value[: limit - 1].rstrip() + "…"


def _role_guidance(format_name: str) -> str:
    return {
        "single_feature": "first feature/preview; later detail/variation/alternate",
        "continuation": (
            "first continuation/preview; then development; "
            "end resolution/alternate"
        ),
        "mini_story": "opening/preview, development scene(s), final resolution",
        "variation_pack": "variation/preview/alternate only",
        "outfit_feature": (
            "outfit_reveal/preview first; then detail/variation/alternate"
        ),
        "seasonal": "seasonal_hero/preview first; then detail/variation/alternate",
        "trend": "trend_hero/preview first; then detail/variation/alternate",
        "evergreen": "feature/preview first; then detail/variation/alternate",
        "experimental": "experiment/preview only",
        "duo": (
            "feature/interaction/preview first; "
            "then interaction/detail/variation/alternate"
        ),
        "group": (
            "feature/interaction/preview first; "
            "then interaction/detail/variation/alternate"
        ),
    }[format_name]
