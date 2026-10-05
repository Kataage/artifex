from __future__ import annotations

import json

from artifex.llm import ChatMessage, StructuredGenerator
from artifex.packs.models import ContentPackPlan, PackRecord
from artifex.packs.repository import PackRepository
from artifex.planner.models import SelectedConcept
from artifex.series.models import SeriesProfile, SeriesStatus


class PackPlanner:
    def __init__(
        self,
        generator: StructuredGenerator,
        repository: PackRepository,
    ) -> None:
        self._generator = generator
        self._repository = repository

    async def plan_and_persist(
        self,
        concept: SelectedConcept,
        *,
        series: SeriesProfile | None = None,
    ) -> PackRecord:
        messages = self._messages(concept, series)

        def validate(plan: ContentPackPlan) -> None:
            self._validate_plan(plan, concept, series)

        plan = await self._generator.generate(
            ContentPackPlan,
            messages,
            schema_name="artifex_content_pack_plan",
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
            },
        )

    @staticmethod
    def _validate_plan(
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
        if plan.prior_pack_ids != series.prior_pack_ids:
            raise ValueError("pack prior_pack_ids must exactly match series history")
        if set(plan.character_ids) - set(series.character_ids):
            raise ValueError("pack introduces characters outside series configuration")
        if series.preferred_format is not None and plan.format is not series.preferred_format:
            raise ValueError("pack format does not match series preferred_format")
        if plan.unresolved_hooks_carried != series.unresolved_hooks:
            raise ValueError("pack must carry the current unresolved series hooks")

    @staticmethod
    def _messages(
        concept: SelectedConcept,
        series: SeriesProfile | None,
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
                content=(
                    "You are the Content Pack planner for Artifex. Convert the already "
                    "selected concept into a complete multi-scene production plan before "
                    "any image is generated. Preserve character identity, outfit/state "
                    "continuity where intended, while making each scene visually useful "
                    "and distinct. Do not write Danbooru tags or ComfyUI graphs. Describe "
                    "structured visual intent. Publication tier is intent only and may be "
                    "reclassified by policy/evaluation later. Return only schema-valid JSON."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(payload, ensure_ascii=False),
            ),
        )
