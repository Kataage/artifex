from __future__ import annotations

import math
from datetime import UTC, datetime

from artifex.characters import CharacterRegistry
from artifex.config.models import CharacterRegistryConfig
from artifex.domain import LoRAPolicy
from artifex.loras import LoRARegistry
from artifex.performance import PerformanceLearningService
from artifex.planner.models import CharacterOption, PlanningContext
from artifex.planner.repository import ConceptRepository
from artifex.trends.planner import TrendPlannerContext


class PlanningContextBuilder:
    def __init__(
        self,
        characters: CharacterRegistry,
        loras: LoRARegistry,
        concepts: ConceptRepository,
        character_config: CharacterRegistryConfig,
        *,
        trends: TrendPlannerContext | None = None,
        performance: PerformanceLearningService | None = None,
    ) -> None:
        self._characters = characters
        self._loras = loras
        self._concepts = concepts
        self._character_config = character_config
        self._trends = trends
        self._performance = performance

    def build(self, *, as_of: datetime | None = None) -> PlanningContext:
        now = as_of or datetime.now(UTC)
        options: list[CharacterOption] = []
        profiles = self._characters.list(enabled_only=True)
        performance_by_character = (
            self._performance.character_effects(
                tuple(profile.id for profile in profiles),
                as_of=now,
            )
            if self._performance is not None
            else {}
        )

        for profile in profiles:
            if profile.readiness < self._character_config.minimum_readiness:
                continue

            production_loras = self._loras.for_character(
                profile.id,
                model_family=(
                    profile.model_families[0]
                    if profile.model_families
                    else None
                ),
                production_only=True,
            )
            if profile.lora_policy is LoRAPolicy.REQUIRED and not production_loras:
                continue

            readiness = profile.readiness
            if production_loras and profile.lora_policy in {
                LoRAPolicy.REQUIRED,
                LoRAPolicy.PREFERRED,
            }:
                readiness = min(
                    readiness,
                    max(lora.readiness for lora in production_loras),
                )

            performance = performance_by_character.get(profile.id)
            options.append(
                CharacterOption(
                    id=profile.id,
                    display_name=profile.display_name,
                    namespace=profile.namespace,
                    branch=profile.branch,
                    group=profile.group,
                    readiness=readiness,
                    recent_use_penalty=self._recent_use_penalty(
                        profile.last_used_at,
                        now,
                    ),
                    historical_performance=(
                        performance.score if performance is not None else 0.5
                    ),
                    historical_performance_confidence=(
                        performance.confidence if performance is not None else 0.0
                    ),
                    historical_performance_reason=(
                        performance.reason if performance is not None
                        else "performance learning disabled"
                    ),
                    notes=profile.generation_notes,
                )
            )

        if not options:
            raise RuntimeError(
                "no production-ready characters are available for planning"
            )

        context = PlanningContext(
            as_of=now,
            characters=tuple(options),
            recent_concepts=self._concepts.recent_summaries(limit=30),
            evergreen_prompts=(
                "classic character-focused portrait or full-body illustration",
                "casual daily-life scene",
                "strong seasonal visual without relying on a current trend",
                "recognizable fan-favorite character presentation",
            ),
        )
        if self._trends is not None:
            context = self._trends.attach(context)
        return context

    @staticmethod
    def _recent_use_penalty(
        last_used_at: datetime | None,
        now: datetime,
    ) -> float:
        if last_used_at is None:
            return 0.0
        last = (
            last_used_at.replace(tzinfo=UTC)
            if last_used_at.tzinfo is None
            else last_used_at.astimezone(UTC)
        )
        current = now.astimezone(UTC)
        age_hours = max(0.0, (current - last).total_seconds() / 3600.0)
        return math.exp(-age_hours / 36.0)
