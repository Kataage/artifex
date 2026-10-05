from __future__ import annotations

from artifex.config.models import PlannerMixConfig
from artifex.planner.models import IdeaSource, PlanningContext


def source_quotas(
    candidate_count: int,
    mix: PlannerMixConfig,
    context: PlanningContext,
) -> dict[IdeaSource, int]:
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")

    weights: dict[IdeaSource, float] = {
        IdeaSource.EVERGREEN: mix.evergreen,
        IdeaSource.TREND: mix.trend if context.trend_signals else 0.0,
        IdeaSource.SEASONAL: mix.seasonal if context.seasonal_events else 0.0,
        IdeaSource.EXPLORATION: mix.exploration,
    }
    active = {source: weight for source, weight in weights.items() if weight > 0}
    if not active:
        active = {IdeaSource.EXPLORATION: 1.0}

    total_weight = sum(active.values())
    normalized = {source: weight / total_weight for source, weight in active.items()}
    raw = {source: candidate_count * weight for source, weight in normalized.items()}
    quotas = {source: int(value) for source, value in raw.items()}

    remaining = candidate_count - sum(quotas.values())
    order = sorted(
        active,
        key=lambda source: (raw[source] - quotas[source], normalized[source], source.value),
        reverse=True,
    )
    for source in order[:remaining]:
        quotas[source] += 1

    return {source: count for source, count in quotas.items() if count > 0}
