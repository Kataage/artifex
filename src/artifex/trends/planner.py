from __future__ import annotations

from artifex.config.models import TrendConfig
from artifex.planner.models import PlanningContext
from artifex.trends.repository import TrendRepository
from artifex.trends.seasonal_repository import SeasonalRepository


class TrendPlannerContext:
    def __init__(
        self,
        repository: TrendRepository,
        config: TrendConfig,
        *,
        seasonal: SeasonalRepository | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._seasonal = seasonal

    def attach(self, context: PlanningContext) -> PlanningContext:
        signals = (
            self._repository.active_summaries(as_of=context.as_of)
            if self._config.enabled
            else ()
        )
        seasonal = (
            self._seasonal.active_summaries(as_of=context.as_of)
            if self._config.seasonal_enabled and self._seasonal is not None
            else ()
        )
        return context.model_copy(
            update={
                "trend_signals": signals,
                "seasonal_events": seasonal,
            }
        )
