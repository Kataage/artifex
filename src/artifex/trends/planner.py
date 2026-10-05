from __future__ import annotations

from artifex.config.models import TrendConfig
from artifex.planner.models import PlanningContext
from artifex.trends.repository import TrendRepository


class TrendPlannerContext:
    def __init__(self, repository: TrendRepository, config: TrendConfig) -> None:
        self._repository = repository
        self._config = config

    def attach(self, context: PlanningContext) -> PlanningContext:
        signals = (
            self._repository.active_summaries(as_of=context.as_of)
            if self._config.enabled
            else ()
        )
        return context.model_copy(update={"trend_signals": signals})
