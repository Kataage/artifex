from __future__ import annotations

from artifex.planner.models import PlanningContext
from artifex.trends.repository import TrendRepository


class TrendPlannerContext:
    def __init__(self, repository: TrendRepository) -> None:
        self._repository = repository

    def attach(self, context: PlanningContext) -> PlanningContext:
        signals = self._repository.active_summaries(as_of=context.as_of)
        return context.model_copy(update={"trend_signals": signals})
