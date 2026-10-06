from __future__ import annotations

from artifex.performance.service import PerformanceLearningService
from artifex.planner.models import ConceptCandidate, PlanningContext
from artifex.planner.scoring import CandidateSignals, SignalProvider


class PerformanceAwareSignalProvider:
    def __init__(
        self,
        base: SignalProvider,
        performance: PerformanceLearningService,
    ) -> None:
        self._base = base
        self._performance = performance

    async def evaluate(
        self,
        candidate: ConceptCandidate,
        context: PlanningContext,
    ) -> CandidateSignals:
        base = await self._base.evaluate(candidate, context)
        prediction = self._performance.candidate_performance(
            candidate,
            as_of=context.as_of,
        )
        return base.model_copy(
            update={
                "historical_performance": prediction.score,
                "historical_performance_confidence": prediction.confidence,
                "historical_performance_reason": prediction.reason,
            }
        )
