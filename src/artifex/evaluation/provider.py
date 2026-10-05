from __future__ import annotations

from typing import Protocol

from artifex.evaluation.models import EvaluationContext, RawEvaluationSignals


class EvaluationProvider(Protocol):
    async def evaluate(self, context: EvaluationContext) -> RawEvaluationSignals: ...
