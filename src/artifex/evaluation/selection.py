from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from artifex.db import Database
from artifex.db.models import EvaluationRow
from artifex.domain import ResultState, SceneState
from artifex.evaluation.repository import EvaluationRepository
from artifex.runtime import RuntimeStore


class SelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_id: str
    state: ResultState
    attempt_id: str | None
    aggregate: float | None


class AttemptSelector:
    def __init__(
        self,
        database: Database,
        runtime: RuntimeStore,
        evaluations: EvaluationRepository,
    ) -> None:
        self._database = database
        self._runtime = runtime
        self._evaluations = evaluations

    def select(self, scene_id: str) -> SelectionResult:
        rows = self._evaluations.for_scene(scene_id)
        if not rows:
            raise ValueError(f"scene has no evaluations: {scene_id}")

        accepted = [row for row in rows if row.result_state == ResultState.ACCEPTED.value]
        review = [row for row in rows if row.result_state == ResultState.REVIEW.value]
        rejected = [row for row in rows if row.result_state == ResultState.REJECTED.value]

        if accepted:
            best = self._best(accepted)
            self._runtime.select_scene_attempt(
                scene_id,
                best.attempt_id,
                SceneState.ACCEPTED,
            )
            return self._result(scene_id, ResultState.ACCEPTED, best)

        if review:
            best = self._best(review)
            self._runtime.select_scene_attempt(
                scene_id,
                best.attempt_id,
                SceneState.REVIEW,
            )
            return self._result(scene_id, ResultState.REVIEW, best)

        best_rejected = self._best(rejected)
        self._runtime.select_scene_attempt(
            scene_id,
            None,
            SceneState.REJECTED,
        )
        return SelectionResult(
            scene_id=scene_id,
            state=ResultState.REJECTED,
            attempt_id=None,
            aggregate=self._aggregate(best_rejected),
        )

    @staticmethod
    def _aggregate(row: EvaluationRow) -> float:
        value = row.scores_json.get("aggregate", 0.0)
        return float(value)

    @classmethod
    def _best(cls, rows: list[EvaluationRow]) -> EvaluationRow:
        return max(
            rows,
            key=lambda row: (cls._aggregate(row), row.attempt_id),
        )

    @classmethod
    def _result(
        cls,
        scene_id: str,
        state: ResultState,
        row: EvaluationRow,
    ) -> SelectionResult:
        return SelectionResult(
            scene_id=scene_id,
            state=state,
            attempt_id=row.attempt_id,
            aggregate=cls._aggregate(row),
        )
