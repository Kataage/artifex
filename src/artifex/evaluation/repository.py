from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select

from artifex.db import Database
from artifex.db.models import EvaluationRow, GenerationAttemptRow
from artifex.evaluation.models import EvaluationResult


def _new_id() -> str:
    return uuid4().hex


class GenerationAttemptRepository:
    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def create(
        self,
        *,
        scene_id: str,
        prompt: str,
        negative_prompt: str,
        seed: int | None,
        provenance: Mapping[str, Any],
        parent_attempt_id: str | None = None,
        backend_status: str = "queued",
    ) -> GenerationAttemptRow:
        now = datetime.now(UTC)
        with self._database.session() as session:
            if parent_attempt_id is not None:
                parent = session.get(GenerationAttemptRow, parent_attempt_id)
                if parent is None:
                    raise KeyError(f"unknown parent attempt: {parent_attempt_id}")
                if parent.scene_id != scene_id:
                    raise ValueError("parent attempt belongs to a different scene")

            max_ordinal = session.scalar(
                select(func.max(GenerationAttemptRow.ordinal)).where(
                    GenerationAttemptRow.scene_id == scene_id
                )
            )
            ordinal = int(max_ordinal or 0) + 1
            attempt = GenerationAttemptRow(
                id=self._id_factory(),
                scene_id=scene_id,
                parent_attempt_id=parent_attempt_id,
                ordinal=ordinal,
                backend_status=backend_status,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                provenance_json=dict(provenance),
                error_json=None,
                created_at=now,
            )
            session.add(attempt)
            attempt_id = attempt.id

        return self.require(attempt_id)

    def require(self, attempt_id: str) -> GenerationAttemptRow:
        with self._database.session() as session:
            row = session.get(GenerationAttemptRow, attempt_id)
            if row is None:
                raise KeyError(f"unknown generation attempt: {attempt_id}")
            session.expunge(row)
            return row

    def list_for_scene(self, scene_id: str) -> tuple[GenerationAttemptRow, ...]:
        with self._database.session() as session:
            rows = session.scalars(
                select(GenerationAttemptRow)
                .where(GenerationAttemptRow.scene_id == scene_id)
                .order_by(GenerationAttemptRow.ordinal.asc())
            ).all()
            for row in rows:
                session.expunge(row)
            return tuple(rows)

    def latest_for_scene(self, scene_id: str) -> GenerationAttemptRow | None:
        with self._database.session() as session:
            row = session.scalar(
                select(GenerationAttemptRow)
                .where(GenerationAttemptRow.scene_id == scene_id)
                .order_by(
                    GenerationAttemptRow.ordinal.desc(),
                    GenerationAttemptRow.id.desc(),
                )
                .limit(1)
            )
            if row is None:
                return None
            session.expunge(row)
            return row

    def patch_provenance(
        self,
        attempt_id: str,
        patch: Mapping[str, Any],
    ) -> GenerationAttemptRow:
        with self._database.session() as session:
            row = session.get(GenerationAttemptRow, attempt_id)
            if row is None:
                raise KeyError(f"unknown generation attempt: {attempt_id}")
            row.provenance_json = {**row.provenance_json, **dict(patch)}
        return self.require(attempt_id)

    def record_backend_status(
        self,
        attempt_id: str,
        *,
        status: str,
        error: Mapping[str, Any] | None = None,
    ) -> GenerationAttemptRow:
        with self._database.session() as session:
            row = session.get(GenerationAttemptRow, attempt_id)
            if row is None:
                raise KeyError(f"unknown generation attempt: {attempt_id}")
            row.backend_status = status
            row.error_json = dict(error) if error is not None else None
        return self.require(attempt_id)


class EvaluationRepository:
    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def record(self, result: EvaluationResult) -> EvaluationRow:
        now = datetime.now(UTC)
        with self._database.session() as session:
            attempt = session.get(GenerationAttemptRow, result.attempt_id)
            if attempt is None:
                raise KeyError(f"unknown generation attempt: {result.attempt_id}")
            existing = session.scalar(
                select(EvaluationRow).where(
                    EvaluationRow.attempt_id == result.attempt_id
                )
            )
            if existing is not None:
                raise ValueError(
                    f"attempt already has an evaluation: {result.attempt_id}"
                )
            row = EvaluationRow(
                id=self._id_factory(),
                attempt_id=result.attempt_id,
                result_state=result.state.value,
                scores_json=result.scores.model_dump(mode="json"),
                reasons_json=list(result.reasons),
                created_at=now,
            )
            session.add(row)
            evaluation_id = row.id
        return self.require(evaluation_id)

    def require(self, evaluation_id: str) -> EvaluationRow:
        with self._database.session() as session:
            row = session.get(EvaluationRow, evaluation_id)
            if row is None:
                raise KeyError(f"unknown evaluation: {evaluation_id}")
            session.expunge(row)
            return row

    def for_scene(self, scene_id: str) -> tuple[EvaluationRow, ...]:
        with self._database.session() as session:
            attempt_ids = select(GenerationAttemptRow.id).where(
                GenerationAttemptRow.scene_id == scene_id
            )
            rows = session.scalars(
                select(EvaluationRow)
                .where(EvaluationRow.attempt_id.in_(attempt_ids))
                .order_by(EvaluationRow.created_at.asc(), EvaluationRow.id.asc())
            ).all()
            for row in rows:
                session.expunge(row)
            return tuple(rows)
