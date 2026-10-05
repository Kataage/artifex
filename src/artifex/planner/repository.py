from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import ConceptRow
from artifex.planner.models import (
    CandidateScore,
    ConceptCandidate,
    RecentConceptSummary,
    ScoredConcept,
    SelectedConcept,
)


def _new_id() -> str:
    return uuid4().hex


class ConceptRepository:
    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def persist_selection(
        self,
        scored: Sequence[ScoredConcept],
        *,
        selected_key: str,
    ) -> SelectedConcept:
        selected_result: SelectedConcept | None = None
        now = datetime.now(UTC)

        with self._database.session() as session:
            for item in scored:
                concept_id = self._id_factory()
                is_selected = item.candidate.candidate_key == selected_key
                if item.score.rejected_reason is not None:
                    status = "rejected_candidate"
                elif is_selected:
                    status = "idea"
                else:
                    status = "candidate"

                payload = {
                    "candidate": item.candidate.model_dump(mode="json"),
                    "score": item.score.model_dump(mode="json"),
                    "selected": is_selected,
                }
                session.add(
                    ConceptRow(
                        id=concept_id,
                        status=status,
                        payload_json=payload,
                        score=item.score.aggregate,
                        similarity_score=item.score.similarity_penalty,
                        created_at=now,
                    )
                )
                if is_selected:
                    selected_result = SelectedConcept(
                        concept_id=concept_id,
                        candidate=item.candidate,
                        score=item.score,
                    )

        if selected_result is None:
            raise KeyError(f"selected candidate was not persisted: {selected_key}")
        return selected_result

    def get_selected(self, concept_id: str) -> SelectedConcept | None:
        with self._database.session() as session:
            row = session.get(ConceptRow, concept_id)
            if row is None:
                return None
            return self._selected_from_row(row)

    def require_selected(self, concept_id: str) -> SelectedConcept:
        selected = self.get_selected(concept_id)
        if selected is None:
            raise KeyError(f"unknown selected concept: {concept_id}")
        return selected

    def next_idea(self) -> SelectedConcept | None:
        with self._database.session() as session:
            row = session.scalar(
                select(ConceptRow)
                .where(ConceptRow.status == "idea")
                .order_by(ConceptRow.created_at.asc(), ConceptRow.id.asc())
                .limit(1)
            )
            if row is None:
                return None
            return self._selected_from_row(row)

    def idea(self, concept_id: str) -> SelectedConcept:
        with self._database.session() as session:
            row = session.get(ConceptRow, concept_id)
            if row is None or row.status != "idea":
                raise KeyError(f"concept is not an available idea: {concept_id}")
            session.expunge(row)
        return self._selected_from_row(row)

    @staticmethod
    def _selected_from_row(row: ConceptRow) -> SelectedConcept:
        raw_candidate = row.payload_json.get("candidate")
        raw_score = row.payload_json.get("score")
        selected = row.payload_json.get("selected")
        if selected is not True:
            raise KeyError(f"concept is not a selected concept: {row.id}")
        if not isinstance(raw_candidate, dict) or not isinstance(raw_score, dict):
            raise TypeError(f"concept {row.id} is missing candidate/score payload")
        return SelectedConcept(
            concept_id=row.id,
            candidate=ConceptCandidate.model_validate(raw_candidate),
            score=CandidateScore.model_validate(raw_score),
        )

    def set_status(self, concept_id: str, status: str) -> None:
        if not status.strip():
            raise ValueError("concept status must be non-empty")
        with self._database.session() as session:
            row = session.get(ConceptRow, concept_id)
            if row is None:
                raise KeyError(f"unknown concept: {concept_id}")
            row.status = status

    def recent_summaries(self, *, limit: int = 30) -> tuple[RecentConceptSummary, ...]:
        return self.historical_summaries(limit=limit)

    def historical_summaries(
        self,
        *,
        limit: int = 500,
    ) -> tuple[RecentConceptSummary, ...]:
        if limit < 1:
            return ()

        with self._database.session() as session:
            rows = session.scalars(
                select(ConceptRow)
                .order_by(ConceptRow.created_at.desc(), ConceptRow.id.desc())
                .limit(limit)
            ).all()

        summaries: list[RecentConceptSummary] = []
        for row in rows:
            raw_candidate = row.payload_json.get("candidate")
            selected = row.payload_json.get("selected")
            if selected is not True or not isinstance(raw_candidate, dict):
                continue
            try:
                summaries.append(
                    RecentConceptSummary(
                        concept_id=row.id,
                        character_ids=tuple(raw_candidate["character_ids"]),
                        theme=str(raw_candidate["theme"]),
                        setting=str(raw_candidate["setting"]),
                        visual_hook=str(raw_candidate["visual_hook"]),
                    )
                )
            except (KeyError, TypeError, ValidationError):
                continue
        return tuple(summaries)
