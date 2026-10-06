from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import LoRAValidationRunRow


class ValidationRunModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoRAValidationRun(ValidationRunModel):
    id: str
    lora_id: str
    checksum: str | None
    status: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    started_at: datetime
    finished_at: datetime | None = None


class LoRAValidationRunRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def start(
        self,
        lora_id: str,
        *,
        checksum: str | None,
        evidence: dict[str, Any] | None = None,
    ) -> LoRAValidationRun:
        run = LoRAValidationRun(
            id=uuid4().hex,
            lora_id=lora_id,
            checksum=checksum,
            status="running",
            evidence=dict(evidence or {}),
            started_at=datetime.now(UTC),
        )
        with self._database.session() as session:
            session.add(
                LoRAValidationRunRow(
                    id=run.id,
                    lora_id=run.lora_id,
                    checksum=run.checksum,
                    status=run.status,
                    evidence_json=run.evidence,
                    report_json=None,
                    error_json=None,
                    started_at=run.started_at,
                    finished_at=None,
                )
            )
        return run

    def update_evidence(
        self,
        run_id: str,
        evidence: dict[str, Any],
    ) -> LoRAValidationRun:
        with self._database.session() as session:
            row = session.get(LoRAValidationRunRow, run_id)
            if row is None:
                raise KeyError(f"unknown LoRA validation run: {run_id}")
            row.evidence_json = evidence
        return self.require(run_id)

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        evidence: dict[str, Any],
        report: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> LoRAValidationRun:
        finished_at = datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(LoRAValidationRunRow, run_id)
            if row is None:
                raise KeyError(f"unknown LoRA validation run: {run_id}")
            row.status = status
            row.evidence_json = evidence
            row.report_json = report
            row.error_json = error
            row.finished_at = finished_at
        return self.require(run_id)

    def get(self, run_id: str) -> LoRAValidationRun | None:
        with self._database.session() as session:
            row = session.get(LoRAValidationRunRow, run_id)
            return None if row is None else self._from_row(row)

    def require(self, run_id: str) -> LoRAValidationRun:
        run = self.get(run_id)
        if run is None:
            raise KeyError(f"unknown LoRA validation run: {run_id}")
        return run

    def recent(
        self,
        *,
        lora_id: str | None = None,
        limit: int = 20,
    ) -> tuple[LoRAValidationRun, ...]:
        query = select(LoRAValidationRunRow).order_by(
            LoRAValidationRunRow.started_at.desc(),
            LoRAValidationRunRow.id.desc(),
        )
        if lora_id is not None:
            query = query.where(LoRAValidationRunRow.lora_id == lora_id)
        query = query.limit(limit)
        with self._database.session() as session:
            rows = session.scalars(query).all()
            return tuple(self._from_row(row) for row in rows)

    @staticmethod
    def _from_row(row: LoRAValidationRunRow) -> LoRAValidationRun:
        return LoRAValidationRun(
            id=row.id,
            lora_id=row.lora_id,
            checksum=row.checksum,
            status=row.status,
            evidence=dict(row.evidence_json or {}),
            report=(
                None if row.report_json is None else dict(row.report_json)
            ),
            error=None if row.error_json is None else dict(row.error_json),
            started_at=row.started_at,
            finished_at=row.finished_at,
        )
