from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import LlmCallRow


def _new_id() -> str:
    return uuid4().hex


class LlmCallRepository:
    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def begin(
        self,
        *,
        backend: str,
        model: str,
        runtime_version: str | None,
        schema_name: str,
        context_digest: str,
        schema_digest: str,
        input_tokens: int | None,
        repair_index: int,
        temperature: float,
        request_json: dict[str, Any],
        started_at: datetime | None = None,
    ) -> str:
        call_id = self._id_factory()
        with self._database.session() as session:
            session.add(
                LlmCallRow(
                    id=call_id,
                    backend=backend,
                    model=model,
                    runtime_version=runtime_version,
                    schema_name=schema_name,
                    context_digest=context_digest,
                    schema_digest=schema_digest,
                    input_tokens=input_tokens,
                    output_tokens=None,
                    repair_index=repair_index,
                    temperature=temperature,
                    status="running",
                    request_json=request_json,
                    response_text=None,
                    response_digest=None,
                    error_text=None,
                    started_at=started_at or datetime.now(UTC),
                    completed_at=None,
                )
            )
        return call_id

    def succeed(
        self,
        call_id: str,
        *,
        response_text: str,
        response_digest: str,
        output_tokens: int | None,
        completed_at: datetime | None = None,
    ) -> None:
        with self._database.session() as session:
            row = session.get(LlmCallRow, call_id)
            if row is None:
                raise KeyError(f"unknown LLM call: {call_id}")
            row.status = "success"
            row.response_text = response_text
            row.response_digest = response_digest
            row.output_tokens = output_tokens
            row.completed_at = completed_at or datetime.now(UTC)

    def fail(
        self,
        call_id: str,
        *,
        error_text: str,
        completed_at: datetime | None = None,
    ) -> None:
        with self._database.session() as session:
            row = session.get(LlmCallRow, call_id)
            if row is None:
                raise KeyError(f"unknown LLM call: {call_id}")
            row.status = "failed"
            row.error_text = error_text
            row.completed_at = completed_at or datetime.now(UTC)

    def recent(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        with self._database.session() as session:
            rows = session.scalars(
                select(LlmCallRow)
                .order_by(LlmCallRow.started_at.desc(), LlmCallRow.id.desc())
                .limit(limit)
            ).all()
        return tuple(
            {
                "id": row.id,
                "backend": row.backend,
                "model": row.model,
                "runtime_version": row.runtime_version,
                "schema_name": row.schema_name,
                "context_digest": row.context_digest,
                "schema_digest": row.schema_digest,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "repair_index": row.repair_index,
                "temperature": row.temperature,
                "status": row.status,
                "request_json": dict(row.request_json),
                "response_digest": row.response_digest,
                "error_text": row.error_text,
                "started_at": row.started_at,
                "completed_at": row.completed_at,
            }
            for row in rows
        )
