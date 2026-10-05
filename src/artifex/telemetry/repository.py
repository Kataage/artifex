from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from artifex.db import Database
from artifex.db.models import (
    AgentEventRow,
    EvaluationRow,
    GenerationAttemptRow,
    PackRow,
    ReviewQueueRow,
    SettingRow,
)
from artifex.domain import PackState, ResultState
from artifex.review import ReviewState
from artifex.telemetry.models import DailyMetrics, EventSeverity, OperationalEvent

_HEARTBEAT_KEY = "runtime.heartbeat"


class TelemetryRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def record(
        self,
        event_type: str,
        severity: EventSeverity,
        payload: Mapping[str, object] | None = None,
        *,
        created_at: datetime | None = None,
    ) -> OperationalEvent:
        now = created_at or datetime.now(UTC)
        with self._database.session() as session:
            row = AgentEventRow(
                event_type=event_type,
                severity=severity.value,
                payload_json=dict(payload or {}),
                created_at=now,
            )
            session.add(row)
            session.flush()
            event_id = row.id
        return self.require(event_id)

    def require(self, event_id: int) -> OperationalEvent:
        with self._database.session() as session:
            row = session.get(AgentEventRow, event_id)
            if row is None:
                raise KeyError(f"unknown operational event: {event_id}")
            return OperationalEvent(
                id=row.id,
                event_type=row.event_type,
                severity=EventSeverity(row.severity),
                payload=dict(row.payload_json),
                created_at=row.created_at,
            )

    def recent(self, *, limit: int = 100) -> tuple[OperationalEvent, ...]:
        with self._database.session() as session:
            rows = session.scalars(
                select(AgentEventRow)
                .order_by(AgentEventRow.created_at.desc(), AgentEventRow.id.desc())
                .limit(limit)
            ).all()
            return tuple(
                OperationalEvent(
                    id=row.id,
                    event_type=row.event_type,
                    severity=EventSeverity(row.severity),
                    payload=dict(row.payload_json),
                    created_at=row.created_at,
                )
                for row in rows
            )

    def heartbeat(self, *, at: datetime | None = None) -> None:
        now = at or datetime.now(UTC)
        payload = {"at": now.isoformat()}
        with self._database.session() as session:
            row = session.get(SettingRow, _HEARTBEAT_KEY)
            if row is None:
                session.add(
                    SettingRow(
                        key=_HEARTBEAT_KEY,
                        value_json=payload,
                        updated_at=now,
                    )
                )
            else:
                row.value_json = payload
                row.updated_at = now

    def heartbeat_age_seconds(self, *, now: datetime | None = None) -> float | None:
        current = now or datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(SettingRow, _HEARTBEAT_KEY)
            if row is None:
                return None
            age = current - row.updated_at
            return max(0.0, age.total_seconds())

    def daily_metrics(
        self,
        *,
        now: datetime | None = None,
        hours: int = 24,
    ) -> DailyMetrics:
        until = now or datetime.now(UTC)
        since = until - timedelta(hours=hours)
        with self._database.session() as session:
            def count_packs(state: PackState) -> int:
                return int(
                    session.scalar(
                        select(func.count())
                        .select_from(PackRow)
                        .where(PackRow.state == state.value, PackRow.updated_at >= since)
                    )
                    or 0
                )

            attempts = int(
                session.scalar(
                    select(func.count())
                    .select_from(GenerationAttemptRow)
                    .where(GenerationAttemptRow.created_at >= since)
                )
                or 0
            )

            def count_evaluations(state: ResultState) -> int:
                return int(
                    session.scalar(
                        select(func.count())
                        .select_from(EvaluationRow)
                        .where(
                            EvaluationRow.result_state == state.value,
                            EvaluationRow.created_at >= since,
                        )
                    )
                    or 0
                )

            open_reviews = int(
                session.scalar(
                    select(func.count())
                    .select_from(ReviewQueueRow)
                    .where(ReviewQueueRow.state == ReviewState.OPEN.value)
                )
                or 0
            )
            errors = int(
                session.scalar(
                    select(func.count())
                    .select_from(AgentEventRow)
                    .where(
                        AgentEventRow.created_at >= since,
                        AgentEventRow.severity.in_(
                            (EventSeverity.ERROR.value, EventSeverity.CRITICAL.value)
                        ),
                    )
                )
                or 0
            )

        return DailyMetrics(
            since=since,
            until=until,
            finalized_packs=count_packs(PackState.FINALIZED),
            failed_packs=count_packs(PackState.FAILED),
            blocked_packs=count_packs(PackState.BLOCKED),
            generation_attempts=attempts,
            accepted_evaluations=count_evaluations(ResultState.ACCEPTED),
            review_evaluations=count_evaluations(ResultState.REVIEW),
            rejected_evaluations=count_evaluations(ResultState.REJECTED),
            open_reviews=open_reviews,
            errors=errors,
        )
