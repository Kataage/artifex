from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select

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

# Provenance events are authoritative inputs to real-machine qualification.
# Retain delivered Discord interaction receipts as well as unattended daemon
# markers: ordinary bounded telemetry pruning must not invalidate an already
# recorded qualification stage. Receipts contain IDs, not bot tokens or text.
_QUALIFICATION_PROOF_EVENTS = frozenset({
    "daemon.loop_started",
    "daemon.pack_finalized",
    "daemon.loop_stopped",
    "discord.interaction_completed",
})


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

    def record_daemon_pack_finalized_once(self, pack_id: str, run_id: str) -> bool:
        """Atomically append a unique Pack proof for this continuous loop.

        This is invoked only after the scheduler has checked persisted Pack
        FINALIZED/archive evidence. It cannot reconstruct proof if PC-A died
        before the call; the real-machine qualification must still fail closed.
        """
        key = f"daemon.pack_finalized:{run_id}:{pack_id}"
        if len(key) > 200:
            raise ValueError("daemon Pack proof key is too long")
        now = datetime.now(UTC)
        with self._database.session() as session:
            if session.get(SettingRow, key) is not None:
                return False
            session.add(
                SettingRow(
                    key=key,
                    value_json={"run_id": run_id, "pack_id": pack_id},
                    updated_at=now,
                )
            )
            session.add(
                AgentEventRow(
                    event_type="daemon.pack_finalized",
                    severity=EventSeverity.INFO.value,
                    payload_json={"run_id": run_id, "pack_id": pack_id},
                    created_at=now,
                )
            )
        return True

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

    def prune(self, *, max_rows: int) -> int:
        """Bound routine telemetry without deleting qualification receipts.

        Daemon loop provenance and actual Discord interaction receipts are
        retained independently of the routine event quota. An eight-hour soak
        or later routine pruning must not erase authenticated stage evidence.
        """
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        ordinary = AgentEventRow.event_type.not_in(_QUALIFICATION_PROOF_EVENTS)
        with self._database.session() as session:
            keep_ids = (
                select(AgentEventRow.id)
                .where(ordinary)
                .order_by(AgentEventRow.created_at.desc(), AgentEventRow.id.desc())
                .limit(max_rows)
            )
            remove_ids = session.scalars(
                select(AgentEventRow.id).where(
                    ordinary, AgentEventRow.id.not_in(keep_ids),
                )
            ).all()
            if remove_ids:
                session.execute(
                    delete(AgentEventRow).where(AgentEventRow.id.in_(remove_ids))
                )
            return len(remove_ids)

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
