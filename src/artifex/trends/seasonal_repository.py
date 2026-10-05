from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, select

from artifex.config.models import TrendConfig
from artifex.db import Database
from artifex.db.models import SeasonalSignalRow
from artifex.planner.models import SeasonalEventSummary
from artifex.trends.models import RawSeasonalEvent, SeasonalEventRecord


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _stable_id(provider: str, external_id: str) -> str:
    raw = f"{provider}:{external_id}".encode("utf-8")
    return "seasonal_" + hashlib.sha256(raw).hexdigest()[:24]


class SeasonalRepository:
    def __init__(self, database: Database, config: TrendConfig) -> None:
        self._database = database
        self._config = config

    def upsert_many(self, events: Sequence[RawSeasonalEvent]) -> int:
        count = 0
        with self._database.session() as session:
            for event in events:
                event_id = _stable_id(event.provider, event.external_id)
                row = session.get(SeasonalSignalRow, event_id)
                payload = {
                    **event.payload,
                    "source_external_id": event.external_id,
                }
                if row is None:
                    session.add(
                        SeasonalSignalRow(
                            id=event_id,
                            provider=event.provider,
                            external_id=event.external_id,
                            title=event.title,
                            relevance=event.relevance,
                            starts_at=_utc(event.starts_at),
                            ends_at=_utc(event.ends_at),
                            payload_json=payload,
                        )
                    )
                else:
                    row.provider = event.provider
                    row.external_id = event.external_id
                    row.title = event.title
                    row.relevance = event.relevance
                    row.starts_at = _utc(event.starts_at)
                    row.ends_at = _utc(event.ends_at)
                    row.payload_json = payload
                count += 1
        return count

    def expire(self, *, as_of: datetime) -> int:
        current = _utc(as_of).replace(tzinfo=None)
        with self._database.session() as session:
            ids = session.scalars(
                select(SeasonalSignalRow.id).where(
                    SeasonalSignalRow.ends_at < current
                )
            ).all()
            if ids:
                session.execute(
                    delete(SeasonalSignalRow).where(
                        SeasonalSignalRow.id.in_(ids)
                    )
                )
            return len(ids)

    def active_summaries(
        self,
        *,
        as_of: datetime,
        limit: int | None = None,
    ) -> tuple[SeasonalEventSummary, ...]:
        current = _utc(as_of).replace(tzinfo=None)
        query_limit = limit or self._config.max_seasonal_events
        with self._database.session() as session:
            rows = session.scalars(
                select(SeasonalSignalRow)
                .where(
                    SeasonalSignalRow.starts_at <= current,
                    SeasonalSignalRow.ends_at >= current,
                )
                .order_by(
                    SeasonalSignalRow.relevance.desc(),
                    SeasonalSignalRow.ends_at.asc(),
                    SeasonalSignalRow.id.asc(),
                )
                .limit(query_limit)
            ).all()
        return tuple(
            SeasonalEventSummary(
                id=row.id,
                title=row.title,
                relevance=row.relevance,
                provider=row.provider,
                starts_at=row.starts_at,
                ends_at=row.ends_at,
            )
            for row in rows
        )

    def records(self, *, as_of: datetime) -> tuple[SeasonalEventRecord, ...]:
        current = _utc(as_of).replace(tzinfo=None)
        with self._database.session() as session:
            rows = session.scalars(
                select(SeasonalSignalRow)
                .where(SeasonalSignalRow.ends_at >= current)
                .order_by(
                    SeasonalSignalRow.starts_at.asc(),
                    SeasonalSignalRow.id.asc(),
                )
            ).all()
        return tuple(
            SeasonalEventRecord(
                id=row.id,
                provider=row.provider,
                external_id=row.external_id,
                title=row.title,
                relevance=row.relevance,
                starts_at=row.starts_at,
                ends_at=row.ends_at,
                payload=dict(row.payload_json),
            )
            for row in rows
        )
