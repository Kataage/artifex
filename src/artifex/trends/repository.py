from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import delete, select

from artifex.config.models import TrendConfig
from artifex.db import Database
from artifex.db.models import TrendSignalRow
from artifex.planner.models import TrendSignalSummary
from artifex.trends.models import NormalizedTrendSignal
from artifex.trends.normalization import as_utc


class TrendRepository:
    def __init__(self, database: Database, config: TrendConfig) -> None:
        self._database = database
        self._config = config

    def upsert_many(self, signals: Sequence[NormalizedTrendSignal]) -> int:
        count = 0
        with self._database.session() as session:
            for signal in signals:
                row = session.get(TrendSignalRow, signal.id)
                providers = sorted({source.provider for source in signal.sources})
                payload = {
                    **signal.payload,
                    "normalized_topic": signal.normalized_topic,
                    "sources": [
                        source.model_dump(mode="json")
                        for source in signal.sources
                    ],
                }
                if row is None:
                    session.add(
                        TrendSignalRow(
                            id=signal.id,
                            provider=",".join(providers),
                            topic=signal.topic,
                            strength=signal.strength,
                            confidence=signal.confidence,
                            observed_at=signal.observed_at,
                            expires_at=signal.expires_at,
                            payload_json=payload,
                        )
                    )
                else:
                    row.provider = ",".join(providers)
                    row.topic = signal.topic
                    row.strength = signal.strength
                    row.confidence = signal.confidence
                    row.observed_at = signal.observed_at
                    row.expires_at = signal.expires_at
                    row.payload_json = payload
                count += 1
        return count

    def expire(self, *, as_of: datetime) -> int:
        cutoff = as_utc(as_of).replace(tzinfo=None)
        with self._database.session() as session:
            result = session.execute(
                delete(TrendSignalRow).where(
                    TrendSignalRow.expires_at.is_not(None),
                    TrendSignalRow.expires_at <= cutoff,
                )
            )
            return int(result.rowcount or 0)

    def active_summaries(
        self,
        *,
        as_of: datetime,
        limit: int | None = None,
    ) -> tuple[TrendSignalSummary, ...]:
        now = as_utc(as_of)
        query_limit = limit or self._config.max_summary_signals

        with self._database.session() as session:
            rows = session.scalars(
                select(TrendSignalRow)
                .where(
                    (TrendSignalRow.expires_at.is_(None))
                    | (TrendSignalRow.expires_at > now.replace(tzinfo=None))
                )
                .order_by(
                    TrendSignalRow.strength.desc(),
                    TrendSignalRow.confidence.desc(),
                    TrendSignalRow.observed_at.desc(),
                    TrendSignalRow.id.asc(),
                )
            ).all()

        summaries: list[TrendSignalSummary] = []
        for row in rows:
            freshness = self._freshness(row.observed_at, now)
            effective = row.strength * row.confidence * freshness
            if effective < self._config.minimum_effective_strength:
                continue
            summaries.append(
                TrendSignalSummary(
                    id=row.id,
                    topic=row.topic,
                    strength=row.strength,
                    confidence=row.confidence,
                    freshness=freshness,
                )
            )

        summaries.sort(
            key=lambda item: (
                -(item.strength * item.confidence * item.freshness),
                item.id,
            )
        )
        return tuple(summaries[:query_limit])

    def _freshness(self, observed_at: datetime, as_of: datetime) -> float:
        observed = as_utc(observed_at)
        now = as_utc(as_of)
        age_hours = max(0.0, (now - observed).total_seconds() / 3600.0)
        half_life = self._config.freshness_half_life_hours
        return math.exp(-math.log(2.0) * age_hours / half_life)
