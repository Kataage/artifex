from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.config.models import TrendConfig
from artifex.db import Database
from artifex.trends.models import NormalizedTrendSignal, TrendSourceReference
from artifex.trends.repository import TrendRepository


def _signal(now: datetime) -> NormalizedTrendSignal:
    return NormalizedTrendSignal(
        id="trend-1",
        normalized_topic="topic",
        topic="Topic",
        strength=0.8,
        confidence=0.75,
        observed_at=now,
        expires_at=now + timedelta(hours=24),
        sources=(
            TrendSourceReference(
                provider="test",
                external_id="1",
                strength=0.8,
                confidence=0.75,
                observed_at=now,
            ),
        ),
    )


def test_repository_applies_freshness_and_expiry(tmp_path: Path) -> None:
    now = datetime(2026, 10, 5, 0, tzinfo=UTC)
    database = Database(f"sqlite:///{(tmp_path / 'trends.sqlite3').as_posix()}")
    database.migrate()
    repository = TrendRepository(
        database,
        TrendConfig(freshness_half_life_hours=8),
    )
    assert repository.upsert_many((_signal(now),)) == 1

    summaries = repository.active_summaries(as_of=now + timedelta(hours=8))
    assert len(summaries) == 1
    assert summaries[0].freshness == pytest.approx(0.5)
    assert summaries[0].confidence == pytest.approx(0.75)

    expired = repository.expire(as_of=now + timedelta(hours=25))
    assert expired == 1
    assert repository.active_summaries(as_of=now + timedelta(hours=25)) == ()
    database.dispose()


def test_repository_filters_weak_effective_signals(tmp_path: Path) -> None:
    now = datetime(2026, 10, 5, 0, tzinfo=UTC)
    database = Database(f"sqlite:///{(tmp_path / 'weak.sqlite3').as_posix()}")
    database.migrate()
    repository = TrendRepository(
        database,
        TrendConfig(minimum_effective_strength=0.5),
    )
    weak = _signal(now).model_copy(update={"strength": 0.2, "confidence": 0.5})
    repository.upsert_many((weak,))

    assert repository.active_summaries(as_of=now) == ()
    database.dispose()
