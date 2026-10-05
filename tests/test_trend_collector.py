from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.config.models import TrendConfig
from artifex.db import Database
from artifex.planner.models import CharacterOption, PlanningContext
from artifex.trends import (
    RawTrendSignal,
    TrendCollector,
    TrendPlannerContext,
    TrendRepository,
)


class GoodProvider:
    name = "good"

    async def collect(self, *, as_of: datetime) -> tuple[RawTrendSignal, ...]:
        return (
            RawTrendSignal(
                provider=self.name,
                external_id="fresh-1",
                topic="Fresh topic",
                strength=0.9,
                confidence=0.8,
                observed_at=as_of,
            ),
        )


class BrokenProvider:
    name = "broken"

    async def collect(self, *, as_of: datetime) -> tuple[RawTrendSignal, ...]:
        del as_of
        raise RuntimeError("provider unavailable")


class EmptyProvider:
    name = "empty"

    async def collect(self, *, as_of: datetime) -> tuple[RawTrendSignal, ...]:
        del as_of
        return ()


@pytest.mark.asyncio
async def test_provider_failure_is_isolated_and_success_is_cached(tmp_path: Path) -> None:
    now = datetime(2026, 10, 5, 12, tzinfo=UTC)
    database = Database(f"sqlite:///{(tmp_path / 'collector.sqlite3').as_posix()}")
    database.migrate()
    config = TrendConfig()
    repository = TrendRepository(database, config)
    collector = TrendCollector(
        (BrokenProvider(), GoodProvider()),
        repository,
        config,
    )

    report = await collector.collect(as_of=now)

    assert report.collected_raw == 1
    assert report.persisted_normalized == 1
    assert report.provider_failures == {"broken": "provider unavailable"}
    assert len(report.active_signals) == 1
    assert report.active_signals[0].topic == "Fresh topic"

    cached_only = TrendCollector((BrokenProvider(),), repository, config)
    cached_report = await cached_only.collect(as_of=now + timedelta(hours=1))
    assert cached_report.collected_raw == 0
    assert len(cached_report.active_signals) == 1
    database.dispose()


@pytest.mark.asyncio
async def test_no_trends_does_not_block_planner_context(tmp_path: Path) -> None:
    now = datetime(2026, 10, 5, 12, tzinfo=UTC)
    database = Database(f"sqlite:///{(tmp_path / 'empty.sqlite3').as_posix()}")
    database.migrate()
    config = TrendConfig()
    repository = TrendRepository(database, config)
    report = await TrendCollector((EmptyProvider(),), repository, config).collect(
        as_of=now
    )
    assert report.active_signals == ()

    context = PlanningContext(
        as_of=now,
        characters=(CharacterOption(id="char-a", display_name="A"),),
    )
    attached = TrendPlannerContext(repository, config).attach(context)
    assert attached.trend_signals == ()
    database.dispose()


def test_disabled_trends_are_not_attached_to_planner(tmp_path: Path) -> None:
    now = datetime(2026, 10, 5, 12, tzinfo=UTC)
    database = Database(f"sqlite:///{(tmp_path / 'disabled.sqlite3').as_posix()}")
    database.migrate()
    enabled = TrendConfig()
    repository = TrendRepository(database, enabled)
    from artifex.trends.normalization import merge_signals

    repository.upsert_many(
        merge_signals(
            (
                RawTrendSignal(
                    provider="test",
                    topic="Cached trend",
                    strength=1,
                    confidence=1,
                    observed_at=now,
                ),
            ),
            enabled,
        )
    )
    context = PlanningContext(
        as_of=now,
        characters=(CharacterOption(id="char-a", display_name="A"),),
    )

    attached = TrendPlannerContext(
        repository,
        TrendConfig(enabled=False),
    ).attach(context)

    assert attached.trend_signals == ()
    database.dispose()
