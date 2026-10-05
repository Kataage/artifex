from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from artifex.config.models import TrendConfig
from artifex.db import Database
from artifex.planner.models import CharacterOption, PlanningContext
from artifex.research.models import (
    ResearchEvidence,
    ResearchIntent,
    ResearchSearchResponse,
    SearchSource,
)
from artifex.telemetry import TelemetryRepository
from artifex.trends import (
    RawTrendSignal,
    SeasonalCalendarProvider,
    SeasonalCollector,
    SeasonalRepository,
    SignalHealthRepository,
    SignalIngestionService,
    TrendCollector,
    TrendPlannerContext,
    TrendRepository,
    current_web_trend_provider,
)


class FakeResearchService:
    async def search(
        self,
        request: object,
        *,
        now: datetime | None = None,
        use_cache: bool = True,
    ) -> ResearchSearchResponse:
        del use_cache
        current = now or datetime.now(UTC)
        source = getattr(request, "source")
        query = getattr(request, "query")
        return ResearchSearchResponse(
            run_id=f"run-{source.value}",
            query=query,
            source=source,
            intent=ResearchIntent.CURRENT,
            cached=False,
            degraded=False,
            evidence=(
                ResearchEvidence(
                    id=f"evidence-{source.value}",
                    run_id=f"run-{source.value}",
                    provider="fake-public-source",
                    source=source,
                    rank=1,
                    url="https://example.com/item",
                    canonical_url="https://example.com/item",
                    title=(
                        "Very long external title "
                        + "x" * 500
                    ),
                    snippet="untrusted snippet should never enter Trend summary",
                    observed_at=current,
                    content_hash=f"hash-{source.value}",
                    adult=False,
                ),
            ),
        )


class GoodTrendProvider:
    name = "good-trend"

    async def collect(self, *, as_of: datetime) -> tuple[RawTrendSignal, ...]:
        return (
            RawTrendSignal(
                provider=self.name,
                external_id="source-1",
                topic="fresh illustration topic",
                strength=0.9,
                confidence=0.8,
                observed_at=as_of,
            ),
        )


class BrokenTrendProvider:
    name = "broken-trend"

    async def collect(self, *, as_of: datetime) -> tuple[RawTrendSignal, ...]:
        del as_of
        raise RuntimeError("source unavailable")


@pytest.mark.asyncio
async def test_research_trend_adapter_bounds_external_text_and_keeps_source_ids() -> None:
    config = TrendConfig(
        current_queries=("hololive fanart",),
        max_external_topic_chars=80,
        max_provider_results=4,
    )
    provider = current_web_trend_provider(
        FakeResearchService(),  # type: ignore[arg-type]
        config,
        region="jp-jp",
        safesearch="moderate",
    )
    now = datetime(2026, 10, 6, 0, tzinfo=UTC)

    signals = await provider.collect(as_of=now)

    assert len(signals) == 1
    assert len(signals[0].topic) <= 80
    assert signals[0].external_id == "evidence-news"
    assert signals[0].payload["research_evidence_id"] == "evidence-news"
    assert "snippet" not in signals[0].payload


@pytest.mark.asyncio
async def test_builtin_calendar_populates_halloween_and_planner_context(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 10, 6, 0, tzinfo=UTC)
    database = Database(
        f"sqlite:///{(tmp_path / 'seasonal.sqlite3').as_posix()}"
    )
    database.migrate()
    config = TrendConfig()
    seasonal = SeasonalRepository(database, config)
    collector = SeasonalCollector(
        (SeasonalCalendarProvider(config),),
        seasonal,
    )

    report = await collector.collect(as_of=now)
    assert report.provider_failures == {}
    assert any(event.title == "Halloween" for event in report.active_events)

    context = PlanningContext(
        as_of=now,
        characters=(CharacterOption(id="char-a", display_name="A"),),
    )
    attached = TrendPlannerContext(
        TrendRepository(database, config),
        config,
        seasonal=seasonal,
    ).attach(context)

    halloween = next(
        event for event in attached.seasonal_events
        if event.title == "Halloween"
    )
    assert halloween.id.startswith("seasonal_")
    assert halloween.provider == "builtin-seasonal-calendar"
    assert halloween.starts_at is not None
    assert halloween.ends_at is not None
    database.dispose()


@pytest.mark.asyncio
async def test_provider_failure_is_telemetried_but_other_signals_survive(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 10, 6, 0, tzinfo=UTC)
    database = Database(
        f"sqlite:///{(tmp_path / 'signals.sqlite3').as_posix()}"
    )
    database.migrate()
    config = TrendConfig()
    trend_repo = TrendRepository(database, config)
    seasonal_repo = SeasonalRepository(database, config)
    telemetry = TelemetryRepository(database)
    service = SignalIngestionService(
        TrendCollector(
            (BrokenTrendProvider(), GoodTrendProvider()),
            trend_repo,
            config,
        ),
        SeasonalCollector(
            (SeasonalCalendarProvider(config),),
            seasonal_repo,
        ),
        trend_repo,
        seasonal_repo,
        SignalHealthRepository(database),
        telemetry,
    )

    report = await service.refresh(as_of=now)
    snapshot = service.snapshot(as_of=now)

    assert report.trend.provider_failures == {
        "broken-trend": "source unavailable"
    }
    assert len(snapshot.trends) == 1
    assert snapshot.seasonal
    health = {
        (item.kind, item.provider): item
        for item in snapshot.sources
    }
    assert health[("trend", "good-trend")].state == "healthy"
    assert health[("trend", "broken-trend")].state == "degraded"
    assert health[("trend", "broken-trend")].consecutive_failures == 1
    events = telemetry.recent(limit=10)
    assert any(event.event_type == "signals.provider_failure" for event in events)
    database.dispose()


def test_source_health_survives_repository_restart(tmp_path: Path) -> None:
    now = datetime(2026, 10, 6, 0, tzinfo=UTC)
    database = Database(
        f"sqlite:///{(tmp_path / 'health.sqlite3').as_posix()}"
    )
    database.migrate()
    first = SignalHealthRepository(database)
    first.failure(
        "provider-a",
        kind="trend",
        error="temporary failure",
        at=now,
    )

    second = SignalHealthRepository(database)
    restored = second.get("provider-a", kind="trend")

    assert restored is not None
    assert restored.state == "degraded"
    assert restored.consecutive_failures == 1
    assert restored.last_error == "temporary failure"
    database.dispose()
