from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from artifex.planner.models import SeasonalEventSummary, TrendSignalSummary
from artifex.telemetry import EventSeverity, TelemetryRepository
from artifex.trends.collector import TrendCollectionReport, TrendCollector
from artifex.trends.health import SignalHealthRepository
from artifex.trends.models import SignalSourceHealth
from artifex.trends.repository import TrendRepository
from artifex.trends.seasonal_collector import (
    SeasonalCollectionReport,
    SeasonalCollector,
)
from artifex.trends.seasonal_repository import SeasonalRepository


class SignalRefreshReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    refreshed_at: datetime
    trend: TrendCollectionReport
    seasonal: SeasonalCollectionReport


class SignalSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    trends: tuple[TrendSignalSummary, ...]
    seasonal: tuple[SeasonalEventSummary, ...]
    sources: tuple[SignalSourceHealth, ...]


class SignalIngestionService:
    def __init__(
        self,
        trend_collector: TrendCollector,
        seasonal_collector: SeasonalCollector,
        trend_repository: TrendRepository,
        seasonal_repository: SeasonalRepository,
        health: SignalHealthRepository,
        telemetry: TelemetryRepository,
    ) -> None:
        self._trend_collector = trend_collector
        self._seasonal_collector = seasonal_collector
        self._trend_repository = trend_repository
        self._seasonal_repository = seasonal_repository
        self._health = health
        self._telemetry = telemetry

    async def refresh(
        self,
        *,
        as_of: datetime | None = None,
    ) -> SignalRefreshReport:
        current = as_of or datetime.now(UTC)

        trend = await self._trend_collector.collect(as_of=current)
        seasonal = await self._seasonal_collector.collect(as_of=current)

        self._apply_health(
            kind="trend",
            counts=trend.provider_counts,
            failures=trend.provider_failures,
            at=current,
        )
        self._apply_health(
            kind="seasonal",
            counts=seasonal.provider_counts,
            failures=seasonal.provider_failures,
            at=current,
        )

        severity = (
            EventSeverity.WARNING
            if trend.provider_failures or seasonal.provider_failures
            else EventSeverity.INFO
        )
        self._telemetry.record(
            "signals.refresh",
            severity,
            {
                "trend_raw": trend.collected_raw,
                "trend_active": len(trend.active_signals),
                "trend_failures": trend.provider_failures,
                "seasonal_raw": seasonal.collected_raw,
                "seasonal_active": len(seasonal.active_events),
                "seasonal_failures": seasonal.provider_failures,
            },
            created_at=current,
        )
        return SignalRefreshReport(
            refreshed_at=current,
            trend=trend,
            seasonal=seasonal,
        )

    def snapshot(
        self,
        *,
        as_of: datetime | None = None,
    ) -> SignalSnapshot:
        current = as_of or datetime.now(UTC)
        return SignalSnapshot(
            as_of=current,
            trends=self._trend_repository.active_summaries(as_of=current),
            seasonal=self._seasonal_repository.active_summaries(as_of=current),
            sources=self._health.list(),
        )

    def _apply_health(
        self,
        *,
        kind: str,
        counts: dict[str, int],
        failures: dict[str, str],
        at: datetime,
    ) -> None:
        for provider, count in counts.items():
            self._health.success(
                provider,
                kind=kind,
                count=count,
                at=at,
            )
        for provider, error in failures.items():
            health = self._health.failure(
                provider,
                kind=kind,
                error=error,
                at=at,
            )
            self._telemetry.record(
                "signals.provider_failure",
                EventSeverity.WARNING,
                {
                    "provider": provider,
                    "kind": kind,
                    "error": error[:1000],
                    "consecutive_failures": health.consecutive_failures,
                },
                created_at=at,
            )
