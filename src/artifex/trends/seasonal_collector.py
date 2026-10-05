from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from artifex.planner.models import SeasonalEventSummary
from artifex.trends.models import RawSeasonalEvent
from artifex.trends.seasonal_repository import SeasonalRepository


class SeasonalProvider(Protocol):
    name: str

    async def collect(self, *, as_of: datetime) -> tuple[RawSeasonalEvent, ...]: ...


class SeasonalCollectionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    collected_raw: int
    persisted: int
    expired: int
    provider_failures: dict[str, str]
    provider_counts: dict[str, int]
    active_events: tuple[SeasonalEventSummary, ...]


class SeasonalCollector:
    def __init__(
        self,
        providers: Sequence[SeasonalProvider],
        repository: SeasonalRepository,
    ) -> None:
        self._providers = tuple(providers)
        self._repository = repository

    async def collect(self, *, as_of: datetime) -> SeasonalCollectionReport:
        raw: list[RawSeasonalEvent] = []
        failures: dict[str, str] = {}
        counts: dict[str, int] = {}
        for provider in self._providers:
            try:
                events = await provider.collect(as_of=as_of)
            except Exception as exc:  # noqa: BLE001
                failures[provider.name] = str(exc)
                continue
            counts[provider.name] = len(events)
            raw.extend(events)

        persisted = self._repository.upsert_many(raw)
        expired = self._repository.expire(as_of=as_of)
        active = self._repository.active_summaries(as_of=as_of)
        return SeasonalCollectionReport(
            collected_raw=len(raw),
            persisted=persisted,
            expired=expired,
            provider_failures=failures,
            provider_counts=counts,
            active_events=active,
        )
