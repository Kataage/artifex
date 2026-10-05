from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from artifex.config.models import TrendConfig
from artifex.planner.models import TrendSignalSummary
from artifex.trends.models import RawTrendSignal
from artifex.trends.normalization import merge_signals
from artifex.trends.provider import TrendProvider
from artifex.trends.repository import TrendRepository


class TrendCollectionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    collected_raw: int
    persisted_normalized: int
    expired: int
    provider_failures: dict[str, str]
    provider_counts: dict[str, int]
    active_signals: tuple[TrendSignalSummary, ...]


class TrendCollector:
    def __init__(
        self,
        providers: Sequence[TrendProvider],
        repository: TrendRepository,
        config: TrendConfig,
    ) -> None:
        self._providers = tuple(providers)
        self._repository = repository
        self._config = config

    async def collect(self, *, as_of: datetime) -> TrendCollectionReport:
        raw: list[RawTrendSignal] = []
        failures: dict[str, str] = {}
        counts: dict[str, int] = {}

        if self._config.enabled:
            for provider in self._providers:
                try:
                    signals = await provider.collect(as_of=as_of)
                except Exception as exc:  # noqa: BLE001
                    # Provider implementations are an isolation boundary. A broken
                    # trend source must never stop evergreen/seasonal production.
                    failures[provider.name] = str(exc)
                    continue
                counts[provider.name] = len(signals)
                raw.extend(signals)

        normalized = merge_signals(raw, self._config)
        persisted = self._repository.upsert_many(normalized)
        expired = self._repository.expire(as_of=as_of)
        active = self._repository.active_summaries(as_of=as_of)

        return TrendCollectionReport(
            collected_raw=len(raw),
            persisted_normalized=persisted,
            expired=expired,
            provider_failures=failures,
            provider_counts=counts,
            active_signals=active,
        )
