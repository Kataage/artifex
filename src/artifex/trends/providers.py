from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from artifex.config.models import SeasonalEventConfig, TrendConfig
from artifex.research import (
    ResearchIntent,
    ResearchSearchRequest,
    ResearchService,
    SafeSearch,
    SearchSource,
)
from artifex.research.models import ResearchEvidence
from artifex.research.security import sanitize_text
from artifex.trends.models import RawSeasonalEvent, RawTrendSignal


class ResearchTrendProvider:
    """Convert bounded ResearchService evidence into normalized trend candidates."""

    def __init__(
        self,
        name: str,
        service: ResearchService,
        config: TrendConfig,
        requests: Sequence[ResearchSearchRequest],
        *,
        confidence: float,
    ) -> None:
        self.name = name
        self._service = service
        self._config = config
        self._requests = tuple(requests)
        self._confidence = confidence

    async def collect(self, *, as_of: datetime) -> tuple[RawTrendSignal, ...]:
        collected: list[RawTrendSignal] = []
        seen: set[str] = set()
        for request in self._requests:
            response = await self._service.search(request, now=as_of)
            count = max(1, len(response.evidence))
            for evidence in response.evidence:
                topic = _topic(evidence, self._config.max_external_topic_chars)
                if not topic:
                    continue
                key = topic.casefold()
                if key in seen:
                    continue
                seen.add(key)
                rank_strength = max(
                    0.15,
                    1.0 - ((evidence.rank - 1) / max(1, count)),
                )
                recency = _recency(evidence, as_of)
                score_strength = _score_strength(evidence)
                strength = max(
                    0.0,
                    min(
                        1.0,
                        rank_strength * 0.55
                        + recency * 0.25
                        + score_strength * 0.20,
                    ),
                )
                confidence = self._confidence * (0.75 if response.degraded else 1.0)
                collected.append(
                    RawTrendSignal(
                        provider=self.name,
                        external_id=evidence.id,
                        topic=topic,
                        strength=strength,
                        confidence=max(0.0, min(1.0, confidence)),
                        observed_at=as_of,
                        expires_at=as_of
                        + timedelta(hours=self._config.default_ttl_hours),
                        payload={
                            "research_run_id": response.run_id,
                            "research_evidence_id": evidence.id,
                            "research_provider": evidence.provider,
                            "source": evidence.source.value,
                            "rank": evidence.rank,
                            "untrusted_external_data": True,
                        },
                    )
                )
        return tuple(collected)


def current_web_trend_provider(
    service: ResearchService,
    config: TrendConfig,
    *,
    region: str,
    safesearch: str,
) -> ResearchTrendProvider:
    requests: list[ResearchSearchRequest] = []
    sources = (SearchSource.NEWS, SearchSource.WEB, SearchSource.IMAGES)
    for index, query in enumerate(config.current_queries):
        requests.append(
            ResearchSearchRequest(
                query=query,
                source=sources[index % len(sources)],
                intent=ResearchIntent.CURRENT,
                max_results=config.max_provider_results,
                region=region,
                safesearch=SafeSearch(safesearch),
                timelimit="w",
            )
        )
    return ResearchTrendProvider(
        "research-current",
        service,
        config,
        requests,
        confidence=0.70,
    )


def tag_trend_provider(
    service: ResearchService,
    config: TrendConfig,
    *,
    region: str,
    safesearch: str,
) -> ResearchTrendProvider:
    requests = tuple(
        ResearchSearchRequest(
            query=query,
            source=SearchSource.TAGS,
            intent=ResearchIntent.CURRENT,
            max_results=config.max_provider_results,
            region=region,
            safesearch=SafeSearch(safesearch),
        )
        for query in config.tag_queries
    )
    return ResearchTrendProvider(
        "gelbooru-tags",
        service,
        config,
        requests,
        confidence=0.82,
    )


class SeasonalCalendarProvider:
    name = "builtin-seasonal-calendar"

    def __init__(self, config: TrendConfig) -> None:
        self._config = config

    async def collect(self, *, as_of: datetime) -> tuple[RawSeasonalEvent, ...]:
        current = _utc(as_of)
        definitions = (*_BUILTIN_EVENTS, *self._config.custom_seasonal_events)
        events: list[RawSeasonalEvent] = []

        for year in (current.year - 1, current.year, current.year + 1):
            for definition in definitions:
                try:
                    event_date = datetime(
                        year,
                        definition.month,
                        definition.day,
                        12,
                        tzinfo=UTC,
                    )
                except ValueError:
                    continue
                starts = event_date - timedelta(days=definition.lead_days)
                ends = event_date + timedelta(days=definition.trail_days, hours=12)
                if not (starts <= current <= ends):
                    continue
                distance_days = abs((event_date - current).total_seconds()) / 86400.0
                proximity = math.exp(
                    -distance_days / max(1.0, definition.lead_days / 3.0)
                )
                relevance = min(
                    1.0,
                    max(
                        definition.relevance * 0.55,
                        definition.relevance * (0.55 + 0.45 * proximity),
                    ),
                )
                events.append(
                    RawSeasonalEvent(
                        provider=self.name,
                        external_id=f"{definition.id}:{year}",
                        title=definition.title,
                        relevance=relevance,
                        starts_at=starts,
                        ends_at=ends,
                        payload={
                            "event_id": definition.id,
                            "event_date": event_date.date().isoformat(),
                            "calendar_version": "illustration-calendar-v1",
                        },
                    )
                )

        events.sort(
            key=lambda item: (
                -item.relevance,
                item.ends_at,
                item.external_id,
            )
        )
        return tuple(events[: self._config.max_seasonal_events])


def _topic(evidence: ResearchEvidence, limit: int) -> str:
    title = sanitize_text(evidence.title, max_chars=limit)
    if evidence.source is SearchSource.TAGS:
        title = title.replace("_", " ")
    title = " ".join(title.split())
    if len(title) < 2:
        return ""
    return title


def _recency(evidence: ResearchEvidence, as_of: datetime) -> float:
    timestamp = evidence.published_at or evidence.observed_at
    current = _utc(as_of)
    observed = _utc(timestamp)
    age_hours = max(0.0, (current - observed).total_seconds() / 3600.0)
    return math.exp(-age_hours / (24.0 * 7.0))


def _score_strength(evidence: ResearchEvidence) -> float:
    raw = evidence.metadata.get("count")
    if isinstance(raw, int | float) and raw > 0:
        return min(1.0, math.log1p(float(raw)) / 15.0)
    return 0.5


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


_BUILTIN_EVENTS: tuple[SeasonalEventConfig, ...] = (
    SeasonalEventConfig(
        id="new-year",
        title="New Year / oshogatsu",
        month=1,
        day=1,
        relevance=1.0,
        lead_days=30,
        trail_days=5,
    ),
    SeasonalEventConfig(
        id="valentines",
        title="Valentine's Day",
        month=2,
        day=14,
        relevance=0.95,
        lead_days=24,
        trail_days=2,
    ),
    SeasonalEventConfig(
        id="hinamatsuri",
        title="Hinamatsuri",
        month=3,
        day=3,
        relevance=0.78,
        lead_days=18,
        trail_days=2,
    ),
    SeasonalEventConfig(
        id="white-day",
        title="White Day",
        month=3,
        day=14,
        relevance=0.82,
        lead_days=18,
        trail_days=2,
    ),
    SeasonalEventConfig(
        id="sakura-season",
        title="Cherry blossom / sakura season",
        month=4,
        day=1,
        relevance=0.95,
        lead_days=30,
        trail_days=18,
    ),
    SeasonalEventConfig(
        id="tanabata",
        title="Tanabata",
        month=7,
        day=7,
        relevance=0.88,
        lead_days=24,
        trail_days=3,
    ),
    SeasonalEventConfig(
        id="summer-festival",
        title="Japanese summer festival season",
        month=8,
        day=1,
        relevance=0.88,
        lead_days=30,
        trail_days=30,
    ),
    SeasonalEventConfig(
        id="halloween",
        title="Halloween",
        month=10,
        day=31,
        relevance=1.0,
        lead_days=30,
        trail_days=3,
    ),
    SeasonalEventConfig(
        id="christmas",
        title="Christmas",
        month=12,
        day=25,
        relevance=1.0,
        lead_days=35,
        trail_days=3,
    ),
    SeasonalEventConfig(
        id="year-end",
        title="Year-end / winter holiday season",
        month=12,
        day=31,
        relevance=0.85,
        lead_days=20,
        trail_days=2,
    ),
)
