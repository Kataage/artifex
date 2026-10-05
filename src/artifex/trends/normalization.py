from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from artifex.config.models import TrendConfig
from artifex.trends.models import (
    NormalizedTrendSignal,
    RawTrendSignal,
    TrendSourceReference,
)

_TOPIC_SEPARATORS = re.compile(r"[\W_]+", re.UNICODE)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def normalize_topic(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_TOPIC_SEPARATORS.sub(" ", normalized).split())


def stable_trend_id(normalized_topic: str) -> str:
    digest = hashlib.sha256(normalized_topic.encode("utf-8")).hexdigest()[:24]
    return f"trend_{digest}"


def merge_signals(
    signals: Sequence[RawTrendSignal],
    config: TrendConfig,
) -> tuple[NormalizedTrendSignal, ...]:
    grouped: dict[str, list[RawTrendSignal]] = {}
    for signal in signals:
        key = normalize_topic(signal.topic)
        if not key:
            continue
        grouped.setdefault(key, []).append(signal)

    merged: list[NormalizedTrendSignal] = []
    default_ttl = timedelta(hours=config.default_ttl_hours)

    for key in sorted(grouped):
        items = grouped[key]
        items.sort(
            key=lambda item: (
                item.strength * item.confidence,
                as_utc(item.observed_at),
                item.provider,
                item.external_id or "",
            ),
            reverse=True,
        )
        representative = items[0]
        observed_at = max(as_utc(item.observed_at) for item in items)
        expiries = tuple(
            as_utc(item.expires_at)
            if item.expires_at is not None
            else as_utc(item.observed_at) + default_ttl
            for item in items
        )
        expires_at = max(expiries)

        confidence_product = 1.0
        for item in items:
            confidence_product *= 1.0 - item.confidence
        combined_confidence = 1.0 - confidence_product

        combined_strength = max(item.strength for item in items)

        sources = tuple(
            TrendSourceReference(
                provider=item.provider,
                external_id=item.external_id,
                strength=item.strength,
                confidence=item.confidence,
                observed_at=as_utc(item.observed_at),
            )
            for item in sorted(
                items,
                key=lambda item: (
                    item.provider.casefold(),
                    item.external_id or "",
                    as_utc(item.observed_at),
                ),
            )
        )
        merged.append(
            NormalizedTrendSignal(
                id=stable_trend_id(key),
                normalized_topic=key,
                topic=representative.topic.strip(),
                strength=combined_strength,
                confidence=combined_confidence,
                observed_at=observed_at,
                expires_at=expires_at,
                sources=sources,
                payload={"source_count": len(sources)},
            )
        )

    return tuple(merged)
