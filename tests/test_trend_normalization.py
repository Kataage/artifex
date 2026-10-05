from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from artifex.config.models import TrendConfig
from artifex.trends.models import RawTrendSignal
from artifex.trends.normalization import merge_signals, normalize_topic


def test_normalize_topic_is_unicode_and_punctuation_stable() -> None:
    assert normalize_topic(" ＃Hololive＿Summer!! ") == "hololive summer"
    assert normalize_topic("天音かなた・新衣装") == "天音かなた 新衣装"


def test_duplicate_topics_merge_sources_and_confidence() -> None:
    now = datetime(2026, 10, 5, 12, tzinfo=UTC)
    signals = (
        RawTrendSignal(
            provider="provider-a",
            external_id="a1",
            topic="#Hololive Summer",
            strength=0.8,
            confidence=0.5,
            observed_at=now,
        ),
        RawTrendSignal(
            provider="provider-b",
            external_id="b1",
            topic="hololive_summer",
            strength=0.7,
            confidence=0.6,
            observed_at=now + timedelta(minutes=5),
        ),
    )

    merged = merge_signals(signals, TrendConfig())

    assert len(merged) == 1
    signal = merged[0]
    assert signal.normalized_topic == "hololive summer"
    assert signal.strength == pytest.approx(0.8)
    assert signal.confidence == pytest.approx(0.8)
    assert tuple(source.provider for source in signal.sources) == (
        "provider-a",
        "provider-b",
    )
    assert signal.expires_at == now + timedelta(minutes=5, hours=24)
