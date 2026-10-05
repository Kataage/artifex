from __future__ import annotations

from datetime import datetime
from typing import Protocol

from artifex.trends.models import RawTrendSignal


class TrendProviderError(RuntimeError):
    pass


class TrendProvider(Protocol):
    name: str

    async def collect(self, *, as_of: datetime) -> tuple[RawTrendSignal, ...]: ...
