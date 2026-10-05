from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from artifex.telemetry import EventSeverity, TelemetryRepository
from artifex.trends.service import SignalIngestionService, SignalRefreshReport


class MaintenanceTask(Protocol):
    async def maintain(self) -> object: ...


class MaintenanceGroup:
    def __init__(self, tasks: Sequence[MaintenanceTask]) -> None:
        self._tasks = tuple(tasks)

    async def maintain(self) -> tuple[object, ...]:
        results: list[object] = []
        for task in self._tasks:
            results.append(await task.maintain())
        return tuple(results)


class SignalIngestionMaintenance:
    """Run signal refresh on cadence without controlling production health state."""

    def __init__(
        self,
        service: SignalIngestionService,
        telemetry: TelemetryRepository,
        *,
        interval_seconds: float,
    ) -> None:
        self._service = service
        self._telemetry = telemetry
        self._interval_seconds = interval_seconds
        self._last_run_monotonic: float | None = None

    async def maintain(self) -> SignalRefreshReport | None:
        now = time.monotonic()
        if (
            self._last_run_monotonic is not None
            and now - self._last_run_monotonic < self._interval_seconds
        ):
            return None

        self._last_run_monotonic = now
        try:
            return await self._service.refresh(as_of=datetime.now(UTC))
        except Exception as exc:  # noqa: BLE001
            # Signal freshness is helpful but never a production-critical backend.
            self._telemetry.record(
                "signals.refresh_failed",
                EventSeverity.WARNING,
                {"error": str(exc)[:1000]},
            )
            return None
