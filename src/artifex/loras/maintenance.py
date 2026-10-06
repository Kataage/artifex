from __future__ import annotations

import time

from artifex.loras.automated import LoRAValidationMatrixRunner
from artifex.loras.discovery import DiscoveryResult, LoRADiscovery
from artifex.loras.runs import LoRAValidationRun
from artifex.telemetry import EventSeverity, TelemetryRepository


class LoRADiscoveryMaintenance:
    def __init__(
        self,
        discovery: LoRADiscovery,
        telemetry: TelemetryRepository,
        *,
        roots: tuple,
        interval_seconds: float,
    ) -> None:
        self._discovery = discovery
        self._telemetry = telemetry
        self._roots = roots
        self._interval_seconds = interval_seconds
        self._last_run_monotonic: float | None = None

    async def maintain(self) -> DiscoveryResult | None:
        now = time.monotonic()
        if (
            self._last_run_monotonic is not None
            and now - self._last_run_monotonic < self._interval_seconds
        ):
            return None
        self._last_run_monotonic = now
        try:
            result = self._discovery.scan(self._roots)
        except Exception as exc:  # noqa: BLE001
            self._telemetry.record(
                "loras.discovery_failed",
                EventSeverity.ERROR,
                {"error": str(exc)[:2000]},
            )
            return None

        severity = (
            EventSeverity.WARNING
            if result.failed or result.removed or result.invalidated
            else EventSeverity.INFO
        )
        self._telemetry.record(
            "loras.discovery",
            severity,
            {
                "discovered": len(result.discovered),
                "unchanged": len(result.unchanged),
                "failed": len(result.failed),
                "removed": len(result.removed),
                "invalidated": len(result.invalidated),
            },
        )
        return result


class LoRAValidationMaintenance:
    def __init__(
        self,
        runner: LoRAValidationMatrixRunner,
        telemetry: TelemetryRepository,
        *,
        interval_seconds: float,
        assets_per_cycle: int,
    ) -> None:
        self._runner = runner
        self._telemetry = telemetry
        self._interval_seconds = interval_seconds
        self._assets_per_cycle = assets_per_cycle
        self._last_run_monotonic: float | None = None

    async def maintain(self) -> tuple[LoRAValidationRun, ...] | None:
        now = time.monotonic()
        if (
            self._last_run_monotonic is not None
            and now - self._last_run_monotonic < self._interval_seconds
        ):
            return None
        self._last_run_monotonic = now
        try:
            runs = await self._runner.run_pending(limit=self._assets_per_cycle)
        except Exception as exc:  # noqa: BLE001
            self._telemetry.record(
                "loras.validation_failed",
                EventSeverity.ERROR,
                {"error": str(exc)[:2000]},
            )
            return None

        if runs:
            statuses: dict[str, int] = {}
            for run in runs:
                statuses[run.status] = statuses.get(run.status, 0) + 1
            self._telemetry.record(
                "loras.validation",
                (
                    EventSeverity.WARNING
                    if any(run.status != "promoted" for run in runs)
                    else EventSeverity.INFO
                ),
                {
                    "count": len(runs),
                    "statuses": statuses,
                    "run_ids": [run.id for run in runs],
                },
            )
        return runs
