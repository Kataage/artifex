"""Best-effort PC-A qualification evidence collection during native daemon operation.

This is a passive scanner of already-finalized Packs and authenticated
Discord telemetry. It never creates Pack work, launches a GPU process, or
authorizes production. The strict QualificationService validators alone
decide whether a pending stage may be recorded.
"""
from __future__ import annotations

import asyncio
import logging
import time

from artifex.db import Database
from artifex.qualification.collector import QualificationEvidenceCollector
from artifex.qualification.service import QualificationService
from artifex.telemetry import EventSeverity, TelemetryRepository


_LOG = logging.getLogger(__name__)


class QualificationEvidenceMaintenance:
    """Low-frequency asynchronous, bounded, failure-isolated evidence ingestion."""

    def __init__(
        self,
        service: QualificationService,
        database: Database,
        telemetry: TelemetryRepository,
        *,
        interval_seconds: float = 900,
        scan_limit: int = 10000,
    ) -> None:
        if interval_seconds < 60:
            raise ValueError("automatic evidence interval must be at least 60 seconds")
        if not 1 <= scan_limit <= 100000:
            raise ValueError("automatic evidence scan_limit must be 1..100000")
        self._service = service
        self._database = database
        self._telemetry = telemetry
        self._interval_seconds = interval_seconds
        self._scan_limit = scan_limit
        self._last_run_monotonic: float | None = None

    def _collect(self) -> dict[str, object] | None:
        session_id = self._service.active_auto_collection_session_id()
        if session_id is None:
            return None
        # Rechecks doctor, same-host baseline and production configuration;
        # only authoritative, persisted Packs/Discord evidence can become PASS.
        return QualificationEvidenceCollector(
            self._service, self._database,
        ).collect(session_id, apply=True, scan_limit=self._scan_limit)

    async def maintain(self) -> dict[str, object] | None:
        now = time.monotonic()
        if (
            self._last_run_monotonic is not None
            and now - self._last_run_monotonic < self._interval_seconds
        ):
            return None
        self._last_run_monotonic = now
        try:
            # Large archive evidence hashes cannot block the scheduler loop.
            report = await asyncio.to_thread(self._collect)
        except Exception as exc:  # noqa: BLE001 - background scanner is noncritical
            # Do not report file paths, tokens or arbitrary exception strings.
            try:
                self._telemetry.record(
                    "qualification.auto_collect_failed", EventSeverity.WARNING,
                    {"error_type": type(exc).__name__},
                )
            except Exception:  # noqa: BLE001 - never take down production
                _LOG.warning("Qualification failure telemetry could not be stored")
            return None
        if report is not None:
            stages = report.get("stages", [])
            recorded: list[str] = []
            if isinstance(stages, list):
                recorded = [
                    str(item["stage"])
                    for item in stages
                    if isinstance(item, dict) and item.get("state") == "recorded"
                    and isinstance(item.get("stage"), str)
                ]
            if recorded:
                try:
                    self._telemetry.record(
                        "qualification.auto_collected", EventSeverity.INFO,
                        {
                            "session_id": report.get("session_id"),
                            "stages": recorded,
                            "scan_incomplete": report.get("scan_incomplete") is True,
                        },
                    )
                except Exception:  # noqa: BLE001 - no telemetry failure may stop daemon
                    _LOG.warning("Qualification success telemetry could not be stored")
        return report
