from __future__ import annotations

from pathlib import Path

import pytest

from artifex.db import Database
from artifex.operations import MaintenanceGroup, SignalIngestionMaintenance
from artifex.telemetry import TelemetryRepository


class WorkingSignalService:
    def __init__(self) -> None:
        self.calls = 0

    async def refresh(self, *, as_of: object = None) -> object:
        del as_of
        self.calls += 1
        return {"ok": True}


class BrokenSignalService:
    async def refresh(self, *, as_of: object = None) -> object:
        del as_of
        raise RuntimeError("network stack unavailable")


class OtherMaintenance:
    def __init__(self) -> None:
        self.calls = 0

    async def maintain(self) -> object:
        self.calls += 1
        return "health-ok"


@pytest.mark.asyncio
async def test_signal_maintenance_is_cadenced_and_composable(
    tmp_path: Path,
) -> None:
    database = Database(
        f"sqlite:///{(tmp_path / 'maintenance.sqlite3').as_posix()}"
    )
    database.migrate()
    telemetry = TelemetryRepository(database)
    service = WorkingSignalService()
    other = OtherMaintenance()
    signal = SignalIngestionMaintenance(
        service,  # type: ignore[arg-type]
        telemetry,
        interval_seconds=3600,
    )
    group = MaintenanceGroup((other, signal))

    first = await group.maintain()
    second = await group.maintain()

    assert first[0] == "health-ok"
    assert service.calls == 1
    assert other.calls == 2
    assert second[1] is None
    database.dispose()


@pytest.mark.asyncio
async def test_signal_maintenance_failure_does_not_escape_to_daemon(
    tmp_path: Path,
) -> None:
    database = Database(
        f"sqlite:///{(tmp_path / 'failure.sqlite3').as_posix()}"
    )
    database.migrate()
    telemetry = TelemetryRepository(database)
    signal = SignalIngestionMaintenance(
        BrokenSignalService(),  # type: ignore[arg-type]
        telemetry,
        interval_seconds=1,
    )

    result = await signal.maintain()

    assert result is None
    assert any(
        event.event_type == "signals.refresh_failed"
        for event in telemetry.recent(limit=10)
    )
    database.dispose()
