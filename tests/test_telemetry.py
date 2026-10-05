from __future__ import annotations

from pathlib import Path

from artifex.db import Database
from artifex.telemetry import EventSeverity, TelemetryRepository


def test_telemetry_retention_bounds_operational_event_growth(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'telemetry.sqlite3').as_posix()}")
    database.migrate()
    telemetry = TelemetryRepository(database)

    for index in range(25):
        telemetry.record(
            "qualification.event",
            EventSeverity.INFO,
            {"index": index},
        )

    removed = telemetry.prune(max_rows=7)
    remaining = telemetry.recent(limit=100)

    assert removed == 18
    assert len(remaining) == 7
    assert {int(item.payload["index"]) for item in remaining} == set(range(18, 25))
    database.dispose()
