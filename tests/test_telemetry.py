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


def test_unattended_pack_proof_is_atomic_and_unique_across_restarts(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'proof.sqlite3').as_posix()}")
    database.migrate()
    first = TelemetryRepository(database)
    assert first.record_daemon_pack_finalized_once("pack-1", "run-a") is True
    assert first.record_daemon_pack_finalized_once("pack-1", "run-a") is False

    restarted = TelemetryRepository(database)
    assert restarted.record_daemon_pack_finalized_once("pack-1", "run-a") is False
    assert restarted.record_daemon_pack_finalized_once("pack-1", "run-b") is True
    proofs = [event for event in restarted.recent()
              if event.event_type == "daemon.pack_finalized"]
    assert len(proofs) == 2
    assert {(p.payload["run_id"], p.payload["pack_id"]) for p in proofs} == {
        ("run-a", "pack-1"), ("run-b", "pack-1")
    }
    restarted.prune(max_rows=1)
    assert len([event for event in restarted.recent()
                if event.event_type == "daemon.pack_finalized"]) == 2
    database.dispose()


def test_discord_receipts_survive_bounded_telemetry_pruning(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'discord-proof.sqlite3').as_posix()}")
    database.migrate()
    telemetry = TelemetryRepository(database)
    event = telemetry.record(
        "discord.interaction_completed",
        EventSeverity.INFO,
        {
            "command": "status", "interaction_id": "123",
            "guild_id": 11, "channel_id": 22, "user_id": 33,
            "role_ids": [], "response_ok": True,
        },
    )
    for index in range(20):
        telemetry.record("routine.health", EventSeverity.INFO, {"index": index})
    assert telemetry.prune(max_rows=2) == 18
    assert telemetry.require(event.id).payload["interaction_id"] == "123"
    ordinary = [item for item in telemetry.recent(limit=100)
                if item.event_type == "routine.health"]
    assert len(ordinary) == 2
    database.dispose()
