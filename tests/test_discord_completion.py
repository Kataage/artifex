from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from artifex.db import Database
from artifex.db.models import CompletionNotificationRow, PackRow
from artifex.domain import PackState
from artifex.operations.completion_notifications import CompletionNotificationMaintenance
from artifex.runtime import RuntimeStore
from artifex.telemetry import TelemetryRepository


class FakeCompletionSink:
    def __init__(self) -> None:
        self.ready = True
        self.deliver = True
        self.fail_after_send: BaseException | None = None
        self.messages: list[object] = []
        self.history_results: dict[str, str | None] = {}
        self.history_fail: Exception | None = None
        self.lookups: list[str] = []

    def is_ready(self) -> bool:
        return self.ready

    async def notify(self, notification: object) -> bool:
        if not self.deliver:
            return False
        self.messages.append(notification)
        if self.fail_after_send is not None:
            raise self.fail_after_send
        return True

    async def find_completion_message(self, pack_id: str) -> str | None:
        self.lookups.append(pack_id)
        if self.history_fail is not None:
            raise self.history_fail
        return self.history_results.get(pack_id)


def _setup(tmp_path: Path) -> tuple[Database, TelemetryRepository]:
    database = Database(
        f"sqlite:///{(tmp_path / 'completion.sqlite3').as_posix()}"
    )
    database.migrate()
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            PackRow(
                id="pack-1",
                state=PackState.EVALUATING.value,
                format_type="single_feature",
                payload_json={},
                checkpoint_json={
                    "archive_complete": True,
                    "archive_manifest_sha256": "a" * 64,
                },
                created_at=now,
                updated_at=now,
            )
        )
    RuntimeStore(database).transition_pack("pack-1", PackState.FINALIZED)
    return database, TelemetryRepository(database)


def _state(database: Database) -> str:
    with database.session() as session:
        row = session.get(CompletionNotificationRow, "pack-1")
        assert row is not None
        return row.state


@pytest.mark.asyncio
async def test_finalization_atomically_enqueues_and_dispatches_once(
    tmp_path: Path,
) -> None:
    db, telemetry = _setup(tmp_path)
    assert _state(db) == "pending"
    sink = FakeCompletionSink()
    maintenance = CompletionNotificationMaintenance(db, telemetry, sink)

    assert await maintenance.maintain() == 1
    assert _state(db) == "sent"
    assert len(sink.messages) == 1
    assert "pack-1" in sink.messages[0].body  # type: ignore[attr-defined]

    restarted = CompletionNotificationMaintenance(db, telemetry, sink)
    assert await restarted.maintain() == 0
    assert len(sink.messages) == 1
    assert len([
        e for e in telemetry.recent()
        if e.event_type == "discord.completion_sent"
    ]) == 1
    db.dispose()


@pytest.mark.asyncio
async def test_offline_discord_does_not_claim_notice(tmp_path: Path) -> None:
    db, telemetry = _setup(tmp_path)
    sink = FakeCompletionSink()
    sink.ready = False
    maintenance = CompletionNotificationMaintenance(db, telemetry, sink)

    assert await maintenance.maintain() == 0
    assert _state(db) == "pending"
    sink.ready = True
    assert await maintenance.maintain() == 1
    assert _state(db) == "sent"
    db.dispose()


@pytest.mark.asyncio
async def test_explicitly_unsent_notice_may_retry(tmp_path: Path) -> None:
    db, telemetry = _setup(tmp_path)
    sink = FakeCompletionSink()
    sink.deliver = False
    maintenance = CompletionNotificationMaintenance(db, telemetry, sink)

    assert await maintenance.maintain() == 0
    assert _state(db) == "pending"
    assert sink.messages == []
    sink.deliver = True
    assert await maintenance.maintain() == 1
    assert len(sink.messages) == 1
    db.dispose()


@pytest.mark.asyncio
async def test_process_death_after_remote_send_never_automatically_resends(
    tmp_path: Path,
) -> None:
    db, telemetry = _setup(tmp_path)
    sink = FakeCompletionSink()
    sink.fail_after_send = SystemExit("PC-A hard exited after Discord send")
    maintenance = CompletionNotificationMaintenance(db, telemetry, sink)

    with pytest.raises(SystemExit, match="hard exited"):
        await maintenance.maintain()
    assert _state(db) == "sending"
    assert len(sink.messages) == 1

    sink.fail_after_send = None
    restarted = CompletionNotificationMaintenance(db, telemetry, sink)
    assert _state(db) == "uncertain"
    assert await restarted.maintain() == 0
    assert len(sink.messages) == 1
    assert any(
        e.event_type == "discord.completion_uncertain"
        for e in telemetry.recent()
    )
    db.dispose()


@pytest.mark.asyncio
async def test_ambiguous_remote_failure_is_quarantined(tmp_path: Path) -> None:
    db, telemetry = _setup(tmp_path)
    sink = FakeCompletionSink()
    sink.fail_after_send = TimeoutError("response lost")
    maintenance = CompletionNotificationMaintenance(db, telemetry, sink)

    assert await maintenance.maintain() == 0
    assert _state(db) == "uncertain"
    assert len(sink.messages) == 1
    assert await maintenance.maintain() == 0
    assert len(sink.messages) == 1
    db.dispose()


@pytest.mark.asyncio
async def test_ambiguous_discord_reply_reconciles_existing_post_without_resend(
    tmp_path: Path,
) -> None:
    db, telemetry = _setup(tmp_path)
    sink = FakeCompletionSink()
    sink.fail_after_send = TimeoutError("Discord accepted, response lost")
    worker = CompletionNotificationMaintenance(
        db, telemetry, sink, reconcile_interval_seconds=0,
    )
    assert await worker.maintain() == 0
    assert _state(db) == "uncertain"
    assert len(sink.messages) == 1
    assert "Artifex-Completion-ID: pack-1" in sink.messages[0].body  # type: ignore[attr-defined]

    sink.fail_after_send = None
    sink.history_results["pack-1"] = "1133557799"
    # Simulate new PC-A worker after restart.
    restarted = CompletionNotificationMaintenance(
        db, telemetry, sink, reconcile_interval_seconds=0,
    )
    assert await restarted.maintain() == 0
    assert _state(db) == "sent"
    assert len(sink.messages) == 1
    with db.session() as session:
        row = session.get(CompletionNotificationRow, "pack-1")
        assert row is not None
        assert row.discord_message_id == "1133557799"
    assert any(
        event.event_type == "discord.completion_reconciled"
        and event.payload["discord_message_id"] == "1133557799"
        for event in telemetry.recent()
    )
    db.dispose()


@pytest.mark.asyncio
async def test_inconclusive_history_never_authorizes_resend(
    tmp_path: Path,
) -> None:
    db, telemetry = _setup(tmp_path)
    sink = FakeCompletionSink()
    sink.fail_after_send = TimeoutError("unknown status")
    worker = CompletionNotificationMaintenance(
        db, telemetry, sink, reconcile_interval_seconds=0,
    )
    assert await worker.maintain() == 0
    assert _state(db) == "uncertain"
    sink.fail_after_send = None

    for _ in range(3):
        assert await worker.maintain() == 0
    assert len(sink.messages) == 1
    assert _state(db) == "uncertain"
    assert sink.lookups
    db.dispose()


@pytest.mark.asyncio
async def test_inaccessible_history_remains_uncertain_and_emits_diagnostic(
    tmp_path: Path,
) -> None:
    db, telemetry = _setup(tmp_path)
    sink = FakeCompletionSink()
    sink.fail_after_send = TimeoutError("response lost")
    worker = CompletionNotificationMaintenance(
        db, telemetry, sink, reconcile_interval_seconds=0,
    )
    await worker.maintain()
    sink.fail_after_send = None
    sink.history_fail = PermissionError("missing read message history permission")

    assert await worker.maintain() == 0
    assert _state(db) == "uncertain"
    assert len(sink.messages) == 1
    assert any(
        event.event_type == "discord.completion_reconcile_failed"
        and event.payload["reason"] == "PermissionError"
        for event in telemetry.recent()
    )
    db.dispose()


@pytest.mark.asyncio
async def test_invalid_history_message_id_cannot_mark_sent(
    tmp_path: Path,
) -> None:
    db, telemetry = _setup(tmp_path)
    sink = FakeCompletionSink()
    sink.fail_after_send = TimeoutError("response lost")
    worker = CompletionNotificationMaintenance(
        db, telemetry, sink, reconcile_interval_seconds=0,
    )
    await worker.maintain()
    sink.fail_after_send = None
    sink.history_results["pack-1"] = "not-a-discord-snowflake"

    assert await worker.maintain() == 0
    assert _state(db) == "uncertain"
    assert len(sink.messages) == 1
    db.dispose()
