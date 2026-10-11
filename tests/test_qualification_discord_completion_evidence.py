from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from artifex.config.models import ArtifexSettings
from artifex.db import Database
from artifex.db.models import CompletionNotificationRow, PackRow
from artifex.qualification.discord_completion_evidence import (
    DiscordCompletionEvidence,
    inspect_discord_completion_evidence,
)


def _config(path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.discord.enabled = True
    settings.discord.allowed_user_ids = (111,)
    settings.storage.database_url = f"sqlite:///{path.as_posix()}"
    return settings


def _notice(
    database: Database, pack_id: str, state: str,
    *,
    message_id: str | None = None,
    verified: bool = False,
) -> None:
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(PackRow(
            id=pack_id, state="finalized", format_type="single_feature",
            payload_json={}, checkpoint_json={}, created_at=now, updated_at=now,
        ))
        session.add(CompletionNotificationRow(
            pack_id=pack_id, state=state,
            discord_message_id=message_id,
            discord_verified_at=now if verified else None,
            created_at=now, updated_at=now,
        ))


def test_disabled_and_missing_db_never_claim_empty_delivery(tmp_path: Path) -> None:
    db_path = tmp_path / "nonexistent" / "artifex.sqlite3"
    settings = _config(db_path)
    missing = inspect_discord_completion_evidence(settings)
    assert missing.status == "missing_database"
    assert missing.total is None
    assert missing.sent_history_verified is None
    assert not db_path.parent.exists()
    assert missing.qualifies_discord_controls is False
    assert missing.operator_read_receipt_verified is False

    settings.discord.enabled = False
    disabled = inspect_discord_completion_evidence(settings)
    assert disabled.status == "disabled"
    assert disabled.total is None
    assert not db_path.parent.exists()


def test_unsupported_backend_and_outdated_schema_are_unknown(
    tmp_path: Path,
) -> None:
    settings = _config(tmp_path / "old.sqlite3")
    settings.storage.database_url = "postgresql://localhost/artifex"
    assert inspect_discord_completion_evidence(settings).status == "unsupported_backend"
    settings.storage.database_url = f"sqlite:///{(tmp_path / 'old.sqlite3').as_posix()}"
    with sqlite3.connect(tmp_path / "old.sqlite3") as db:
        db.execute("CREATE TABLE completion_notifications (state TEXT)")
    result = inspect_discord_completion_evidence(settings)
    assert result.status == "missing_schema"
    assert result.total is None


def test_sqlite_read_only_aggregates_exact_receipt_states(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "live.sqlite3"
    settings = _config(db_path)
    database = Database(settings.storage.database_url)
    database.migrate()
    _notice(database, "pending-1", "pending")
    _notice(database, "sending-1", "sending")
    _notice(database, "uncertain-1", "uncertain")
    _notice(database, "dismissed-1", "dismissed")
    _notice(database, "sent-api-1", "sent", message_id="1133557798")
    _notice(database, "sent-history-1", "sent",
            message_id="1133557799", verified=True)
    _notice(database, "sent-legacy-1", "sent")
    prior_mtime = db_path.stat().st_mtime_ns
    result = inspect_discord_completion_evidence(settings)

    assert result.status == "observed"
    assert result.total == 7
    assert result.pending == 1
    assert result.sending == 1
    assert result.uncertain == 1
    assert result.dismissed == 1
    assert result.sent_api_only == 1
    assert result.sent_history_verified == 1
    assert result.sent_without_receipt == 1
    assert result.inconsistent_rows == 0
    assert "/artifex notices" in result.next_action
    assert result.independent_current_discord_check is False
    assert result.operator_read_receipt_verified is False
    assert result.qualifies_discord_controls is False
    assert db_path.stat().st_mtime_ns == prior_mtime
    assert DiscordCompletionEvidence.model_validate(
        result.model_dump(mode="json")
    ) == result
    database.dispose()


def test_contradictory_saved_receipts_never_count_as_confirmed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corrupt.sqlite3"
    settings = _config(path)
    db = Database(settings.storage.database_url)
    db.migrate()
    _notice(db, "bad-1", "pending", message_id="9912")
    _notice(db, "bad-2", "sent", verified=True)
    _notice(db, "good-1", "sent", message_id="9922", verified=True)
    report = inspect_discord_completion_evidence(settings)
    assert report.status == "inconsistent"
    assert report.total == 3
    assert report.inconsistent_rows == 2
    assert report.sent_history_verified == 1
    assert "do not resend" in report.next_action
    db.dispose()


def test_refuse_symlink_sqlite_evidence_without_following_it(tmp_path: Path) -> None:
    database_file = tmp_path / "real.sqlite3"
    database_file.touch()
    link = tmp_path / "alias.sqlite3"
    try:
        link.symlink_to(database_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this Windows host")
    report = inspect_discord_completion_evidence(_config(link))
    assert report.status == "unavailable"
    assert report.total is None
