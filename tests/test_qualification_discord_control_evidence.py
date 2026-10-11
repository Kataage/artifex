from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.config.models import ArtifexSettings
from artifex.db import Database
from artifex.qualification.discord_control_evidence import (
    DiscordControlEvidence,
    inspect_discord_control_evidence,
)
from artifex.telemetry import EventSeverity, TelemetryRepository


def _settings(path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.discord.enabled = True
    settings.discord.guild_id = 123
    settings.discord.channel_id = 456
    settings.discord.allowed_user_ids = (789,)
    settings.discord.allowed_role_ids = (555,)
    settings.storage.database_url = f"sqlite:///{path.as_posix()}"
    return settings


def _event(
    telemetry: TelemetryRepository, command: str, identity: str,
    *,
    when: datetime, user: int = 789, roles: list[int] | None = None,
    channel: int = 456, response_ok: bool = True,
) -> None:
    telemetry.record(
        "discord.interaction_completed", EventSeverity.INFO,
        {
            "command": command,
            "interaction_id": identity,
            "guild_id": 123,
            "channel_id": channel,
            "user_id": user,
            "role_ids": roles if roles is not None else [],
            "response_ok": response_ok,
        },
        created_at=when,
    )


def test_no_session_or_missing_database_is_unknown_not_zero(tmp_path: Path) -> None:
    path = tmp_path / "uncreated" / "store.sqlite3"
    config = _settings(path)
    now = datetime.now(UTC)
    no_session = inspect_discord_control_evidence(
        config, session_id=None, session_started=None, now=now,
    )
    assert no_session.status == "no_session"
    assert no_session.matched_distinct_interactions is None
    assert no_session.missing_commands == ()
    missing = inspect_discord_control_evidence(
        config, session_id="trial", session_started=now, now=now,
    )
    assert missing.status == "missing_database"
    assert missing.examined_recent_events is None
    assert not path.parent.exists()

    config.discord.enabled = False
    assert inspect_discord_control_evidence(
        config, session_id="trial", session_started=now, now=now,
    ).status == "disabled"


def test_missing_config_and_schema_and_symlink_are_unknown(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    db_file = tmp_path / "old.sqlite3"
    config = _settings(db_file)
    config.discord.guild_id = None
    assert inspect_discord_control_evidence(
        config, session_id="trial", session_started=now, now=now,
    ).status == "missing_config"
    config.discord.guild_id = 123
    with sqlite3.connect(db_file) as db:
        db.execute("CREATE TABLE agent_events (id INTEGER, created_at TEXT)")
    result = inspect_discord_control_evidence(
        config, session_id="trial", session_started=now, now=now,
    )
    assert result.status == "missing_schema"
    assert result.observed_commands == ()

    alias = tmp_path / "alias.sqlite3"
    try:
        alias.symlink_to(db_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this Windows host")
    assert inspect_discord_control_evidence(
        _settings(alias), session_id="trial", session_started=now, now=now,
    ).status == "unavailable"


def test_session_scoped_distinct_authorized_six_controls_fail_closed(
    tmp_path: Path,
) -> None:
    config = _settings(tmp_path / "telemetry.sqlite3")
    db = Database(config.storage.database_url)
    db.migrate()
    telemetry = TelemetryRepository(db)
    now = datetime.now(UTC)
    start = now - timedelta(minutes=20)
    before = start - timedelta(minutes=1)
    _event(telemetry, "approve", "100", when=before)
    _event(telemetry, "status", "101", when=now)
    _event(telemetry, "pause", "102", when=now, user=999, roles=[555])
    _event(telemetry, "resume", "103", when=now, channel=666)
    _event(telemetry, "approve", "101", when=now)  # duplicate Discord interaction
    _event(telemetry, "reject", "104", when=now, user=999)
    _event(telemetry, "retry", "105", when=now, response_ok=False)
    _event(telemetry, "retry", "106", when=now, roles=[True])
    mtime = (tmp_path / "telemetry.sqlite3").stat().st_mtime_ns
    evidence = inspect_discord_control_evidence(
        config, session_id="trial-1", session_started=start, now=now,
    )
    assert evidence.status == "incomplete"
    assert evidence.observed_commands == ("pause", "status")
    assert evidence.missing_commands == ("approve", "reject", "resume", "retry")
    assert evidence.matched_distinct_interactions == 2
    assert evidence.examined_recent_events == 8
    assert evidence.qualifies_discord_controls is False
    assert evidence.recorded_stage_revalidated is False
    assert evidence.independent_discord_response_check is False
    assert "approve, reject, resume, retry" in evidence.next_action
    assert (tmp_path / "telemetry.sqlite3").stat().st_mtime_ns == mtime

    for n, command in enumerate(("approve", "reject", "resume", "retry"), 200):
        _event(telemetry, command, str(n), when=now)
    full = inspect_discord_control_evidence(
        config, session_id="trial-1", session_started=start, now=now,
    )
    assert full.status == "observed"
    assert len(full.observed_commands) == 6
    assert full.missing_commands == ()
    assert full.matched_distinct_interactions == 6
    assert full.qualifies_discord_controls is False
    assert "qualify collect" in full.next_action
    assert DiscordControlEvidence.model_validate(
        full.model_dump(mode="json"),
    ) == full
    db.dispose()


def test_malformed_interactions_and_inaccessible_non_sqlite_fail_closed(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    config = _settings(tmp_path / "events.sqlite3")
    with sqlite3.connect(tmp_path / "events.sqlite3") as db:
        db.execute(
            "CREATE TABLE agent_events "
            "(id INTEGER, event_type TEXT, created_at TEXT, payload_json TEXT)"
        )
        db.execute(
            "INSERT INTO agent_events VALUES (1, ?, ?, ?)",
            ("discord.interaction_completed", now.isoformat(), "{not-json"),
        )
    observed = inspect_discord_control_evidence(
        config, session_id="trial", session_started=now-timedelta(minutes=2),
        now=now,
    )
    assert observed.status == "incomplete"
    assert observed.observed_commands == ()
    assert len(observed.missing_commands) == 6
    config.storage.database_url = "postgresql://localhost/artifex"
    assert inspect_discord_control_evidence(
        config, session_id="trial", session_started=now, now=now,
    ).status == "unsupported_backend"
