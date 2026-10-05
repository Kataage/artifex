from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from artifex.config.models import DiscordConfig
from artifex.db import Database
from artifex.db.models import AgentEventRow, PackRow
from artifex.discord.bot import token_from_environment
from artifex.discord.summary import DailySummaryBuilder
from artifex.domain import PackState


def test_enabled_discord_requires_operator_allowlist() -> None:
    with pytest.raises(ValueError, match="allowed_user_ids"):
        DiscordConfig(enabled=True)


def test_token_is_read_only_from_configured_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DiscordConfig(token_env="ARTIFEX_TEST_DISCORD_TOKEN")
    monkeypatch.delenv("ARTIFEX_TEST_DISCORD_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="environment variable is missing"):
        token_from_environment(config)

    monkeypatch.setenv("ARTIFEX_TEST_DISCORD_TOKEN", "super-secret")
    assert token_from_environment(config) == "super-secret"


def test_daily_summary_counts_recent_operational_events(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'summary.sqlite3').as_posix()}")
    database.migrate()
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            PackRow(
                id="done",
                state=PackState.FINALIZED.value,
                format_type="single_feature",
                payload_json={},
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            AgentEventRow(
                event_type="backend.failed",
                severity="error",
                payload_json={},
                created_at=now,
            )
        )

    summary = DailySummaryBuilder(database).build(now=now)

    assert "finalized=1" in summary
    assert "errors=1" in summary
    database.dispose()
