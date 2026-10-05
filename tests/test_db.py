from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import SettingRow


def test_database_session_commits(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "artifex.sqlite3"
    db = Database(f"sqlite:///{db_path.as_posix()}")
    db.migrate()

    with db.session() as session:
        session.add(
            SettingRow(
                key="agent.state",
                value_json="running",
                updated_at=datetime.now(UTC),
            )
        )

    with db.session() as session:
        value = session.scalar(
            select(SettingRow.value_json).where(SettingRow.key == "agent.state")
        )

    assert value == "running"
    db.dispose()
