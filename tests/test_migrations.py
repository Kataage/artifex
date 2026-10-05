from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect

from artifex.db.migrations import current_revision, upgrade_database

EXPECTED_TABLES = {
    "agent_events",
    "characters",
    "concepts",
    "evaluations",
    "generation_attempts",
    "loras",
    "packs",
    "policy_decisions",
    "research_briefs",
    "research_evidence",
    "research_runs",
    "review_queue",
    "scenes",
    "series",
    "settings",
    "trend_signals",
}


def test_initial_migration_creates_expected_schema(tmp_path: Path) -> None:
    path = tmp_path / "schema.sqlite3"
    url = f"sqlite:///{path.as_posix()}"

    upgrade_database(url)

    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert EXPECTED_TABLES <= tables
    assert current_revision(url) == "0002_research"
