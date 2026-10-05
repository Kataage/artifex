from __future__ import annotations

from pathlib import Path

import pytest

from artifex.db import Database
from artifex.series import SeriesRepository, SeriesStatus


def test_series_records_episode_continuity_and_hooks(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'series.sqlite3').as_posix()}")
    database.migrate()
    repository = SeriesRepository(database, id_factory=lambda: "series-1")
    series = repository.create(
        title="Angel at the Window",
        character_ids=("char-a",),
        preferred_format="single_feature",
        continuity_state={"outfit": "white dress"},
        unresolved_hooks=("letter",),
    )

    assert series.current_episode == 0
    updated = repository.record_completed_pack(
        "series-1",
        pack_id="pack-1",
        episode_number=1,
        continuity_updates={"location": "window room"},
        hooks_resolved=("letter",),
        hooks_added=("feather",),
    )

    assert updated.current_episode == 1
    assert updated.prior_pack_ids == ("pack-1",)
    assert updated.continuity_state == {
        "outfit": "white dress",
        "location": "window room",
    }
    assert updated.unresolved_hooks == ("feather",)

    reloaded = SeriesRepository(database).require("series-1")
    assert reloaded == updated
    database.dispose()


def test_series_rejects_episode_skip(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'skip.sqlite3').as_posix()}")
    database.migrate()
    repository = SeriesRepository(database, id_factory=lambda: "series-1")
    repository.create(title="Series", character_ids=("char-a",))

    with pytest.raises(ValueError, match="expected episode 1"):
        repository.record_completed_pack(
            "series-1",
            pack_id="pack-2",
            episode_number=2,
        )
    database.dispose()


def test_series_status_persists(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'status.sqlite3').as_posix()}")
    database.migrate()
    repository = SeriesRepository(database, id_factory=lambda: "series-1")
    repository.create(title="Series", character_ids=("char-a",))
    paused = repository.set_status("series-1", SeriesStatus.PAUSED)
    assert paused.status is SeriesStatus.PAUSED
    database.dispose()
