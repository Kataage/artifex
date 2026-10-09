from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.db import Database
from artifex.db.models import PackRow
from artifex.qualification.collector import QualificationEvidenceCollector
from artifex.qualification.models import (
    REQUIRED_STAGES,
    QualificationSession,
    QualificationStage,
    QualificationStageEvidence,
)
from artifex.telemetry import EventSeverity, TelemetryRepository


class _ReadOnlyQualificationStub:
    """Test paging mechanics without generating files or awarding a stage PASS."""

    def __init__(self, started: datetime) -> None:
        self.session = QualificationSession(
            session_id="keyset-fixture",
            created_at=started,
            updated_at=started,
            hostname="pc-a",
            environment={},
            configuration={},
            workflow={},
            assets=(),
            loras=(),
            doctor_ready=True,
            doctor={},
            stages={
                stage.value: QualificationStageEvidence(stage=stage)
                for stage in REQUIRED_STAGES
            },
        )
        self.seen: list[str] = []

    def load(self, session_id: str) -> QualificationSession:
        assert session_id == self.session.session_id
        return self.session

    def require_collection_candidate(self, session: QualificationSession) -> None:
        assert session is self.session

    def inspect_pack(self, pack_id: str) -> dict[str, object]:
        self.seen.append(pack_id)
        return {
            "character_ids": ["late"] if pack_id == "pack-0298" else [],
            "publication_tiers": [],
        }

    def unattended_run_packs(self, *, since: datetime) -> dict[str, tuple[str, ...]]:
        assert since == self.session.created_at
        return {}

    def validate_candidate(
        self,
        session: QualificationSession,
        stage: QualificationStage,
        pack_ids: tuple[str, ...],
    ) -> None:
        assert session is self.session
        if (
            stage is QualificationStage.SINGLE_CHARACTER
            and pack_ids == ("pack-0298",)
        ):
            return
        raise ValueError("not verified")

    def record(self, *args: object, **kwargs: object) -> None:
        pytest.fail("read-only preview must never mutate a qualification")


def _seed(
    database: Database, *, started: datetime, count: int,
    same_timestamp: bool = False,
) -> None:
    with database.session() as session:
        for index in range(count):
            session.add(PackRow(
                id=f"pack-{index:04d}",
                state="finalized",
                format_type="single",
                payload_json={},
                checkpoint_json={},
                created_at=(
                    started + timedelta(seconds=1)
                    if same_timestamp
                    else started + timedelta(seconds=index + 1)
                ),
                updated_at=started + timedelta(seconds=count + 2),
            ))


@pytest.mark.parametrize("same_timestamp", [False, True])
def test_full_keyset_scan_finds_late_pack_after_first_250(
    tmp_path: Path, same_timestamp: bool,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'keyset.sqlite3').as_posix()}")
    database.migrate()
    started = datetime.now(UTC) - timedelta(minutes=15)
    _seed(database, started=started, count=311, same_timestamp=same_timestamp)
    fake = _ReadOnlyQualificationStub(started)
    collector = QualificationEvidenceCollector(fake, database)  # type: ignore[arg-type]
    report = collector.collect("keyset-fixture", max_packs=17)
    assert report["mode"] == "preview"
    assert report["scan_pages"] == 19
    assert report["finalized_packs_scanned"] == 311
    assert report["scan_truncated"] is False
    assert report["scan_incomplete"] is False
    assert len(fake.seen) == len(set(fake.seen)) == 311
    assert fake.seen == [f"pack-{index:04d}" for index in range(311)]
    entry = next(x for x in report["stages"] if x["stage"] == "single_character")
    assert entry["state"] == "ready"
    assert entry["pack_ids"] == ["pack-0298"]
    assert report["production_qualified"] is False
    database.dispose()


def test_scan_limit_explicitly_reports_omitted_packs(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'limit.sqlite3').as_posix()}")
    database.migrate()
    started = datetime.now(UTC) - timedelta(minutes=15)
    _seed(database, started=started, count=311)
    fake = _ReadOnlyQualificationStub(started)
    collector = QualificationEvidenceCollector(fake, database)  # type: ignore[arg-type]
    incomplete = collector.collect(
        "keyset-fixture", max_packs=17, scan_limit=250,
    )
    assert incomplete["finalized_packs_scanned"] == 250
    assert incomplete["scan_truncated"] is True
    assert incomplete["scan_incomplete"] is True
    assert "pack-0298" not in fake.seen
    entry = next(x for x in incomplete["stages"] if x["stage"] == "single_character")
    assert entry["state"] == "missing"

    fake.seen.clear()
    complete = collector.collect(
        "keyset-fixture", max_packs=31, scan_limit=311,
    )
    assert complete["scan_truncated"] is False
    assert len(fake.seen) == 311
    with pytest.raises(ValueError, match="scan_limit"):
        collector.collect("keyset-fixture", scan_limit=0)
    database.dispose()


def test_daemon_provenance_survives_routine_telemetry_retention(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'prune.sqlite3').as_posix()}")
    database.migrate()
    telemetry = TelemetryRepository(database)
    run_id = "a" * 32
    telemetry.record(
        "daemon.loop_started", EventSeverity.INFO, {"run_id": run_id},
    )
    for index in range(3):
        telemetry.record(
            "daemon.pack_finalized", EventSeverity.INFO,
            {"run_id": run_id, "pack_id": f"pack-{index}"},
        )
    telemetry.record(
        "daemon.loop_stopped", EventSeverity.INFO, {"run_id": run_id},
    )
    for index in range(40):
        telemetry.record(
            "health.component_changed", EventSeverity.INFO,
            {"index": index},
        )
    removed = telemetry.prune(max_rows=5)
    remaining = telemetry.recent(limit=100)
    assert removed == 35
    assert len(remaining) == 10
    assert len([x for x in remaining if x.event_type == "daemon.loop_started"]) == 1
    assert len([x for x in remaining if x.event_type == "daemon.pack_finalized"]) == 3
    assert len([x for x in remaining if x.event_type == "daemon.loop_stopped"]) == 1
    routine = [x for x in remaining if x.event_type == "health.component_changed"]
    assert len(routine) == 5
    assert {int(x.payload["index"]) for x in routine} == set(range(35, 40))
    assert telemetry.prune(max_rows=5) == 0
    database.dispose()
