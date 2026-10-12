"""8-hour soak must observe a live, continuous PC-A daemon, not just HTTP."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.config.models import ArtifexSettings
from artifex.db import Database
from artifex.qualification.daemon_start_observation import (
    observe_active_production_daemon,
)
from artifex.telemetry import EventSeverity, TelemetryRepository

NOW = datetime(2026, 10, 12, 12, tzinfo=UTC)
RUN_ID = "0123456789abcdef" * 2


def _setup(tmp_path: Path) -> tuple[ArtifexSettings, Database, TelemetryRepository]:
    path = tmp_path / "real-telemetry.sqlite3"
    settings = ArtifexSettings()
    settings.storage.database_url = f"sqlite:///{path.as_posix()}"
    settings.operations.heartbeat_stale_seconds = 120
    settings.operations.heartbeat_interval_seconds = 30
    database = Database(settings.storage.database_url)
    database.migrate()
    return settings, database, TelemetryRepository(database)


def test_fresh_heartbeat_and_open_real_loop_required(
    tmp_path: Path,
) -> None:
    settings, db, telemetry = _setup(tmp_path)
    try:
        telemetry.record(
            "daemon.loop_started", EventSeverity.INFO, {"run_id": RUN_ID},
            created_at=NOW - timedelta(minutes=2),
        )
        telemetry.heartbeat(at=NOW - timedelta(seconds=20))
        result = observe_active_production_daemon(settings, now=NOW)
        assert result.ready
        assert result.state == "observed_active_loop"
        assert result.heartbeat_age_seconds == 20
        assert result.loop_fingerprint is not None
        assert len(result.loop_fingerprint) == 64
        assert result.loop_fingerprint not in result.model_dump_json()
        assert RUN_ID not in result.model_dump_json()
        assert result.database_read_only
        assert not result.processes_launched
        assert not result.gpu_jobs_submitted
        assert not result.continuous_run_proven
        assert not result.production_qualified

        telemetry.record(
            "daemon.loop_stopped", EventSeverity.INFO, {"run_id": RUN_ID},
            created_at=NOW - timedelta(seconds=10),
        )
        stopped = observe_active_production_daemon(settings, now=NOW)
        assert stopped.state == "loop_not_active"
        assert not stopped.ready
    finally:
        db.dispose()


def test_manual_heartbeat_without_daemon_loop_cannot_start_soak(
    tmp_path: Path,
) -> None:
    settings, db, telemetry = _setup(tmp_path)
    try:
        telemetry.heartbeat(at=NOW)
        result = observe_active_production_daemon(settings, now=NOW)
        assert result.state == "loop_marker_missing"
        assert not result.ready
    finally:
        db.dispose()


def test_stale_heartbeat_blocks_even_if_latest_event_is_loop_start(
    tmp_path: Path,
) -> None:
    settings, db, telemetry = _setup(tmp_path)
    try:
        telemetry.record(
            "daemon.loop_started", EventSeverity.INFO, {"run_id": RUN_ID},
        )
        telemetry.heartbeat(at=NOW - timedelta(minutes=20))
        result = observe_active_production_daemon(settings, now=NOW)
        assert result.state == "heartbeat_stale"
        assert not result.ready
    finally:
        db.dispose()


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ('{"at": "not-a-date"}', "database_unavailable"),
        ('{"at": "2026-10-12T12:00:00"}', "heartbeat_invalid"),
        ('{"at": "2026-10-12T13:00:00Z"}', "heartbeat_invalid"),
        ('{"something_else": "x"}', "heartbeat_invalid"),
        ('[]', "heartbeat_invalid"),
    ],
)
def test_invalid_worker_clock_or_json_never_passes(
    tmp_path: Path, stored: str, expected: str,
) -> None:
    settings, db, telemetry = _setup(tmp_path)
    try:
        telemetry.heartbeat(at=NOW)
        with sqlite3.connect(
            Path(settings.storage.database_url.removeprefix("sqlite:///"))
        ) as conn:
            conn.execute(
                "UPDATE settings SET value_json = ? "
                "WHERE key = 'runtime.heartbeat'", (stored,),
            )
        result = observe_active_production_daemon(settings, now=NOW)
        assert not result.ready
        assert result.state == expected
    finally:
        db.dispose()


@pytest.mark.parametrize(
    ("event", "payload", "expected"),
    [
        ("daemon.loop_started", {"run_id": "invalid"}, "loop_marker_invalid"),
        ("daemon.loop_started", {"run_id": None}, "loop_marker_invalid"),
        ("daemon.loop_stopped", {"run_id": RUN_ID}, "loop_not_active"),
    ],
)
def test_bad_event_provenance_is_never_accepted(
    tmp_path: Path, event: str, payload: dict[str, str | None], expected: str,
) -> None:
    settings, db, telemetry = _setup(tmp_path)
    try:
        telemetry.heartbeat(at=NOW)
        telemetry.record(event, EventSeverity.INFO, payload)
        report = observe_active_production_daemon(settings, now=NOW)
        assert not report.ready
        assert report.state == expected
        assert RUN_ID not in report.model_dump_json()
    finally:
        db.dispose()


def test_unknown_database_is_never_created(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.sqlite3"
    settings = ArtifexSettings()
    settings.storage.database_url = f"sqlite:///{path.as_posix()}"
    report = observe_active_production_daemon(settings, now=NOW)
    assert report.state == "database_unavailable"
    assert not path.exists()


def test_preflight_includes_daemon_state_and_hides_saved_identifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex.qualification.soak_start_preflight import preflight_soak_start

    settings, db, telemetry = _setup(tmp_path)
    try:
        # Start with the safest case: the production daemon has exited.
        telemetry.record(
            "daemon.loop_stopped", EventSeverity.INFO, {"run_id": RUN_ID},
        )
        telemetry.heartbeat(at=NOW)
        from artifex.qualification import soak_start_preflight as gate
        from artifex.qualification.soak_observer import SoakSample

        monkeypatch.setattr(gate, "sample_soak", lambda s, t: SoakSample(
            observed_at=NOW, elapsed_seconds=0,
            llm_ok=True, comfyui_ok=True, renderer_ok=True,
            gpu_used_mib=1000, ram_used_mib=2000, free_disk_gib=40,
            telemetry_rows=2, severe_events=0, finalized_packs=0,
            asset_sha256={
                "production_checkpoint": "a" * 64,
                "refiner_checkpoint": "b" * 64,
                "vae": "c" * 64, "upscale_model": "d" * 64,
            },
        ))
        settings.qualification.require_renderer_owner_observation = False
        # Freeze time for the DB observer only; pure sample still has a
        # sufficiently recent synthetic timestamp in this fixture.
        monkeypatch.setattr(gate, "observe_active_production_daemon",
            lambda s: observe_active_production_daemon(s, now=NOW))
        result = preflight_soak_start(settings)
        assert result.daemon_state == "loop_not_active"
        assert not result.sample_attempted
        assert result.sample_checked_utc is None
        assert "daemon_not_live" in result.issues
        assert not result.ready
        assert not result.daemon_start_observed
        assert RUN_ID not in result.model_dump_json()
        assert not result.production_qualified
    finally:
        db.dispose()
