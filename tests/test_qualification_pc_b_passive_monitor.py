"""Offline tests for the daemon's passive PC-B monitor and stale snapshots."""
from __future__ import annotations

import asyncio
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

import artifex.qualification.pc_b_monitor as monitor
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.models import (
    REQUIRED_STAGES,
    QualificationSession,
    QualificationStageEvidence,
)
from artifex.qualification.pc_b_monitor import (
    PCBEvidenceMonitorSnapshot,
    PCBEvidenceRefreshMaintenance,
    read_pc_b_evidence_monitor,
)

NOW = datetime(2026, 10, 12, 1, 0, tzinfo=UTC)
SESSION = "20261012T010000Z-12345678"


def _settings(root: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.qualification.evidence_dir = root
    settings.render_nodes.primary = "gpu-b"
    settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://192.0.2.10:8188",
        attestation_url="http://192.0.2.10:8190",
    )
    return settings


def _session(root: Path) -> None:
    path = root / SESSION
    path.mkdir(parents=True)
    data = QualificationSession(
        session_id=SESSION, created_at=NOW, updated_at=NOW,
        hostname=socket.gethostname(),
        environment={}, configuration={}, workflow={}, assets=(), loras=(),
        doctor_ready=True, doctor={},
        stages={
            item.value: QualificationStageEvidence(stage=item)
            for item in REQUIRED_STAGES
        },
    )
    (path / "qualification.json").write_text(
        data.model_dump_json(), encoding="utf-8",
    )


def _status(status: str) -> object:
    return type("Observation", (), {"status": status})()


def test_no_primary_no_session_and_missing_snapshots_never_become_zero(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "evidence")
    assert read_pc_b_evidence_monitor(
        settings, session_id=SESSION, now=NOW, max_age_seconds=1800,
    ).status == "missing"
    assert not settings.qualification.evidence_dir.exists()
    assert read_pc_b_evidence_monitor(
        settings, session_id=None, now=NOW, max_age_seconds=1800,
    ).status == "no_session"

    settings.render_nodes.nodes.clear()
    settings.render_nodes.primary = None
    assert read_pc_b_evidence_monitor(
        settings, session_id=SESSION, now=NOW, max_age_seconds=1800,
    ).status == "disabled"


def test_monitored_identity_is_historical_only_and_never_qualifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "evidence")
    _session(settings.qualification.evidence_dir)
    calls: list[str] = []

    def owner(*args: object, **kwargs: object) -> object:
        calls.append("owner")
        return _status("correlated_read_only")

    def survival(*args: object, **kwargs: object) -> object:
        calls.append("survival")
        return _status("replayed_and_live_identity_matched")

    monkeypatch.setattr(monitor, "correlate_live_pc_b_owner_report", owner)
    monkeypatch.setattr(monitor, "review_live_pc_b_survival_evidence", survival)
    telemetry = Mock()
    maint = PCBEvidenceRefreshMaintenance(settings, telemetry, interval_seconds=900)
    snapshot = asyncio.run(maint.maintain())
    assert snapshot is not None
    assert calls == ["owner", "survival"]
    assert snapshot.owner_status == "correlated_read_only"
    assert snapshot.survival_status == "replayed_and_live_identity_matched"
    assert snapshot.actual_gpu_qualified is False
    assert snapshot.historical_event_authenticated is False
    assert snapshot.issue_93_closed is False
    assert snapshot.production_qualified is False
    assert snapshot.services_mutated is False
    assert snapshot.gpu_jobs_submitted is False
    path = settings.qualification.evidence_dir / "pc-b-passive-monitor.json"
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp"))
    assert not path.parent.joinpath("qualification.json").exists()

    view = read_pc_b_evidence_monitor(
        settings, session_id=SESSION,
        now=snapshot.captured_utc + timedelta(seconds=60),
        max_age_seconds=1800,
    )
    assert view.status == "historical_observation"
    assert view.owner_status == snapshot.owner_status
    assert view.survival_status == snapshot.survival_status
    assert view.age_seconds == 60
    assert view.live_rechecked_by_this_overview is False
    assert view.historical_event_authenticated is False
    assert view.stage_pass_registered is False
    assert view.production_qualified is False
    assert "NOT current owner checks" in view.next_action
    assert asyncio.run(maint.maintain()) is None  # throttled, no network call
    assert calls == ["owner", "survival"]
    telemetry.record.assert_not_called()


def test_probe_failure_does_not_hide_other_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "evidence")
    _session(settings.qualification.evidence_dir)

    def fail(*args: object, **kwargs: object) -> object:
        raise OSError("secretBearerTokenNeverPersistThis")

    monkeypatch.setattr(monitor, "correlate_live_pc_b_owner_report", fail)
    monkeypatch.setattr(
        monitor, "review_live_pc_b_survival_evidence",
        lambda *a, **k: _status("replayed_historical_identity"),
    )
    maint = PCBEvidenceRefreshMaintenance(settings, Mock())
    sample = asyncio.run(maint.maintain())
    assert sample is not None
    assert sample.owner_check_failed is True
    assert sample.owner_status == "probe_failed"
    assert sample.survival_check_failed is False
    assert sample.survival_status == "replayed_historical_identity"
    payload = (
        settings.qualification.evidence_dir / "pc-b-passive-monitor.json"
    ).read_text(encoding="utf-8")
    assert "secretBearerTokenNeverPersistThis" not in payload
    assert "http://192.0.2.10" not in payload
    assert "probe_failed" in payload


def test_saved_observation_invalidates_on_age_clock_and_config_or_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "evidence")
    _session(settings.qualification.evidence_dir)
    monkeypatch.setattr(
        monitor, "correlate_live_pc_b_owner_report",
        lambda *a, **k: _status("blocked"),
    )
    monkeypatch.setattr(
        monitor, "review_live_pc_b_survival_evidence",
        lambda *a, **k: _status("remote_unavailable"),
    )
    sample = asyncio.run(PCBEvidenceRefreshMaintenance(settings, Mock()).maintain())
    assert sample is not None
    def view(**kwargs: object) -> monitor.PCBEvidenceMonitorView:
        return read_pc_b_evidence_monitor(
            settings, session_id=SESSION, now=sample.captured_utc,
            max_age_seconds=1800, **kwargs,
        )

    assert read_pc_b_evidence_monitor(
        settings, session_id=SESSION,
        now=sample.captured_utc + timedelta(seconds=1801),
        max_age_seconds=1800,
    ).status == "stale"
    assert read_pc_b_evidence_monitor(
        settings, session_id=SESSION,
        now=sample.captured_utc - timedelta(seconds=31),
        max_age_seconds=1800,
    ).status == "stale"
    assert read_pc_b_evidence_monitor(
        settings, session_id="different-session",
        now=sample.captured_utc, max_age_seconds=1800,
    ).status == "context_mismatch"
    settings.render_nodes.nodes["gpu-b"].base_url = "http://192.0.2.20:8188"
    assert view().status == "context_mismatch"
    settings.render_nodes.nodes["gpu-b"].base_url = "http://192.0.2.10:8188"
    path = settings.qualification.evidence_dir / "pc-b-passive-monitor.json"
    path.write_text("{broken", encoding="utf-8")
    assert view().status == "invalid"
    assert PCBEvidenceMonitorSnapshot.model_validate(
        sample.model_dump(mode="json")
    ) == sample


def test_monitor_rejects_symlink_and_bad_parent_without_side_effects(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "evidence")
    _session(settings.qualification.evidence_dir)
    path = settings.qualification.evidence_dir / "pc-b-passive-monitor.json"
    fake = tmp_path / "real-monitor.json"
    fake.write_text("sensitive original", encoding="utf-8")
    try:
        path.symlink_to(fake)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this Windows host")
    result = read_pc_b_evidence_monitor(
        settings, session_id=SESSION, now=NOW, max_age_seconds=1800,
    )
    assert result.status == "missing"
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            monitor, "correlate_live_pc_b_owner_report",
            lambda *a, **k: _status("blocked"),
        )
        monkeypatch.setattr(
            monitor, "review_live_pc_b_survival_evidence",
            lambda *a, **k: _status("blocked"),
        )
        assert asyncio.run(
            PCBEvidenceRefreshMaintenance(settings, Mock()).maintain()
        ) is None
    finally:
        monkeypatch.undo()
    assert fake.read_text(encoding="utf-8") == "sensitive original"


def test_missing_session_never_creates_monitor_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "evidence")
    settings.qualification.evidence_dir.mkdir()
    monkeypatch.setattr(
        monitor, "correlate_live_pc_b_owner_report",
        lambda *a, **k: pytest.fail("unexpected PC-B fetch"),
    )
    assert asyncio.run(
        PCBEvidenceRefreshMaintenance(settings, Mock()).maintain()
    ) is None
    assert not list(settings.qualification.evidence_dir.iterdir())


def test_monitor_interval_has_floor() -> None:
    with pytest.raises(ValueError):
        PCBEvidenceRefreshMaintenance(
            _settings(Path("data/qualification")), Mock(),
            interval_seconds=299,
        )
