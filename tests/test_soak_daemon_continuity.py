"""Read-only PC-A daemon continuity must survive every soak sample."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from artifex.config.models import ArtifexSettings
from artifex.qualification.daemon_start_observation import DaemonStartObservation
from artifex.qualification.soak_observer import (
    SoakSample,
    observe_soak,
    verify_soak_evidence,
)

BASE = datetime(2026, 10, 12, 12, tzinfo=UTC)
ASSETS = {
    "production_checkpoint": "a" * 64,
    "refiner_checkpoint": "b" * 64,
    "vae": "c" * 64,
    "upscale_model": "d" * 64,
}


def _trace(
    tmp_path: Path,
    *,
    change: Literal["none", "stop", "restart", "exception"] = "none",
) -> tuple[Path, object]:
    settings = ArtifexSettings()
    settings.qualification.require_renderer_owner_observation = False
    clock = [0.0]
    probed: list[float] = []

    def observer(_: ArtifexSettings) -> DaemonStartObservation:
        elapsed = clock[0]
        probed.append(elapsed)
        if change == "exception" and elapsed >= 3600:
            raise RuntimeError("SECRET bearer and C:/operator/private-path")
        if change == "stop" and elapsed >= 3600:
            return DaemonStartObservation(state="heartbeat_stale", ready=False)
        marker = "b" if change == "restart" and elapsed >= 3600 else "a"
        return DaemonStartObservation(
            state="observed_active_loop",
            ready=True,
            loop_fingerprint=marker * 64,
        )

    def sample(_: ArtifexSettings, elapsed: float) -> SoakSample:
        hour = int(elapsed / 3600)
        return SoakSample(
            observed_at=BASE + timedelta(seconds=elapsed),
            elapsed_seconds=elapsed,
            llm_ok=True, comfyui_ok=True, renderer_ok=True,
            gpu_used_mib=4000, ram_used_mib=2000, free_disk_gib=30,
            telemetry_rows=20 + hour, severe_events=0,
            finalized_packs=40 + hour,
            asset_sha256=ASSETS,
        )

    path = tmp_path / "daemon-soak.jsonl"
    report = observe_soak(
        settings, output=path, duration_hours=8, interval_seconds=3600,
        require_daemon_observation=True, daemon_observer=observer,
        sample=sample,
        monotonic=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        now=lambda: BASE + timedelta(seconds=clock[0]),
    )
    assert probed
    return path, report


def test_unchanged_loop_is_checked_at_all_nine_intervals(tmp_path: Path) -> None:
    path, report = _trace(tmp_path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["schema_version"] == 3
    assert lines[0]["daemon_observation_required"] is True
    assert report.ready_for_soak_review
    assert report.observed_samples == 9
    assert report.daemon_observed_samples == 9
    assert report.daemon_incidents == 0
    assert report.production_qualified is False
    assert all(
        line["daemon_state"] == "observed_active_loop"
        and len(line["daemon_loop_fingerprint"]) == 64
        for line in lines[1:-1]
    )
    assert verify_soak_evidence(
        path, require_daemon_observation=True
    ).ready_for_soak_review


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("stop", "not observed active"),
        ("restart", "loop changed"),
        ("exception", "not observed active"),
    ],
)
def test_stopped_restarted_or_unreadable_loop_fails_fast(
    tmp_path: Path, change: str, message: str,
) -> None:
    path, report = _trace(tmp_path, change=change)  # type: ignore[arg-type]
    assert not report.ready_for_soak_review
    assert report.observed_samples == 2
    assert report.daemon_incidents == 1
    assert any(message in issue for issue in report.issues)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]["stopped_early"] is True
    assert "SECRET" not in path.read_text()
    assert "private-path" not in path.read_text()


def test_legacy_trace_never_satisfies_production_daemon_policy(
    tmp_path: Path,
) -> None:
    path, report = _trace(tmp_path)
    assert report.ready_for_soak_review
    rows = path.read_text().splitlines()
    head = json.loads(rows[0])
    head["schema_version"] = 2
    head["daemon_observation_required"] = False
    rows[0] = json.dumps(head)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert not verify_soak_evidence(
        path, require_daemon_observation=True
    ).ready_for_soak_review


def test_removed_or_switched_middle_sample_cannot_verify(
    tmp_path: Path,
) -> None:
    path, report = _trace(tmp_path)
    assert report.ready_for_soak_review
    rows = path.read_text().splitlines()
    sample = json.loads(rows[4])
    sample["daemon_loop_fingerprint"] = None
    rows[4] = json.dumps(sample)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    checked = verify_soak_evidence(path, require_daemon_observation=True)
    assert not checked.ready_for_soak_review
    assert checked.daemon_incidents == 1
    assert any("fingerprint" in issue for issue in checked.issues)
