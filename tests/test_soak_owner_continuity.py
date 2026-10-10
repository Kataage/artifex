from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from artifex.config.models import ArtifexSettings, RenderNodeConfig, RenderNodesConfig
from artifex.qualification.soak_observer import (
    SoakOwnerObservation,
    SoakSample,
    observe_soak,
    sample_soak,
    verify_soak_evidence,
)
from artifex.render_node.models import (
    RemoteRendererOwnerAudit,
    RenderAssetDigest,
    RenderNodeAttestation,
)

BASE = datetime(2026, 10, 9, 0, 0, tzinfo=UTC)
CHECKS = (
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
)
ASSETS = {
    "production_checkpoint": "a" * 64,
    "refiner_checkpoint": "b" * 64,
    "vae": "c" * 64,
    "upscale_model": "d" * 64,
}


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_nodes = RenderNodesConfig(
        primary="gpu-b",
        nodes={
            "gpu-b": RenderNodeConfig(
                base_url="http://render.test:8188",
                attestation_url="http://render.test:8190",
            ),
        },
    )
    assert settings.qualification.require_renderer_owner_observation
    return settings


def _owner(
    elapsed: float, *,
    pid: int = 404,
    started: str = "2026-10-09T00:00:00Z",
    status: str = "observed_stable",
) -> SoakOwnerObservation:
    return SoakOwnerObservation(
        node_id="gpu-b",
        captured_utc=BASE + timedelta(seconds=elapsed),
        status=status,
        process_observation_verified=status == "observed_stable",
        actual_listener_pid=pid,
        actual_process_started_utc=started,
        launcher_pid=401,
        receipt_schema=2,
        scheduler_state="Running",
        checks={key: "pass" for key in CHECKS},
    )


def _sample(elapsed: float, owner: SoakOwnerObservation | None) -> SoakSample:
    hour = int(elapsed // 3600)
    return SoakSample(
        observed_at=BASE + timedelta(seconds=elapsed),
        elapsed_seconds=elapsed,
        llm_ok=True, comfyui_ok=True, renderer_ok=True,
        gpu_used_mib=4096, ram_used_mib=2048, free_disk_gib=20,
        telemetry_rows=10 + hour,
        severe_events=0, finalized_packs=5 + hour,
        asset_sha256=ASSETS,
        owner_observation=owner,
    )


def _run(
    tmp_path: Path, *,
    change: str = "stable",
) -> tuple[Path, Any]:
    clock = [0.0]
    def clock_now() -> float:
        return clock[0]

    def advance(seconds: float) -> None:
        clock[0] += seconds

    def wall() -> datetime:
        return BASE + timedelta(seconds=clock[0])

    def probe(settings: ArtifexSettings, elapsed: float) -> SoakSample:
        o = _owner(elapsed)
        if elapsed >= 3600:
            if change == "missing":
                return _sample(elapsed, None)
            if change == "reused_pid":
                o = _owner(elapsed, pid=505)
            if change == "process_restarted":
                o = _owner(elapsed, started="2026-10-09T01:00:00Z")
            if change == "timezone_equivalent":
                # Exactly the same original Windows process instant.
                o = _owner(elapsed, started="2026-10-09T09:00:00+09:00")
            if change == "offset_equivalent":
                o = _owner(elapsed, started="2026-10-08T17:00:00-07:00")
            if change == "naive_start":
                o = _owner(elapsed, started="2026-10-09T00:00:00")
            if change == "malformed_start":
                o = _owner(elapsed, started="not-an-ISO-timestamp")
            if change == "future_start":
                o = _owner(elapsed, started="2026-10-10T00:00:00Z")
            if change == "same_pid_new_start":
                o = _owner(elapsed, started="2026-10-09T00:00:01Z")
            if change == "blocked":
                o = _owner(elapsed, status="blocked")
            if change == "unproven_tcp":
                o = o.model_copy(update={
                    "checks": {**o.checks, "tcp_ownership": "unknown"},
                })
            if change == "stale":
                o = o.model_copy(update={
                    "captured_utc": BASE + timedelta(seconds=elapsed - 200),
                })
            if change == "wrong_node":
                o = o.model_copy(update={"node_id": "other"})
            if change == "transport_failure":
                raise httpx.ConnectError("PC-B unavailable")
        return _sample(elapsed, o)

    evidence = tmp_path / f"{change}.jsonl"
    result = observe_soak(
        _settings(), output=evidence, duration_hours=8, interval_seconds=3600,
        sample=probe, monotonic=clock_now, sleep=advance, now=wall,
        qualification_session_id="test-session",
    )
    return evidence, result


def test_owner_continuity_can_be_reviewed_without_production_qualification(
    tmp_path: Path,
) -> None:
    trace, assessment = _run(tmp_path)
    header = json.loads(trace.read_text().splitlines()[0])
    assert header["schema_version"] == 2
    assert header["owner_observation_required"] is True
    assert assessment.ready_for_soak_review
    assert assessment.owner_observed_samples == 9
    assert assessment.owner_incidents == 0
    assert assessment.production_qualified is False
    repeated = verify_soak_evidence(
        trace, require_owner_observation=True,
    )
    assert repeated.evidence_sha256 == assessment.evidence_sha256


@pytest.mark.parametrize("change", [
    "timezone_equivalent",
    "offset_equivalent",
])
def test_same_windows_comfyui_process_does_not_fail_for_time_zone_spelling(
    tmp_path: Path, change: str,
) -> None:
    trace, report = _run(tmp_path, change=change)
    assert trace.is_file()
    assert report.ready_for_soak_review
    assert report.owner_observed_samples == 9
    assert report.owner_incidents == 0
    assert not report.production_qualified
    rechecked = verify_soak_evidence(trace, require_owner_observation=True)
    assert rechecked.ready_for_soak_review
    assert rechecked.evidence_sha256 == report.evidence_sha256
    assert rechecked.owner_incidents == 0


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ("missing", "snapshot missing"),
        ("reused_pid", "PID or creation time changed"),
        ("process_restarted", "PID or creation time changed"),
        ("same_pid_new_start", "PID or creation time changed"),
        ("naive_start", "creation timestamp invalid"),
        ("malformed_start", "creation timestamp invalid"),
        ("future_start", "creation timestamp invalid"),
        ("blocked", "blocked or incomplete"),
        ("unproven_tcp", "blocked or incomplete"),
        ("stale", "observation missing timestamp or stale"),
        ("wrong_node", "belongs to another renderer"),
        ("transport_failure", "Probe exception"),
    ],
)
def test_every_owner_lapse_blocks_eight_hour_readiness(
    tmp_path: Path, change: str, fragment: str,
) -> None:
    trace, result = _run(tmp_path, change=change)
    assert not result.ready_for_soak_review
    assert result.production_qualified is False
    assert any(fragment in issue for issue in result.issues), result.issues
    assert trace.is_file()


@pytest.mark.parametrize("field", [
    "captured_utc",
    "observed_at",
])
def test_soak_trace_with_timezone_naive_observation_fails_without_exception(
    tmp_path: Path, field: str,
) -> None:
    trace, report = _run(tmp_path)
    assert report.ready_for_soak_review
    lines = trace.read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[2])
    if field == "captured_utc":
        second["owner_observation"]["captured_utc"] = "2026-10-09T01:00:00"
    else:
        second["observed_at"] = "2026-10-09T01:00:00"
    lines[2] = json.dumps(second)
    trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assessment = verify_soak_evidence(trace, require_owner_observation=True)
    assert not assessment.ready_for_soak_review
    assert assessment.owner_incidents >= 1
    assert any("timestamp" in issue for issue in assessment.issues)


def test_legacy_trace_cannot_satisfy_new_owner_policy(tmp_path: Path) -> None:
    trace, report = _run(tmp_path)
    assert report.ready_for_soak_review
    lines = trace.read_text().splitlines()
    header = json.loads(lines[0])
    header["schema_version"] = 1
    header["owner_observation_required"] = False
    lines[0] = json.dumps(header)
    trace.write_text("\n".join(lines) + "\n")
    assert verify_soak_evidence(trace).ready_for_soak_review
    rejected = verify_soak_evidence(trace, require_owner_observation=True)
    assert not rejected.ready_for_soak_review
    assert any("does not declare mandatory" in issue for issue in rejected.issues)


def test_missing_owner_during_normal_sample_is_rejected_even_if_health_is_good(
    tmp_path: Path,
) -> None:
    trace, _ = _run(tmp_path)
    lines = trace.read_text().splitlines()
    sample = json.loads(lines[2])
    sample["owner_observation"] = None
    lines[2] = json.dumps(sample)
    trace.write_text("\n".join(lines) + "\n")
    assert not verify_soak_evidence(trace).ready_for_soak_review


def test_actual_probe_fetches_remote_owner_and_records_safe_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.qualification.soak_observer as soak_module

    settings = _settings()
    settings.storage.database_url = "sqlite:///ignored.db"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={
                "system": {"comfyui_version": "0.9"},
                "devices": [{"vram_total": 8192 * 1048576,
                             "vram_free": 4096 * 1048576}],
            })
        raise AssertionError(f"Unexpected API request: {request.url}")

    original_client = httpx.Client
    monkeypatch.setattr(
        soak_module.httpx, "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(soak_module, "_physical_ram_mib", lambda: 5000.0)
    monkeypatch.setattr(soak_module, "_read_db_counts", lambda _: (8, 0, 7))
    monkeypatch.setattr(
        soak_module, "fetch_render_attestation",
        lambda node_id, node, **kwargs: RenderNodeAttestation(
            node_id="gpu-b", hostname="pc-b", created_at=datetime.now(UTC),
            os={"system": "Windows"}, comfyui_base_url="http://render.test:8188",
            nvidia_gpus=({"name": "RTX 3060"},),
            assets=tuple(
                RenderAssetDigest(label=k, path=f"D:/models/{k}",
                                  sha256=v, bytes=123)
                for k, v in ASSETS.items()
            ),
        ),
    )
    owner_fetches: list[str] = []

    def fake_owner(node_id: str, node: RenderNodeConfig, **kwargs: Any) -> RemoteRendererOwnerAudit:
        owner_fetches.append(node_id)
        observed = _owner(0)
        details = observed.model_dump(mode="json")
        details["captured_utc"] = datetime.now(UTC).isoformat()
        payload = {
            "node_id": node_id,
            "audit": {
                "schema_version": 1,
                "captured_utc": details["captured_utc"],
                "status": "observed_stable",
                "checks": {k: {"status": "pass", "reason": "ok"} for k in CHECKS},
                "actual_listener_pid": 404,
                "actual_process_started_utc": "2026-10-09T00:00:00Z",
                "launcher_pid": 401,
                "receipt_schema": 2,
                "scheduler_state": "Running",
                "process_observation_verified": True,
                "restart_authorized": False,
                "child_survival_qualified": False,
                "production_qualified": False,
                "mutated_services": False,
            },
        }
        return RemoteRendererOwnerAudit.model_validate(payload)

    monkeypatch.setattr(soak_module, "fetch_renderer_owner_audit", fake_owner)
    result = sample_soak(settings, 0)
    assert result.owner_observation is not None
    assert result.owner_observation.actual_listener_pid == 404
    assert result.owner_observation.status == "observed_stable"
    assert result.renderer_ok and result.comfyui_ok
    assert owner_fetches == ["gpu-b"]
    assert not result.errors

    def fail_owner(*args: Any, **kwargs: Any) -> RemoteRendererOwnerAudit:
        raise httpx.ConnectError("PC-B unreachable")

    monkeypatch.setattr(soak_module, "fetch_renderer_owner_audit", fail_owner)
    lost = sample_soak(settings, 300)
    assert lost.owner_observation is None
    assert any("authenticated owner audit" in item for item in lost.errors)
