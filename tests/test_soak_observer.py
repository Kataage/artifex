from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig, RenderNodesConfig
from artifex.qualification.soak_observer import (
    SoakSample,
    _gpu_usage_mib,
    _read_db_counts,
    observe_soak,
    sample_soak,
    verify_soak_evidence,
)
from artifex.render_node.models import RenderAssetDigest, RenderNodeAttestation

BASE = datetime(2026, 10, 8, 1, 0, tzinfo=UTC)
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
                output_mode="api",
                attestation_url="http://render.test:8190",
            ),
        },
    )
    return settings


def _sample(
    elapsed: float, *,
    good: bool = True, fatal: bool = False,
    changed: bool = False, progress: bool = True,
    resource: bool = True,
) -> SoakSample:
    hours = int(elapsed / 3600)
    models = dict(ASSETS)
    if changed and elapsed >= 3600:
        models["vae"] = "e" * 64
    return SoakSample(
        observed_at=BASE + timedelta(seconds=elapsed),
        elapsed_seconds=elapsed,
        llm_ok=good, comfyui_ok=good, renderer_ok=good,
        gpu_used_mib=8123.0 if resource else None,
        ram_used_mib=2048.0, free_disk_gib=16.0,
        telemetry_rows=10 + hours if progress else 10,
        severe_events=1 if fatal and elapsed >= 3600 else 0,
        finalized_packs=20 + hours if progress else 20,
        asset_sha256=models,
        errors=() if good else ("comfyui offline",),
    )


def _observe(
    tmp_path: Path, *,
    time_target: float = 8,
    sample_seconds: float = 3600,
    sample_factory: Any = _sample,
) -> tuple[Path, Any]:
    state = [0.0]

    def monotonic() -> float:
        return state[0]

    def sleep(seconds: float) -> None:
        state[0] += seconds

    def now() -> datetime:
        return BASE + timedelta(seconds=state[0])

    path = tmp_path / "soak.jsonl"
    result = observe_soak(
        _settings(),
        output=path,
        duration_hours=time_target,
        interval_seconds=sample_seconds,
        sample=lambda settings, elapsed: sample_factory(elapsed),
        monotonic=monotonic,
        sleep=sleep,
        now=now,
    )
    return path, result


def test_eight_hour_trace_can_be_reviewed_without_qualifying_production(
    tmp_path: Path,
) -> None:
    path, result = _observe(tmp_path)
    assert result.ready_for_soak_review
    assert result.production_qualified is False
    assert result.observed_samples == 9
    assert result.elapsed_hours == 8
    assert result.verified_completed_packs == 8
    assert result.observed_telemetry_rows == 8
    assert result.fatal_errors == 0
    assert result.peak_vram_mib == 8123
    assert result.peak_ram_mib == 2048
    assert result.minimum_free_disk_gib == 16
    assert not result.issues
    assert result.evidence_sha256 == verify_soak_evidence(path).evidence_sha256
    with pytest.raises(FileExistsError, match="cannot overwrite"):
        _observe(tmp_path)


@pytest.mark.parametrize(
    ("bad_factory", "message"),
    [
        (lambda elapsed: _sample(elapsed, good=elapsed == 0), "unavailable"),
        (lambda elapsed: _sample(elapsed, fatal=True), "Error or critical"),
        (lambda elapsed: _sample(elapsed, changed=True), "model assets changed"),
        (lambda elapsed: _sample(elapsed, progress=False), "No new persisted"),
        (lambda elapsed: _sample(elapsed, resource=False), "resource or DB telemetry"),
    ],
)
def test_eight_hour_trace_fails_if_activity_or_health_is_not_proved(
    tmp_path: Path,
    bad_factory: Any,
    message: str,
) -> None:
    _, result = _observe(tmp_path, sample_factory=bad_factory)
    assert not result.ready_for_soak_review
    assert result.production_qualified is False
    assert any(message in issue for issue in result.issues)


def test_initial_controller_or_render_failure_stops_early_and_retains_file(
    tmp_path: Path,
) -> None:
    path, result = _observe(
        tmp_path, sample_factory=lambda elapsed: _sample(elapsed, good=False)
    )
    assert path.is_file()
    assert result.observed_samples == 1
    assert not result.ready_for_soak_review
    assert any("Monitor stopped early" in issue for issue in result.issues)
    lines = path.read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == [
        "start", "sample", "end"
    ]


def test_short_or_sparse_observations_never_pass_an_eight_hour_minimum(
    tmp_path: Path,
) -> None:
    _, short = _observe(tmp_path, time_target=0.1, sample_seconds=10)
    assert not short.ready_for_soak_review
    assert any("shorter than required" in issue for issue in short.issues)
    path = tmp_path / "soak.jsonl"
    lines = path.read_text().splitlines()
    # Drop all middle observations to simulate outage/observer being stopped.
    path.write_text("\n".join([lines[0], lines[1], lines[-2], lines[-1]]) + "\n")
    assessment = verify_soak_evidence(path)
    assert not assessment.ready_for_soak_review
    assert any("sampling gap" in issue for issue in assessment.issues)


def test_truncated_or_oversized_or_symlinked_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    path, result = _observe(tmp_path)
    assert result.ready_for_soak_review
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError):
        verify_soak_evidence(path)
    path.write_bytes(b"z" * (12 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="12 MiB"):
        verify_soak_evidence(path)
    link = tmp_path / "symlink.jsonl"
    try:
        link.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink privilege not available")
    with pytest.raises(ValueError, match="symlink"):
        verify_soak_evidence(link)


def test_wall_clock_disagreement_rejected_even_with_valid_monotonic_duration(
    tmp_path: Path,
) -> None:
    path, _ = _observe(tmp_path)
    raw = path.read_text().splitlines()
    last = json.loads(raw[-1])
    last["finished_at"] = (BASE + timedelta(hours=3)).isoformat()
    raw[-1] = json.dumps(last)
    path.write_text("\n".join(raw) + "\n")
    result = verify_soak_evidence(path)
    assert not result.ready_for_soak_review
    assert any("wall-clock" in item for item in result.issues)


def test_sqlite_metrics_are_read_only_and_measure_actual_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifex.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE agent_events (id INTEGER, severity TEXT)")
        db.execute("CREATE TABLE packs (id TEXT, state TEXT)")
        db.executemany(
            "INSERT INTO agent_events VALUES (?,?)",
            [(1, "info"), (2, "warning"), (3, "critical"), (4, "error")],
        )
        db.executemany(
            "INSERT INTO packs VALUES (?,?)",
            [("1", "finalized"), ("2", "finalized"), ("3", "failed")],
        )
    before = path.read_bytes()
    assert _read_db_counts("sqlite:///" + str(path).replace("\\", "/")) == (4, 2, 2)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="local SQLite"):
        _read_db_counts("postgresql://local/db")


def test_comfy_vram_telemetry_requires_exact_free_total_metrics() -> None:
    assert _gpu_usage_mib({
        "devices": [{"vram_total": 12 * 1048576, "vram_free": 4 * 1048576}]
    }) == 8
    assert _gpu_usage_mib({"devices": [{"vram_total": 1, "vram_free": 2}]}) is None
    assert _gpu_usage_mib({"devices": [{"vram_total": 1}]}) is None
    assert _gpu_usage_mib({"devices": []}) is None


def test_real_probe_uses_auth_api_and_checks_db_resources_without_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex.qualification import soak_observer

    settings = _settings()
    settings.storage.database_url = "sqlite:///" + str(tmp_path / "example.db")
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={
                "system": {"comfyui_version": "0.9"},
                "devices": [{
                    "vram_total": 12288 * 1048576,
                    "vram_free": 4096 * 1048576,
                }],
            })
        raise AssertionError(f"unexpected request {request.url}")

    real_client = httpx.Client

    def fake_client(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler))

    def fake_remote(*args: Any, **kwargs: Any) -> RenderNodeAttestation:
        assert args[0] == "gpu-b"
        return RenderNodeAttestation(
            node_id="gpu-b", hostname="pc-b", created_at=datetime.now(UTC),
            os={"system": "Windows"}, comfyui_base_url="http://render.test:8188",
            nvidia_gpus=({"name": "RTX 3060"},),
            assets=tuple(
                RenderAssetDigest(
                    label=name, path=f"D:/models/{name}", sha256=digest, bytes=123,
                ) for name, digest in ASSETS.items()
            ),
        )

    monkeypatch.setattr(soak_observer.httpx, "Client", fake_client)
    monkeypatch.setattr(soak_observer, "fetch_render_attestation", fake_remote)
    monkeypatch.setattr(soak_observer, "_physical_ram_mib", lambda: 2048.0)
    monkeypatch.setattr(soak_observer, "_read_db_counts", lambda _: (15, 0, 5))
    result = sample_soak(settings, 25.0)
    assert result.llm_ok and result.comfyui_ok and result.renderer_ok
    assert result.gpu_used_mib == 8192.0
    assert result.ram_used_mib == 2048.0
    assert result.asset_sha256 == ASSETS
    assert not result.errors
    assert [urlsplit for urlsplit in calls] == [
        ("GET", "http://127.0.0.1:8899/health"),
        ("GET", "http://render.test:8188/system_stats"),
    ]


def test_cli_soak_check_and_watch_fail_closed_and_emit_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex.qualification.soak_observer import SoakAssessment

    config = tmp_path / "local.yaml"
    config.write_text("{}\n", encoding="utf-8")
    runner = CliRunner()
    evidence = tmp_path / "soak.jsonl"
    denied = runner.invoke(app, [
        "qualify", "soak-check", "--config", str(config),
        "--evidence", str(evidence),
    ])
    assert denied.exit_code == 1

    def fake_observe(settings: ArtifexSettings, **kwargs: Any) -> SoakAssessment:
        evidence.write_text("some evidence", encoding="utf-8")
        return SoakAssessment(
            run_id="test", ready_for_soak_review=False,
            elapsed_hours=0.001, observed_samples=1,
            verified_completed_packs=0, observed_telemetry_rows=0, fatal_errors=0,
            peak_vram_mib=None, peak_ram_mib=None, minimum_free_disk_gib=None,
            evidence_sha256="a" * 64, issues=("Insufficient duration",),
        )

    monkeypatch.setattr("artifex.cli.observe_soak", fake_observe)
    args = [
        "qualify", "soak-observe", "--config", str(config),
        "--output", str(evidence), "--hours", "0.0001",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "ready_for_soak_review" in result.output
    assert evidence.is_file()
    again = runner.invoke(app, args)
    assert again.exit_code == 1
    assert evidence.read_text(encoding="utf-8") == "some evidence"
