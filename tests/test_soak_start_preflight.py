"""No-GPU live evidence gate before creating an eight-hour qualification."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from artifex.config.models import ArtifexSettings, RenderNodeConfig, RenderNodesConfig
from artifex.qualification.soak_observer import SoakOwnerObservation, SoakSample
from artifex.qualification.soak_start_preflight import (
    check_soak_start_sample,
    preflight_soak_start,
)

NOW = datetime(2026, 10, 12, 9, tzinfo=UTC)
ASSETS = {
    "production_checkpoint": "a" * 64,
    "refiner_checkpoint": "b" * 64,
    "vae": "c" * 64,
    "upscale_model": "d" * 64,
}
CHECKS = (
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
)


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_nodes = RenderNodesConfig(
        primary="gpu-b",
        nodes={"gpu-b": RenderNodeConfig(
            base_url="http://192.0.2.3:8188",
            attestation_url="http://192.0.2.3:8190",
        )},
    )
    settings.qualification.require_renderer_owner_observation = True
    return settings


def _owner(
    *, node_id: str = "gpu-b", started: str = "2026-10-12T09:00:00Z",
    captured_utc: datetime = NOW,
    status: str = "observed_stable",
    checks: dict[str, str] | None = None,
) -> SoakOwnerObservation:
    return SoakOwnerObservation.model_validate({
        "node_id": node_id, "captured_utc": captured_utc, "status": status,
        "process_observation_verified": status == "observed_stable",
        "actual_listener_pid": 551,
        "actual_process_started_utc": started,
        "launcher_pid": 550, "receipt_schema": 2,
        "scheduler_state": "Running",
        "checks": checks if checks is not None else {
            name: "pass" for name in CHECKS
        },
    })


def _sample(
    *, observed_at: datetime = NOW,
    owner: SoakOwnerObservation | None = None,
) -> SoakSample:
    return SoakSample(
        observed_at=observed_at, elapsed_seconds=0,
        llm_ok=True, comfyui_ok=True, renderer_ok=True,
        gpu_used_mib=4000, ram_used_mib=9000,
        free_disk_gib=42, telemetry_rows=12,
        severe_events=0, finalized_packs=0,
        asset_sha256=dict(ASSETS),
        owner_observation=owner if owner is not None else _owner(),
    )


def test_all_live_dependencies_are_observed_without_qualification_pass() -> None:
    result = check_soak_start_sample(
        _settings(), _sample(), checked_utc=NOW,
    )
    assert result.ready and result.issues == ()
    assert result.mode == "read_only"
    assert result.sample_checked_utc == NOW
    assert not result.qualification_session_created
    assert not result.soak_evidence_file_written
    assert not result.gpu_jobs_submitted
    assert not result.services_modified
    assert not result.production_qualified


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ("llm", "llm_unavailable"),
        ("comfyui", "comfyui_unavailable"),
        ("renderer", "renderer_attestation_unavailable"),
        ("exceptions", "sample_reported_errors"),
        ("gpu", "resource_metrics_missing"),
        ("database", "database_metrics_missing"),
        ("assets", "required_asset_hashes_missing"),
        ("old", "sample_clock_invalid"),
        ("naive", "sample_clock_invalid"),
        ("owner_missing", "renderer_owner_missing"),
        ("owner_node", "renderer_owner_unverified"),
        ("owner_checks", "renderer_owner_unverified"),
        ("owner_start", "renderer_owner_identity_invalid"),
        ("owner_stale", "renderer_owner_snapshot_stale"),
    ],
)
def test_invalid_or_stale_initial_probe_blocks_before_soak(
    changed: str, expected: str,
) -> None:
    sample = _sample()
    replacements: dict[str, object] = {}
    if changed in {"llm", "comfyui", "renderer"}:
        field = {
            "llm": "llm_ok", "comfyui": "comfyui_ok", "renderer": "renderer_ok",
        }[changed]
        replacements[field] = False
    elif changed == "exceptions":
        replacements["errors"] = ("PRIVATE C:/real/token bearer secret",)
    elif changed == "gpu":
        replacements["gpu_used_mib"] = None
    elif changed == "database":
        replacements["telemetry_rows"] = None
    elif changed == "assets":
        replacements["asset_sha256"] = {"production_checkpoint": "fake"}
    elif changed == "old":
        replacements["observed_at"] = NOW - timedelta(minutes=5)
    elif changed == "naive":
        replacements["observed_at"] = NOW.replace(tzinfo=None)
    elif changed == "owner_missing":
        replacements["owner_observation"] = None
    elif changed == "owner_node":
        replacements["owner_observation"] = _owner(node_id="foreign-node")
    elif changed == "owner_checks":
        replacements["owner_observation"] = _owner(
            checks={"native_windows": "pass"},
        )
    elif changed == "owner_start":
        replacements["owner_observation"] = _owner(
            started="2026-10-13T09:00:00Z",
        )
    elif changed == "owner_stale":
        replacements["owner_observation"] = _owner(
            captured_utc=NOW - timedelta(minutes=6),
        )
    sample = sample.model_copy(update=replacements)
    result = check_soak_start_sample(_settings(), sample, checked_utc=NOW)
    assert not result.ready
    assert expected in result.issues
    assert "PRIVATE" not in result.model_dump_json()
    assert "C:/real" not in result.model_dump_json()
    assert not result.production_qualified


def test_probe_exception_is_redacted_and_never_claims_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: object) -> SoakSample:
        raise RuntimeError("secret bearer C:/private/path")
    monkeypatch.setattr(
        "artifex.qualification.soak_start_preflight.sample_soak", explode,
    )
    result = preflight_soak_start(_settings())
    assert result.issues == ("sample_unavailable",)
    assert not result.ready
    assert not result.qualification_session_created
    assert "secret" not in result.model_dump_json()


def test_unconfigured_owner_is_not_silently_accepted_when_policy_requires_it() -> None:
    settings = _settings()
    settings.render_nodes.primary = None
    result = check_soak_start_sample(
        settings, _sample().model_copy(update={"owner_observation": None}),
        checked_utc=NOW,
    )
    # Missing renderer itself is a blocking LLM/ComfyUI/attestation issue on
    # actual sample_soak. This helper should not manufacture owner proof.
    assert result.production_qualified is False
