"""PC-A protected, passive PC-B field prerequisites: never launch GPU work."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.pc_b_remote_field_gates import (
    compile_remote_pc_b_passive_gates,
    inspect_remote_pc_b_passive_gates,
)
from artifex.render_node.installation_audit import RemoteRendererInstallationAudit
from artifex.render_node.models import RemoteOwnerReadinessEvidence

NOW = datetime(2026, 10, 12, 2, tzinfo=UTC)
TOKEN_ENV = "ARTIFEX_RENDER_NODE_TOKEN"


def _installation(
    *,
    node: str = "gpu-b",
    heartbeat: str = "fresh",
    captured: datetime = NOW,
) -> RemoteRendererInstallationAudit:
    return RemoteRendererInstallationAudit.model_validate({
        "node_id": node,
        "audit": {
            "node_id": node, "captured_utc": captured.isoformat(),
            "hostname": "GPU-B", "native_windows": True,
            "managed_renderer_configured": True,
            "renderer_task": {
                "role": "renderer", "status": "running",
                "registered": True, "managed": True,
                "configuration_verified": True, "reason_code": "verified_running",
            },
            "survival_observer_task": {
                "role": "survival-observer", "status": "running",
                "registered": True, "managed": True,
                "configuration_verified": True, "reason_code": "verified_running",
            },
            "evidence_spool": "empty", "observer_heartbeat": heartbeat,
            "safe_for_passive_observation": True,
        },
    })


def _owner(
    *,
    host: str = "gpu-b",
    collected: datetime = NOW,
    started: str = "2026-10-11T23:00:00+00:00",
    pid: Any = 4700,
) -> RemoteOwnerReadinessEvidence:
    return RemoteOwnerReadinessEvidence.model_validate({
        "node_id": "gpu-b", "report": {
            "collected_utc": collected.isoformat(),
            "host": host, "status": "observed_independently",
            "observations_are_independent": True,
            "actual_comfyui_inspected_by_owner_audit": True,
            "production_qualified": False,
            "renderer_restart_authorized": False,
            "configured_comfyui_python": "C:/private/NEVER_SHARE.exe",
            "private_bearer_token": "very-secret-example",
            "checks": {
                "disposable_launcher_fixture": {
                    "status": "pass", "reason": "fixture verified",
                },
                "live_comfyui_owner": {"status": "pass", "reason": "verified"},
            },
            "owner_audit": {
                "status": "observed_stable",
                "actual_listener_pid": pid,
                "actual_process_started_utc": started,
                "CommandLine": "private ComfyUI command line",
            },
        },
    })


def _compile(
    installation: RemoteRendererInstallationAudit | None = None,
    owner: RemoteOwnerReadinessEvidence | None = None,
):
    return compile_remote_pc_b_passive_gates(
        node_id="gpu-b", configured=True,
        installation=installation, cached_owner=owner,
        attempted=2, checked_utc=NOW,
    )


def test_verified_existing_pc_b_prerequisites_are_only_observational() -> None:
    report = _compile(_installation(), _owner())
    assert report.status == "observed_prerequisites"
    assert report.renderer_task_status == "running"
    assert report.observer_task_status == "running"
    assert report.observer_heartbeat_status == "fresh"
    assert report.launcher_fixture_status == "pass"
    assert report.real_comfyui_owner_status == "pass"
    assert report.blockers == ()
    assert report.authenticated_read_only_gets_attempted == 2
    assert report.production_qualified is False
    assert report.gpu_jobs_submitted is False
    assert report.disposable_fixture_started_by_get is False
    assert report.uninterrupted_renderer_survival_proven is False
    assert report.verified_actual_supervisor_loss is False
    assert report.session_stage_pass_registered is False
    serialized = report.model_dump_json()
    for sensitive in ("very-secret-example", "NEVER_SHARE", "4700", "CommandLine"):
        assert sensitive not in serialized


@pytest.mark.parametrize("bad", [
    "stale_installation", "clock_in_future", "heartbeat_stale",
    "foreign_node", "host_mismatch", "stale_launcher_cache",
    "missing_creation_time", "naive_creation_time", "invalid_pid",
    "unverified_fixture", "unverified_owner",
])
def test_stale_or_conflicting_remote_evidence_is_never_prerequisite_pass(
    bad: str,
) -> None:
    install = _installation(
        node="foreign-node" if bad == "foreign_node" else "gpu-b",
        heartbeat="stale" if bad == "heartbeat_stale" else "fresh",
        captured=NOW - timedelta(minutes=5) if bad == "stale_installation"
        else NOW + timedelta(minutes=5) if bad == "clock_in_future" else NOW,
    )
    owner = _owner(
        host="another-machine" if bad == "host_mismatch" else "gpu-b",
        collected=NOW - timedelta(minutes=11)
        if bad == "stale_launcher_cache" else NOW,
        started="" if bad == "missing_creation_time"
        else "2026-10-11T23:00:00" if bad == "naive_creation_time"
        else "2026-10-11T23:00:00+00:00",
        pid=0 if bad == "invalid_pid" else 4700,
    )
    if bad in {"unverified_fixture", "unverified_owner"}:
        field = (
            "disposable_launcher_fixture" if bad == "unverified_fixture"
            else "live_comfyui_owner"
        )
        owner.report["checks"][field]["status"] = "fail"
    report = _compile(install, owner)
    assert report.status == "needs_evidence"
    assert report.blockers
    assert report.production_qualified is False
    assert report.authenticated_read_only_gets_attempted == 2


def test_both_or_one_missing_read_are_bounded_and_nonqualifying() -> None:
    both = _compile()
    assert both.status == "unavailable"
    assert both.installation_probe_status == "unavailable"
    assert both.cached_owner_probe_status == "unavailable"
    assert set(both.blockers) == {
        "pc_b_installation_get_unavailable",
        "pc_b_cached_launcher_get_unavailable",
    }
    only_owner = _compile(None, _owner())
    assert only_owner.status == "needs_evidence"
    assert only_owner.cached_owner_probe_status == "observed"
    assert only_owner.production_qualified is False


def test_native_utc_plus_nine_cached_timestamps_compare_equivalently() -> None:
    owner = _owner(
        collected=datetime(2026, 10, 12, 11, tzinfo=datetime.now().astimezone().tzinfo),
    )
    # Use an exact +09:00 timestamp for deterministic cross-OS behavior.
    owner.report["collected_utc"] = "2026-10-12T11:00:00+09:00"
    assert _compile(_installation(), owner).status == "observed_prerequisites"


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_nodes.primary = "gpu-b"
    settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://192.0.2.10:8188",
        attestation_url="http://192.0.2.10:8190",
        attestation_token_env=TOKEN_ENV,
    )
    return settings


def test_gets_never_run_when_bearer_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    called: list[str] = []

    def forbid(*args: object, **kwargs: object) -> Any:
        called.append("network")
        raise AssertionError("no remote GET without PC-A token")

    result = inspect_remote_pc_b_passive_gates(
        _settings(), now=NOW,
        installation_probe=forbid, owner_cache_probe=forbid,
    )
    assert result.status == "unconfigured"
    assert result.authenticated_read_only_gets_attempted == 0
    assert called == []
    assert result.production_qualified is False


@pytest.mark.parametrize("broken", ["installation", "owner", "both"])
def test_two_authenticated_gets_are_sequential_independent_and_redacted(
    monkeypatch: pytest.MonkeyPatch, broken: str,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "very-secret-example")
    called: list[tuple[str, str, float]] = []

    def installation(node: str, config: RenderNodeConfig, *,
                     timeout_seconds: float) -> RemoteRendererInstallationAudit:
        called.append(("installation", node, timeout_seconds))
        if broken in {"installation", "both"}:
            raise httpx.ConnectError("very-secret-example C:/private")
        return _installation()

    def owner(node: str, config: RenderNodeConfig, *,
              timeout_seconds: float) -> RemoteOwnerReadinessEvidence:
        called.append(("owner", node, timeout_seconds))
        if broken in {"owner", "both"}:
            raise ValueError("very-secret-example C:/private")
        return _owner()

    report = inspect_remote_pc_b_passive_gates(
        _settings(), now=NOW,
        installation_probe=installation, owner_cache_probe=owner,
    )
    assert called == [
        ("installation", "gpu-b", 8.0), ("owner", "gpu-b", 8.0),
    ]
    assert report.authenticated_read_only_gets_attempted == 2
    assert report.status == (
        "unavailable" if broken == "both" else "needs_evidence"
    )
    assert report.production_qualified is False
    assert report.service_mutated is False
    serialized = json.dumps(report.model_dump(mode="json"))
    assert "very-secret-example" not in serialized
    assert "C:/private" not in serialized
