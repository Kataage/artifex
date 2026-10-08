from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy.admission import ComfySubmissionFence
from artifex.comfy.reconciliation import RendererReconcileReport, RendererReconcileSample
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.db import Database
from artifex.domain import AgentState
from artifex.operations.pair_maintenance import (
    RendererGatewayAPI,
    RendererGatewayStatus,
    coordinate_pair_maintenance,
)
from artifex.runtime import RuntimeStore

_SECRET = "pair-maintenance-long-private-gateway-token"


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.comfyui.base_url = "http://pc-b.example:8191"
    settings.comfyui.gateway_token_env = "ARTIFEX_RENDER_NODE_TOKEN"
    settings.comfyui.submission_fence_path = tmp_path / "controller-fence.sqlite"
    settings.render_nodes.nodes = {
        "main": RenderNodeConfig(base_url="http://pc-b.example:8191")
    }
    settings.render_nodes.primary = "main"
    return settings


def _services(tmp_path: Path) -> tuple[Database, RuntimeStore, ComfySubmissionFence]:
    database = Database(f"sqlite:///{tmp_path / 'controller.db'}")
    database.migrate()
    runtime = RuntimeStore(database)
    runtime.reconcile_process_start()
    runtime.set_agent_state(AgentState.RUNNING)
    return database, runtime, ComfySubmissionFence(tmp_path / "controller-fence.sqlite")


class FakeComfy:
    def __init__(self, *, busy: bool = False) -> None:
        self.busy = busy
        self.calls = 0

    async def queue_snapshot(self) -> dict[str, Any]:
        self.calls += 1
        return {
            "queue_running": [[1]] if self.busy else [],
            "queue_pending": [],
        }


class FakeGateway:
    def __init__(self, *, sealed: bool = False, online: bool = True) -> None:
        self.sealed = sealed
        self.online = online
        self.busy = False
        self.fail_seal = False
        self.fail_release = False
        self.calls: list[str] = []
        self.on_release: Any = None

    async def status(self) -> RendererGatewayStatus:
        self.calls.append("status")
        if not self.online:
            raise httpx.ConnectError("gateway offline")
        return RendererGatewayStatus(
            admission_sealed=self.sealed,
            upstream_loopback_configured=True,
            external_loopback_clients_fenced=False,
            restart_authorized=False,
        )

    async def seal(self) -> bool:
        self.calls.append("seal")
        if self.fail_seal:
            raise httpx.ConnectError("remote seal lost")
        if self.busy:
            return False
        self.sealed = True
        return True

    async def release(self) -> None:
        self.calls.append("release")
        if self.fail_release:
            raise httpx.ConnectError("remote release lost")
        if self.on_release is not None:
            self.on_release()
        self.sealed = False


def _verified() -> RendererReconcileReport:
    return RendererReconcileReport(
        status="ready", ready=True, samples=1,
        last=RendererReconcileSample(
            checked_at="2026-10-09T00:00:00+00:00",
            queue_state="idle", running=0, queued=0, workflow_ready=True,
        ),
        next_actions=("run_two_pc_deployment_verify",),
    )


@pytest.mark.asyncio
async def test_pair_preview_never_changes_controller_or_remote(
    tmp_path: Path,
) -> None:
    db, runtime, fence = _services(tmp_path)
    gateway = FakeGateway()
    report = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, FakeComfy(), fence,  # type: ignore[arg-type]
        gateway=gateway,  # type: ignore[arg-type]
    )
    assert report.status == "preview" and not report.completed
    assert not report.controller_paused and not report.controller_sealed
    assert report.renderer_sealed is False
    assert not fence.path.exists()
    assert gateway.calls == ["status"]
    assert not report.restart_authorized and not report.production_qualified
    db.dispose()


@pytest.mark.asyncio
async def test_pair_apply_seals_both_and_preserves_paused_controller(
    tmp_path: Path,
) -> None:
    db, runtime, fence = _services(tmp_path)
    gateway = FakeGateway()
    comfy = FakeComfy()
    report = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, comfy, fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, wait_seconds=3, poll_seconds=0.01,  # type: ignore[arg-type]
    )
    assert report.status == "both_sealed" and report.completed
    assert report.controller_paused and report.controller_sealed
    assert report.renderer_sealed is True
    assert fence.status() and gateway.sealed
    assert gateway.calls == ["status", "seal", "status"]
    assert comfy.calls >= 4
    assert not report.external_loopback_clients_fenced
    assert not report.restart_authorized
    db.dispose()


@pytest.mark.asyncio
async def test_pair_refuses_to_pause_when_gateway_preflight_unreachable(
    tmp_path: Path,
) -> None:
    db, runtime, fence = _services(tmp_path)
    gateway = FakeGateway(online=False)
    report = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, FakeComfy(), fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True,  # type: ignore[arg-type]
    )
    assert report.status == "gateway_unreachable"
    assert runtime.get_agent_state() is AgentState.RUNNING
    assert not fence.path.exists() and not report.controller_sealed
    assert gateway.calls == ["status"]
    db.dispose()


@pytest.mark.asyncio
async def test_pair_failed_remote_seal_keeps_controller_sealed_and_retryable(
    tmp_path: Path,
) -> None:
    db, runtime, fence = _services(tmp_path)
    gateway = FakeGateway()
    gateway.busy = True
    comfy = FakeComfy()
    first = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, comfy, fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, wait_seconds=3, poll_seconds=0.01,  # type: ignore[arg-type]
    )
    assert first.status == "partial" and not first.completed
    assert first.controller_sealed and not first.renderer_sealed
    assert fence.status() and runtime.get_agent_state() is AgentState.PAUSED
    gateway.busy = False
    second = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, comfy, fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, wait_seconds=3, poll_seconds=0.01,  # type: ignore[arg-type]
    )
    assert second.status == "both_sealed" and second.completed
    assert gateway.sealed and fence.status()
    db.dispose()


@pytest.mark.asyncio
async def test_pair_gateway_disconnect_after_controller_seal_fails_closed(
    tmp_path: Path,
) -> None:
    db, runtime, fence = _services(tmp_path)
    gateway = FakeGateway()
    gateway.fail_seal = True
    report = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, FakeComfy(), fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, wait_seconds=3, poll_seconds=0.01,  # type: ignore[arg-type]
    )
    assert report.status == "partial"
    assert report.renderer_sealed is None
    assert fence.status()
    assert runtime.get_agent_state() is AgentState.PAUSED
    assert not report.restart_authorized
    db.dispose()


@pytest.mark.asyncio
async def test_pair_refuses_when_real_comfy_queue_busy(
    tmp_path: Path,
) -> None:
    db, runtime, fence = _services(tmp_path)
    gateway = FakeGateway()
    report = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime,
        FakeComfy(busy=True), fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, wait_seconds=0,  # type: ignore[arg-type]
    )
    assert report.status == "blocked"
    assert not fence.status() and not gateway.sealed
    assert runtime.get_agent_state() is AgentState.PAUSED
    db.dispose()


@pytest.mark.asyncio
async def test_release_denied_unless_both_fences_closed_and_controller_paused(
    tmp_path: Path,
) -> None:
    db, runtime, fence = _services(tmp_path)
    gateway = FakeGateway(sealed=True)
    report = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, FakeComfy(), fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, release=True,  # type: ignore[arg-type]
    )
    assert report.status == "blocked"
    assert gateway.calls == ["status"]
    assert gateway.sealed
    assert runtime.get_agent_state() is AgentState.RUNNING
    db.dispose()


@pytest.mark.asyncio
async def test_release_requires_live_runtime_audit_and_never_resumes_automatically(
    tmp_path: Path,
) -> None:
    db, runtime, fence = _services(tmp_path)
    gateway = FakeGateway()
    await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, FakeComfy(), fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, wait_seconds=3, poll_seconds=0.01,  # type: ignore[arg-type]
    )
    calls = 0

    async def not_ready(*args: Any, **kwargs: Any) -> RendererReconcileReport:
        nonlocal calls
        calls += 1
        return _verified().model_copy(update={"ready": False, "status": "busy"})

    refused = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, FakeComfy(), fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, release=True,
        reconcile_fn=not_ready,  # type: ignore[arg-type]
    )
    assert refused.status == "runtime_unready"
    assert calls == 1 and gateway.sealed and fence.status()
    assert "release" not in gateway.calls

    async def ready(*args: Any, **kwargs: Any) -> RendererReconcileReport:
        assert kwargs["require_idle"] is True
        assert kwargs["wait_seconds"] == 0
        return _verified()

    # Controller admission must remain sealed until the renderer reopens.
    gateway.on_release = lambda: fence.status() or pytest.fail("PC-A released too early")
    released = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, FakeComfy(), fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, release=True,
        reconcile_fn=ready,  # type: ignore[arg-type]
    )
    assert released.status == "released" and released.completed
    assert not gateway.sealed and not fence.status()
    assert runtime.get_agent_state() is AgentState.PAUSED
    assert not released.restart_authorized
    db.dispose()


@pytest.mark.asyncio
async def test_remote_release_failure_keeps_controller_fenced(
    tmp_path: Path,
) -> None:
    db, runtime, fence = _services(tmp_path)
    gateway = FakeGateway()
    await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, FakeComfy(), fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, wait_seconds=3, poll_seconds=0.01,  # type: ignore[arg-type]
    )
    gateway.fail_release = True

    async def ready(*args: Any, **kwargs: Any) -> RendererReconcileReport:
        return _verified()

    result = await coordinate_pair_maintenance(
        _settings(tmp_path), db, runtime, FakeComfy(), fence,  # type: ignore[arg-type]
        gateway=gateway, apply=True, release=True,
        reconcile_fn=ready,  # type: ignore[arg-type]
    )
    assert result.status == "partial" and not result.completed
    assert fence.status() and gateway.sealed
    db.dispose()


@pytest.mark.asyncio
async def test_real_api_rejects_bad_gateway_schema_credential_redirect_and_endpoint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", _SECRET)
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization", ""))
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={
                "admission_sealed": False, "upstream_loopback_configured": True,
                "external_loopback_clients_fenced": False,
                "restart_authorized": False,
            })
        if request.url.path.endswith("/seal"):
            return httpx.Response(409, json={
                "admission_sealed": False, "reason": "queue_busy",
                "restart_authorized": False,
            })
        return httpx.Response(302, headers={"location": "http://evil.test/"})

    client = httpx.AsyncClient(
        base_url="http://pc-b.example:8191",
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    api = RendererGatewayAPI(settings, client=client)
    assert not (await api.status()).admission_sealed
    assert await api.seal() is False
    with pytest.raises(ValueError, match="HTTP 302"):
        await api.release()
    assert seen == ["Bearer " + _SECRET] * 3
    await client.aclose()

    settings.render_nodes.nodes["main"].base_url = "http://other-pc:8191"
    with pytest.raises(ValueError, match="same authenticated gateway"):
        RendererGatewayAPI(settings)
    settings.render_nodes.nodes["main"].base_url = settings.comfyui.base_url
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN")
    with pytest.raises(ValueError, match="missing or too short"):
        RendererGatewayAPI(settings)


def test_pair_cli_help_and_missing_config_fail_closed(tmp_path: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["maintenance", "pair", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "--release" in help_result.output
    missing = runner.invoke(
        app, ["maintenance", "pair", "--config", str(tmp_path / "no.yaml")],
    )
    assert missing.exit_code == 1
    assert "existing non-symlink config" in missing.output
