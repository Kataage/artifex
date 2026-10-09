from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.render_node.attestation import serve_attestation
from artifex.render_node.client import fetch_renderer_safety_inspection
from artifex.render_node.startup_inspection import (
    RemoteRendererSafetyInspection,
    RendererStartupInspection,
)


def _inspection(*, status: str = "owned_observed") -> RendererStartupInspection:
    return RendererStartupInspection.model_validate({
        "captured_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "renderer_node_id": "gpu-b",
        "configured_comfyui_port": 8188,
        "configured_attestation_port": 8190,
        "configured_gateway_port": 8191,
        "config_protected": True,
        "windows_native": True,
        "snapshots_consistent": True,
        "service_ports_coherent": status == "owned_observed",
        "owner_audit_status": "observed_stable",
        "actual_owned_listener_pid": 404 if status == "owned_observed" else None,
        "launch_authorized": False,
        "restart_authorized": False,
        "reattach_authorized": False,
        "production_qualified": False,
        "mutated_services": False,
    })


def _remote(*, status: str = "owned_observed") -> dict[str, Any]:
    return RemoteRendererSafetyInspection(
        node_id="gpu-b", inspection=_inspection(status=status)
    ).model_dump(mode="json")


def _config() -> RenderNodeConfig:
    return RenderNodeConfig(
        base_url="http://pc-b:8191",
        output_mode="api",
        attestation_url="http://pc-b:8190",
        attestation_token_env="ARTIFEX_RENDER_NODE_TOKEN",
    )


def test_remote_safety_client_uses_bearer_without_redirects_or_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "do-not-leak")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        assert request.method == "GET"
        assert request.url.path == "/v1/renderer-safety"
        assert request.headers["authorization"] == "Bearer do-not-leak"
        return httpx.Response(200, json=_remote())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_renderer_safety_inspection(
            "gpu-b", _config(), client=client,
        )
    assert len(calls) == 1
    assert result.inspection.status == "owned_observed"
    assert not result.inspection.launch_authorized
    assert not result.inspection.production_qualified
    assert "do-not-leak" not in result.model_dump_json()


@pytest.mark.parametrize("mode", [
    "wrong_node", "wrong_inner_node", "oversize", "redirect",
    "unavailable", "untrusted_success",
])
def test_remote_safety_client_rejects_untrusted_or_incomplete_responses(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "do-not-leak")

    def handler(request: httpx.Request) -> httpx.Response:
        if mode == "oversize":
            return httpx.Response(200, content=b" " * (64 * 1024 + 1))
        if mode == "redirect":
            return httpx.Response(302, headers={"location": "http://elsewhere/secret"})
        if mode == "unavailable":
            return httpx.Response(503)
        payload = _remote()
        if mode == "wrong_node":
            payload["node_id"] = "foreign"
        elif mode == "wrong_inner_node":
            payload["inspection"]["renderer_node_id"] = "foreign"
        elif mode == "untrusted_success":
            payload["inspection"]["launch_authorized"] = True
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            fetch_renderer_safety_inspection("gpu-b", _config(), client=client)


def test_remote_safety_client_refuses_missing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    with httpx.Client(transport=httpx.MockTransport(
        lambda req: pytest.fail("No unauthenticated request")
    )) as client:
        with pytest.raises(ValueError, match="token is missing"):
            fetch_renderer_safety_inspection("gpu-b", _config(), client=client)


def test_authenticated_server_safety_path_is_read_only_and_not_public(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.render_node.attestation as attestation
    import artifex.render_node.startup_inspection as safety_module

    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "do-not-leak")
    settings = ArtifexSettings()
    settings.render_agent.node_id = "gpu-b"
    settings.render_agent.bind_host = "127.0.0.1"
    settings.render_agent.token_env = "ARTIFEX_RENDER_NODE_TOKEN"
    settings.render_agent.require_token = True
    config = tmp_path / "render-node.yaml"
    config.write_text("{}\n", encoding="utf-8")
    observed: list[str] = []

    def safe_inspect(*args: Any, **kwargs: Any) -> RendererStartupInspection:
        observed.append("read-only")
        return _inspection(status="service_ports_blocked")

    monkeypatch.setattr(safety_module, "inspect_renderer_startup", safe_inspect)
    bound: list[Any] = []
    ready = threading.Event()
    native_server = attestation.ThreadingHTTPServer

    def bind_test_server(addr: tuple[str, int], handler: Any) -> Any:
        server = native_server(("127.0.0.1", 0), handler)
        bound.append(server)
        ready.set()
        return server

    monkeypatch.setattr(attestation, "ThreadingHTTPServer", bind_test_server)
    stop = threading.Event()
    worker = threading.Thread(
        target=serve_attestation,
        args=(settings,),
        kwargs={"stop_event": stop, "owner_config": config},
        daemon=True,
    )
    worker.start()
    try:
        assert ready.wait(5)
        server = bound[0]
        url = f"http://127.0.0.1:{server.server_port}/v1/renderer-safety"
        with httpx.Client(timeout=10, trust_env=False) as client:
            unauth = client.get(url)
            assert unauth.status_code == 401
            assert not observed
            with_auth = {"Authorization": "Bearer do-not-leak"}
            reject_query = client.get(url + "?extra=1", headers=with_auth)
            assert reject_query.status_code == 503
            assert not observed
            result = client.get(url, headers=with_auth)
            assert result.status_code == 200
            content = result.json()
            assert content["node_id"] == "gpu-b"
            assert content["inspection"]["status"] == "service_ports_blocked"
            assert content["inspection"]["launch_authorized"] is False
            assert content["inspection"]["production_qualified"] is False
            assert content["inspection"]["mutated_services"] is False
            assert observed == ["read-only"]
            assert "do-not-leak" not in json.dumps(content)
    finally:
        stop.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
