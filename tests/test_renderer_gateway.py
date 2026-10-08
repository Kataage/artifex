from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from artifex.config.models import ArtifexSettings
from artifex.render_node.comfy_process import ManagedComfyUI
from artifex.render_node.gateway import make_gateway_server, serve_gateway

_SECRET = "test-local-only-renderer-gateway-secret-at-least-24-bytes"


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_agent.gateway.enabled = True
    settings.render_agent.gateway.admission_path = tmp_path / "gateway-fence.sqlite"
    settings.render_agent.comfyui_process.enabled = True
    settings.render_agent.comfyui_process.arguments = (
        "main.py", "--listen", "127.0.0.1", "--port", "8188",
    )
    settings.comfyui.base_url = "http://127.0.0.1:8188"
    return settings


def _start(
    settings: ArtifexSettings, upstream: httpx.Client,
) -> tuple[Any, threading.Event, threading.Thread, httpx.Client]:
    server = make_gateway_server(
        settings, upstream=upstream, bind_host="127.0.0.1", port=0,
    )
    stop = threading.Event()
    worker = threading.Thread(
        target=serve_gateway, args=(server, stop), daemon=True,
    )
    worker.start()
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        headers={"Authorization": f"Bearer {_SECRET}"},
        timeout=5,
        trust_env=False,
    )
    return server, stop, worker, client


def _stop(server: Any, stop: threading.Event, worker: threading.Thread,
          client: httpx.Client, upstream: httpx.Client) -> None:
    client.close()
    stop.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    upstream.close()


def test_gateway_requires_explicit_managed_mode_and_24_char_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    settings.render_agent.gateway.enabled = False
    with pytest.raises(ValueError, match="explicitly enabled"):
        make_gateway_server(settings)
    settings.render_agent.gateway.enabled = True
    settings.render_agent.comfyui_process.enabled = False
    with pytest.raises(ValueError, match="owned managed"):
        make_gateway_server(settings)
    settings.render_agent.comfyui_process.enabled = True
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    with pytest.raises(ValueError, match="24 chars"):
        make_gateway_server(settings)
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", _SECRET)
    settings.comfyui.base_url = "http://pc-b.example:8188"
    with pytest.raises(ValueError, match="local"):
        make_gateway_server(settings)


def test_managed_gateway_refuses_existing_comfy_and_non_loopback_listen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    manager = ManagedComfyUI(settings)
    monkeypatch.setattr(manager, "_healthy", lambda: True)
    with pytest.raises(RuntimeError, match="already-running external"):
        manager.start(threading.Event())
    settings.render_agent.comfyui_process.arguments = (
        "main.py", "--listen", "0.0.0.0", "--port", "8188",
    )
    with pytest.raises(ValueError, match="127.0.0.1"):
        manager.start(threading.Event())
    settings.render_agent.comfyui_process.arguments = (
        "main.py", "--listen", "127.0.0.1", "--listen", "0.0.0.0",
    )
    with pytest.raises(ValueError, match="exactly one"):
        manager.start(threading.Event())


def test_gateway_auth_proxy_api_and_no_header_forwarding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", _SECRET)
    seen: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        seen.append((request.method, request.url.path, request.content))
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "test-123", "number": 0})
        if request.url.path == "/history/test-123":
            return httpx.Response(200, json={})
        if request.url.path == "/view":
            return httpx.Response(200, content=b"pngbytes", headers={"Content-Type": "image/png"})
        if request.url.path == "/object_info":
            return httpx.Response(200, json={"CheckpointLoaderSimple": {}})
        return httpx.Response(200, json={"system": {}, "devices": []})

    upstream = httpx.Client(
        base_url="http://127.0.0.1:8188",
        transport=httpx.MockTransport(handler),
    )
    server, stop, thread, client = _start(_settings(tmp_path), upstream)
    try:
        assert httpx.get(
            f"http://127.0.0.1:{server.server_address[1]}/queue", trust_env=False
        ).status_code == 401
        assert client.get("/v1/gateway/status").json() == {
            "admission_sealed": False,
            "upstream_loopback_configured": True,
            "external_loopback_clients_fenced": False,
            "restart_authorized": False,
        }
        assert client.get("/object_info").status_code == 200
        assert client.get("/history/test-123").status_code == 200
        assert client.get("/view", params={"filename": "test.png"}).content == b"pngbytes"
        assert client.post("/prompt", json={"prompt": {}}).json()["prompt_id"] == "test-123"
        assert client.get("/anything-else").status_code == 404
        assert client.post("/anything-else", json={}).status_code == 404
        assert any(p == "/prompt" for _, p, _ in seen)
    finally:
        _stop(server, stop, thread, client, upstream)


def test_gateway_seal_blocks_all_mutations_and_state_survives_gateway_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", _SECRET)
    busy = False
    submitted = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted
        if request.url.path == "/queue" and request.method == "GET":
            return httpx.Response(
                200,
                json={"queue_running": [[1]] if busy else [], "queue_pending": []},
            )
        if request.url.path == "/prompt":
            submitted += 1
            return httpx.Response(200, json={"prompt_id": "test-456"})
        return httpx.Response(200, json={})

    settings = _settings(tmp_path)
    upstream = httpx.Client(
        base_url=settings.comfyui.base_url,
        transport=httpx.MockTransport(handler),
    )
    server, stop, thread, client = _start(settings, upstream)
    try:
        busy = True
        response = client.post("/v1/gateway/seal", json={})
        assert response.status_code == 409
        assert response.json()["reason"] == "queue_busy"
        assert client.post("/prompt", json={"prompt": {}}).status_code == 200
        busy = False
        response = client.post("/v1/gateway/seal", json={})
        assert response.status_code == 200
        assert response.json()["admission_sealed"] is True
        assert response.json()["restart_authorized"] is False
        assert client.get("/v1/gateway/status").json()["admission_sealed"] is True
        assert client.post("/prompt", json={"prompt": {}}).status_code == 423
        assert client.post("/interrupt", json={}).status_code == 423
        assert submitted == 1
    finally:
        _stop(server, stop, thread, client, upstream)

    upstream2 = httpx.Client(
        base_url=settings.comfyui.base_url,
        transport=httpx.MockTransport(handler),
    )
    server2, stop2, thread2, client2 = _start(settings, upstream2)
    try:
        assert client2.get("/v1/gateway/status").json()["admission_sealed"] is True
        assert client2.post("/prompt", json={"prompt": {}}).status_code == 423
        assert client2.post("/v1/gateway/release", json={}).status_code == 200
        assert client2.get("/v1/gateway/status").json()["admission_sealed"] is False
        assert client2.post("/prompt", json={"prompt": {}}).status_code == 200
        assert submitted == 2
    finally:
        _stop(server2, stop2, thread2, client2, upstream2)


def test_gateway_handles_unverifiable_queue_and_invalid_workflow_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", _SECRET)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": []})
        if request.url.path == "/prompt":
            return httpx.Response(400, json={"error": {"type": "invalid_prompt"}})
        return httpx.Response(404)

    upstream = httpx.Client(
        base_url="http://127.0.0.1:8188",
        transport=httpx.MockTransport(handler),
    )
    server, stop, thread, client = _start(_settings(tmp_path), upstream)
    try:
        assert client.post("/v1/gateway/seal", json={}).status_code == 409
        response = client.post("/prompt", json={"prompt": {}})
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_prompt"
        assert client.get("/v1/gateway/status").json()["admission_sealed"] is False
    finally:
        _stop(server, stop, thread, client, upstream)
