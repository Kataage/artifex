"""PC-B remote cache: no launcher process is triggered by HTTP GET."""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.render_node.attestation import serve_attestation
from artifex.render_node.client import fetch_remote_owner_readiness
from artifex.render_node.remote_owner_readiness import (
    OwnerReadinessCache,
    can_collect_owner_readiness,
)


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_agent.node_id = "gpu-b"
    settings.render_agent.require_token = True
    settings.render_agent.token_env = "ARTIFEX_RENDER_NODE_TOKEN"
    settings.render_agent.gateway.enabled = True
    settings.render_agent.comfyui_process.enabled = True
    settings.render_agent.comfyui_process.executable = Path("C:/Fake/python.exe")
    settings.render_agent.comfyui_process.working_directory = Path("C:/Fake")
    return settings


def test_cache_only_collects_on_worker_not_on_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.render_node.remote_owner_readiness as module

    settings = _settings()
    path = tmp_path / "renderer.yaml"
    path.write_text("render_agent: {}\n", encoding="utf-8")
    monkeypatch.setattr(module, "can_collect_owner_readiness", lambda *a: True)
    called: list[Path] = []
    def collect(settings: ArtifexSettings, *, config: Path) -> dict[str, Any]:
        called.append(config)
        return {
            "status": "blocked",
            "configured_comfyui_python": "C:/private/python.exe",
            "launcher_fixture": {
                "status": "blocked",
                "launcher_identity": {"CommandLine": "secret-command-line"},
                "listener_identity": {"CommandLine": "another-secret"},
            },
            "owner_audit": {"status": "blocked"},
        }

    monkeypatch.setattr(module, "collect_owner_readiness", collect)
    cache = OwnerReadinessCache(settings, owner_config=path)
    assert cache.snapshot() is None
    cache._collect_once()
    for _ in range(10):
        snapshot = cache.snapshot()
        assert snapshot is not None
        assert snapshot["configured_comfyui_python"] is None
        assert snapshot["launcher_fixture"]["launcher_identity"] is None
        assert snapshot["launcher_fixture"]["listener_identity"] is None
        assert "secret" not in json.dumps(snapshot)
    assert called == [path]
    cache._updated_monotonic = -10**8
    assert cache.snapshot() is None


def test_native_worker_requires_protected_configuration_and_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.render_node.remote_owner_readiness as module

    settings = _settings()
    path = tmp_path / "renderer.yaml"
    path.write_text("ok", encoding="utf-8")
    monkeypatch.setattr(
        module, "os",
        SimpleNamespace(
            name="nt", environ={"ARTIFEX_RENDER_NODE_TOKEN": "secret"},
        ),
    )
    assert can_collect_owner_readiness(settings, path)
    settings.render_agent.gateway.enabled = False
    assert not can_collect_owner_readiness(settings, path)
    settings.render_agent.gateway.enabled = True
    settings.render_agent.require_token = False
    assert not can_collect_owner_readiness(settings, path)
    settings.render_agent.require_token = True
    assert not can_collect_owner_readiness(settings, tmp_path / "absent.yaml")
    assert not can_collect_owner_readiness(settings, None)


def test_protected_endpoint_is_cached_and_never_runs_probe_on_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.render_node.attestation as module

    settings = _settings()
    settings.render_agent.bind_host = "127.0.0.1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        settings.render_agent.port = sock.getsockname()[1]
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "unit-token")
    path = tmp_path / "renderer.yaml"
    path.write_text("ok", encoding="utf-8")
    sample = {"status": "blocked", "production_qualified": False}
    started = threading.Event()
    counts: list[str] = []

    class FakeCache:
        def __init__(self, settings: ArtifexSettings, *, owner_config: Path | None):
            assert owner_config == path

        def run(self, stop: threading.Event) -> None:
            counts.append("worker")
            started.set()
            stop.wait(10)

        def snapshot(self) -> dict[str, Any] | None:
            counts.append("cache_get")
            return sample

    monkeypatch.setattr(module, "can_collect_owner_readiness", lambda *a: True)
    monkeypatch.setattr(module, "OwnerReadinessCache", FakeCache)
    stop = threading.Event()
    server = threading.Thread(
        target=serve_attestation,
        args=(settings,), kwargs={"stop_event": stop, "owner_config": path},
        daemon=True,
    )
    server.start()
    base = f"http://127.0.0.1:{settings.render_agent.port}"
    try:
        assert started.wait(timeout=5)
        with httpx.Client(trust_env=False, timeout=3) as client:
            assert client.get(base + "/v1/owner-readiness-evidence").status_code == 401
            for _ in range(3):
                response = client.get(
                    base + "/v1/owner-readiness-evidence",
                    headers={"Authorization": "Bearer unit-token"},
                )
                assert response.status_code == 200
                assert response.json()["node_id"] == "gpu-b"
                assert response.json()["report"] == sample
            query = client.get(
                base + "/v1/owner-readiness-evidence?force=1",
                headers={"Authorization": "Bearer unit-token"},
            )
            assert query.status_code == 503
        assert counts == ["worker", "cache_get", "cache_get", "cache_get"]
    finally:
        stop.set()
        server.join(timeout=5)


def test_http_client_requires_bearer_node_match_no_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "unit-token")
    cfg = RenderNodeConfig(
        base_url="http://localhost:8188",
        attestation_url="http://localhost:8190",
    )
    requested: list[str] = []

    def respond(req: httpx.Request) -> httpx.Response:
        requested.append(req.url.path)
        assert req.headers["authorization"] == "Bearer unit-token"
        return httpx.Response(200, json={
            "node_id": "gpu-b", "report": {"schema_version": 1},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        snapshot = fetch_remote_owner_readiness("gpu-b", cfg, client=client)
    assert snapshot.node_id == "gpu-b"
    assert requested == ["/v1/owner-readiness-evidence"]

    def wrong(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"node_id": "foreign", "report": {}})

    with (
        httpx.Client(transport=httpx.MockTransport(wrong)) as client,
        pytest.raises(ValueError, match="node ID"),
    ):
        fetch_remote_owner_readiness("gpu-b", cfg, client=client)

    def redirect(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://other.invalid/"})

    with (
        httpx.Client(transport=httpx.MockTransport(redirect)) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        fetch_remote_owner_readiness("gpu-b", cfg, client=client)

    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN")
    with pytest.raises(ValueError, match="token is missing"):
        fetch_remote_owner_readiness("gpu-b", cfg)


@pytest.mark.parametrize("require_token", [True, False])
def test_attestation_failed_inventory_never_discloses_private_error(
    monkeypatch: pytest.MonkeyPatch, require_token: bool,
) -> None:
    """The legacy public endpoint must not leak exceptions or local paths."""
    import artifex.render_node.attestation as module

    settings = _settings()
    settings.render_agent.require_token = require_token
    settings.render_agent.bind_host = "127.0.0.1"
    settings.render_agent.gateway.enabled = False
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        settings.render_agent.port = sock.getsockname()[1]
    if require_token:
        monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "secret-bearer-credential")
    else:
        monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    private_message = (
        "D:/private/models/authentication.key "
        "https://private.local/?token=secret-bearer-credential"
    )
    calls: list[int] = []

    def fail_inventory(_: ArtifexSettings):
        calls.append(1)
        raise RuntimeError(private_message)

    monkeypatch.setattr(module, "build_attestation", fail_inventory)
    stop = threading.Event()
    server = threading.Thread(
        target=serve_attestation,
        args=(settings,),
        kwargs={"stop_event": stop, "owner_config": None},
        daemon=True,
    )
    server.start()
    url = f"http://127.0.0.1:{settings.render_agent.port}/v1/attestation?fresh=1"
    headers = (
        {"Authorization": "Bearer secret-bearer-credential"}
        if require_token else {}
    )
    try:
        with httpx.Client(trust_env=False, timeout=3) as client:
            for attempt in range(40):
                try:
                    response = client.get(url, headers=headers)
                    break
                except httpx.ConnectError:
                    if not server.is_alive() and attempt > 1:
                        pytest.fail("Fixture server did not start")
                    time.sleep(0.05)
            else:
                pytest.fail("Fixture server never accepted connections")

            assert response.status_code == 503
            assert response.json() == {"error": "attestation_unavailable"}
            assert "private" not in response.text
            assert "secret-bearer-credential" not in response.text
            assert "RuntimeError" not in response.text
            assert "detail" not in response.text
            assert len(calls) == 1

            if require_token:
                unauthorized = client.get(url)
                assert unauthorized.status_code == 401
                assert unauthorized.json() == {"error": "unauthorized"}
                assert len(calls) == 1
    finally:
        stop.set()
        server.join(timeout=5)


def test_attestation_client_uses_direct_bounded_authenticated_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.render_node.client as client_module

    cfg = RenderNodeConfig(
        base_url="http://127.0.0.1:8188",
        attestation_url="http://pc-b.invalid:8190",
    )
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "private-attestation-token")
    monkeypatch.setenv("HTTPS_PROXY", "http://forbidden-proxy.invalid:3128")
    received: list[httpx.Request] = []
    client_flags: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={
            "node_id": "gpu-b",
            "created_at": "2026-10-13T00:00:00+00:00",
            "hostname": "actual-pc-b",
            "os": {"system": "Windows"},
            "comfyui_base_url": "http://127.0.0.1:8188",
            "inventory_errors": [
                {"path": "lora_asset", "error": "lora_asset_unreadable"},
            ],
        })

    real_client = httpx.Client

    def make_client(*args: Any, **kwargs: Any) -> httpx.Client:
        client_flags.append(dict(kwargs))
        return real_client(
            *args, transport=httpx.MockTransport(respond), **kwargs,
        )

    monkeypatch.setattr(client_module.httpx, "Client", make_client)
    attestation = client_module.fetch_render_attestation(
        "gpu-b", cfg, fresh=True,
    )
    assert attestation.node_id == "gpu-b"
    assert attestation.inventory_errors[0]["error"] == "lora_asset_unreadable"
    assert client_flags[0]["trust_env"] is False
    assert client_flags[0]["follow_redirects"] is False
    assert len(received) == 1
    assert received[0].url.query == b"fresh=1"
    assert received[0].headers["authorization"] == "Bearer private-attestation-token"


@pytest.mark.parametrize("failure", ["redirect", "content_length", "oversized_stream"])
def test_attestation_client_rejects_redirects_and_unbounded_http_payloads(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    import artifex.render_node.client as module

    cfg = RenderNodeConfig(
        base_url="http://127.0.0.1:8188",
        attestation_url="https://pc-b.invalid:8190",
    )
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "private-token")
    # Deliberately tiny cap to exercise both rejection paths with small bytes.
    monkeypatch.setattr(module, "_MAX_ATTESTATION_RESPONSE_BYTES", 80)
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if failure == "redirect":
            return httpx.Response(
                302, headers={"location": "https://foreign.invalid/"},
            )
        if failure == "content_length":
            return httpx.Response(
                200, headers={"content-length": "999999"}, content=b"{}",
            )
        return httpx.Response(200, stream=httpx.ByteStream(b"z" * 81))

    with httpx.Client(
        transport=httpx.MockTransport(respond), trust_env=False,
        follow_redirects=True,
    ) as supplied:
        if failure == "redirect":
            with pytest.raises(httpx.HTTPStatusError):
                module.fetch_render_attestation("gpu-b", cfg, client=supplied)
        else:
            with pytest.raises(ValueError, match="size bound"):
                module.fetch_render_attestation("gpu-b", cfg, client=supplied)
    assert len(requests) == 1
    assert "foreign.invalid" not in requests[0]
