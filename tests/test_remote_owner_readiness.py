"""PC-B remote cache: no launcher process is triggered by HTTP GET."""
from __future__ import annotations

import json
import socket
import threading
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

    with httpx.Client(transport=httpx.MockTransport(wrong)) as client:
        with pytest.raises(ValueError, match="node ID"):
            fetch_remote_owner_readiness("gpu-b", cfg, client=client)

    def redirect(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://other.invalid/"})

    with httpx.Client(transport=httpx.MockTransport(redirect)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            fetch_remote_owner_readiness("gpu-b", cfg, client=client)

    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN")
    with pytest.raises(ValueError, match="token is missing"):
        fetch_remote_owner_readiness("gpu-b", cfg)
