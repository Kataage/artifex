"""Read-only durable PC-B passive survival transport and CLI coverage."""
from __future__ import annotations

import hashlib
import json
import socket
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.render_node.attestation import serve_attestation
from artifex.render_node.client import fetch_remote_survival_trace
from artifex.render_node.survival_spool import (
    automatic_survival_output,
    latest_survival_trace,
)

_FIRST = "survival-20261010T100000Z-11111111111111111111111111111111.json"
_SECOND = "survival-20261010T110000Z-22222222222222222222222222222222.json"


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_agent.node_id = "gpu-b"
    settings.render_agent.survival_evidence_dir = tmp_path / "survival-spool"
    settings.render_agent.token_env = "ARTIFEX_RENDER_NODE_TOKEN"
    settings.render_agent.require_token = True
    settings.render_agent.bind_host = "127.0.0.1"
    return settings


def test_auto_output_is_unique_and_within_configured_pc_b_root(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    first = automatic_survival_output(settings)
    second = automatic_survival_output(settings)
    assert first != second
    assert first.parent == settings.render_agent.survival_evidence_dir
    assert first.name.startswith("survival-")
    assert first.suffix == ".json"


def test_latest_is_deterministic_and_does_not_fallback(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    root = settings.render_agent.survival_evidence_dir
    root.mkdir()
    first = root / _FIRST
    second = root / _SECOND
    first.write_text('{"state":"old-success"}', encoding="utf-8")
    second.write_text('{"state":"latest-blocked"}', encoding="utf-8")
    (root / "not-an-observer.json").write_text('{"secret":"not served"}')
    digest, content = latest_survival_trace(settings)
    assert content == '{"state":"latest-blocked"}'
    assert digest == hashlib.sha256(content.encode()).hexdigest()
    second.write_text("x" * (12 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="size invalid"):
        latest_survival_trace(settings)


def test_symlink_and_inventory_fail_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = settings.render_agent.survival_evidence_dir
    root.mkdir()
    (root / _FIRST).write_text('{"status":"inconclusive"}')
    latest = root / _SECOND
    try:
        latest.symlink_to(root / _FIRST)
    except (OSError, NotImplementedError):
        pytest.skip("Native Windows permissions disallow symlink creation")
    with pytest.raises(ValueError, match="unsafe"):
        latest_survival_trace(settings)
    latest.unlink()
    for i in range(513):
        filename = f"survival-20261010T120000Z-{i:032x}.json"
        (root / filename).write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        latest_survival_trace(settings)


def test_protected_http_get_returns_only_latest_trace_and_never_runs_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    root = settings.render_agent.survival_evidence_dir
    root.mkdir()
    (root / _FIRST).write_text('{"status":"blocked"}', encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        settings.render_agent.port = sock.getsockname()[1]
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "secret-for-test")
    stop = threading.Event()
    server = threading.Thread(
        target=serve_attestation, args=(settings,),
        kwargs={"stop_event": stop}, daemon=True,
    )
    server.start()
    url = f"http://127.0.0.1:{settings.render_agent.port}"
    try:
        with httpx.Client(trust_env=False, timeout=4) as http:
            for _ in range(25):
                try:
                    if http.get(url + "/health").status_code == 200:
                        break
                except httpx.ConnectError:
                    stop.wait(0.03)
            assert http.get(url + "/v1/survival-evidence").status_code == 401
            token = {"Authorization": "Bearer secret-for-test"}
            first = http.get(url + "/v1/survival-evidence", headers=token)
            assert first.status_code == 200
            payload = first.json()
            assert payload["node_id"] == "gpu-b"
            assert payload["content"] == '{"status":"blocked"}'
            assert payload["sha256"] == hashlib.sha256(
                payload["content"].encode(),
            ).hexdigest()
            query = http.get(
                url + "/v1/survival-evidence?file=../../private.txt",
                headers=token,
            )
            assert query.status_code == 400
            assert "private" not in query.text
            (root / _SECOND).write_text('{"status":"inconclusive"}')
            newer = http.get(url + "/v1/survival-evidence", headers=token)
            assert newer.json()["content"] == '{"status":"inconclusive"}'
            (root / _SECOND).unlink()
            (root / _FIRST).unlink()
            assert http.get(
                url + "/v1/survival-evidence", headers=token,
            ).status_code == 503
    finally:
        stop.set()
        server.join(timeout=5)


def test_pc_a_client_enforces_bearer_no_redirect_node_hash_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "pc-b-token")
    cfg = RenderNodeConfig(
        base_url="http://localhost:8188", attestation_url="http://localhost:8190",
    )
    content = '{"status":"blocked"}'
    doc = {
        "node_id": "gpu-b",
        "content": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    def respond(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/survival-evidence"
        assert not req.url.query
        assert req.headers["authorization"] == "Bearer pc-b-token"
        return httpx.Response(200, json=doc)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        received = fetch_remote_survival_trace("gpu-b", cfg, client=http)
    assert received.content == content

    def wrong(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**doc, "node_id": "foreign"})

    with (
        httpx.Client(transport=httpx.MockTransport(wrong)) as http,
        pytest.raises(ValueError, match="node ID"),
    ):
        fetch_remote_survival_trace("gpu-b", cfg, client=http)

    def redirect(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.invalid/"})

    with (
        httpx.Client(transport=httpx.MockTransport(redirect)) as http,
        pytest.raises(httpx.HTTPStatusError),
    ):
        fetch_remote_survival_trace("gpu-b", cfg, client=http)

    def oversized(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (13 * 1024 * 1024 + 1))

    with (
        httpx.Client(transport=httpx.MockTransport(oversized)) as http,
        pytest.raises(ValueError, match="size limit"),
    ):
        fetch_remote_survival_trace("gpu-b", cfg, client=http)
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN")
    with pytest.raises(ValueError, match="token is missing"):
        fetch_remote_survival_trace("gpu-b", cfg)


def test_default_pc_b_cli_writes_generated_spool_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from artifex import cli
    from artifex.render_node import supervisor_survival

    settings = _settings(tmp_path)
    config = tmp_path / "renderer.yaml"
    config.write_text("{}")
    monkeypatch.setattr(cli, "_settings", lambda _: settings)
    requested: list[Path] = []

    class Report:
        status = "inconclusive"

        def model_dump(self, *, mode: str) -> dict[str, Any]:
            return {"status": self.status, "production_qualified": False}

    def observe(*args: Any, **kwargs: Any) -> Report:
        requested.append(kwargs["output"])
        return Report()

    monkeypatch.setattr(supervisor_survival, "observe_supervisor_survival", observe)
    result = CliRunner().invoke(app, [
        "render-node", "survival-observe",
        "--config", str(config), "--duration-seconds", "20",
        "--interval-seconds", "10", "--json",
    ])
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["production_qualified"] is False
    assert len(requested) == 1
    assert requested[0].parent == settings.render_agent.survival_evidence_dir
