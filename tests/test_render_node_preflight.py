from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from artifex.config.models import ArtifexSettings, ComfyUiConfig, RenderAgentConfig
from artifex.render_node.models import (
    RenderAssetDigest,
    RenderLoRAInventoryItem,
    RenderNodeAttestation,
)
from artifex.render_node.preflight import check_render_node

_REQUIRED = (
    "production_checkpoint",
    "refiner_checkpoint",
    "vae",
    "upscale_model",
)


def _settings(root: Path) -> ArtifexSettings:
    return ArtifexSettings(
        render_agent=RenderAgentConfig(
            node_id="rtx3060",
            bind_host="0.0.0.0",
            lora_roots=(root,),
        ),
        comfyui=ComfyUiConfig(base_url="http://comfy.test:8188"),
    )


def _attestation(
    *,
    missing: str | None = None,
    errors: tuple[dict[str, str], ...] = (),
) -> RenderNodeAttestation:
    return RenderNodeAttestation(
        node_id="rtx3060",
        created_at=datetime.now(UTC),
        hostname="render-pc",
        os={"system": "Windows", "release": "11", "version": "11", "machine": "AMD64"},
        nvidia_gpus=({"name": "RTX 3060", "memory_total_mib": 12288},),
        comfyui_base_url="http://comfy.test:8188",
        assets=tuple(
            RenderAssetDigest(label=name, path=f"D:/models/{name}", sha256="a" * 64, bytes=100)
            for name in _REQUIRED
            if name != missing
        ),
        loras=(
            RenderLoRAInventoryItem(
                name="test.safetensors",
                relative_path="test.safetensors",
                path="D:/models/loras/test.safetensors",
                sha256="b" * 64,
                bytes=100,
            ),
        ),
        inventory_errors=errors,
    )


def test_render_preflight_accepts_healthy_windows_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "secret")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/system_stats"
        return httpx.Response(200, json={"system": {"comfyui_version": "0.9.0"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = check_render_node(
            _settings(tmp_path),
            client=client,
            attestation=_attestation(),
        )

    assert report.ready is True
    assert report.comfyui_version == "0.9.0"
    assert report.gpu_count == 1
    assert report.lora_count == 1
    assert not report.issues


def test_render_preflight_reports_missing_models_token_and_lora_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    config = _settings(tmp_path)
    config.render_agent.bind_host = "127.0.0.1"
    errors = ({"path": "D:/models/loras", "error": "access denied"},)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = check_render_node(
            config, client=client, attestation=_attestation(missing="vae", errors=errors)
        )

    assert report.ready is False
    assert any("vae" in issue for issue in report.issues)
    assert any("bearer token" in issue for issue in report.issues)
    assert any("loopback-only" in issue for issue in report.issues)
    assert any("access denied" in issue for issue in report.issues)
    assert any("/system_stats" in issue for issue in report.issues)
