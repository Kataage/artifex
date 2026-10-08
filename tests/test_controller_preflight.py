from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import (
    ArtifexSettings,
    RenderNodeConfig,
    RenderNodesConfig,
)
from artifex.controller_preflight import ControllerPreflight, PreflightCheck, check_controller
from artifex.render_node.models import (
    RenderAssetDigest,
    RenderLoRAInventoryItem,
    RenderNodeAttestation,
)


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings(
        render_nodes=RenderNodesConfig(
            primary="render-b",
            nodes={
                "render-b": RenderNodeConfig(
                    base_url="http://pc-b.test:8188",
                    attestation_url="http://pc-b.test:8190",
                    output_mode="api",
                )
            },
        )
    )
    settings.production.checkpoint = "production.safetensors"
    settings.comfyui.refiner_checkpoint = "refiner.safetensors"
    settings.comfyui.vae = "vae.safetensors"
    settings.comfyui.upscale_model = "upscaler.pt"
    settings.llm.base_url = "http://localhost:8899"
    settings.llm.bootstrap.enabled = False
    return settings


def _attestation(
    *,
    created_at: datetime | None = None,
    inventory_errors: tuple[dict[str, str], ...] = (),
    missing: str | None = None,
    checkpoint: str = "production.safetensors",
) -> RenderNodeAttestation:
    filenames = {
        "production_checkpoint": checkpoint,
        "refiner_checkpoint": "refiner.safetensors",
        "vae": "vae.safetensors",
        "upscale_model": "upscaler.pt",
    }
    return RenderNodeAttestation(
        node_id="render-b",
        created_at=created_at or datetime.now(UTC),
        hostname="render-box",
        os={"system": "Windows"},
        nvidia_gpus=({"name": "RTX 3060"},),
        comfyui_base_url="http://127.0.0.1:8188",
        assets=tuple(
            RenderAssetDigest(
                label=label,
                path=f"D:\\ComfyUI\\models\\{filename}",
                sha256="a" * 64,
                bytes=1024,
            )
            for label, filename in filenames.items()
            if label != missing
        ),
        loras=(
            RenderLoRAInventoryItem(
                name="char.safetensors",
                path="D:/ComfyUI/models/loras/char.safetensors",
                relative_path="char.safetensors",
                sha256="b" * 64,
                bytes=1024,
            ),
        ),
        inventory_errors=inventory_errors,
    )


def _client(attestation: RenderNodeAttestation, *, authorized: bool = True) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 8188 and request.url.path == "/system_stats":
            return httpx.Response(
                200, json={"system": {"comfyui_version": "0.9.0"}}
            )
        if request.url.port == 8190 and request.url.path == "/v1/attestation":
            if request.headers.get("Authorization") != "Bearer secret" or not authorized:
                return httpx.Response(401)
            return httpx.Response(200, json=attestation.model_dump(mode="json"))
        if request.url.port == 8899 and request.url.path == "/health":
            return httpx.Response(200)
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_controller_preflight_checks_authenticated_two_pc_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "secret")
    with _client(_attestation()) as client:
        report = check_controller(_settings(tmp_path), client=client)
    assert report.ready is True
    assert report.renderer_id == "render-b"
    assert all(check.ready for check in report.checks)
    assert {item.name for item in report.checks} >= {
        "comfyui_lan",
        "render_token",
        "render_attestation",
        "render_inventory",
        "asset_production_checkpoint",
        "asset_vae",
        "asset_upscale_model",
        "llm_lan",
    }
    assert "secret" not in report.model_dump_json()


def test_controller_preflight_reports_missing_token_without_credentials_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    with _client(_attestation()) as client:
        report = check_controller(_settings(tmp_path), client=client)
    assert report.ready is False
    assert next(c for c in report.checks if c.name == "render_token").ready is False
    assert all(c.name != "render_attestation" for c in report.checks)


def test_controller_preflight_rejects_unauthorized_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "secret")
    with _client(_attestation(), authorized=False) as client:
        report = check_controller(_settings(tmp_path), client=client)
    assert report.ready is False
    assert "401" in next(
        c for c in report.checks if c.name == "render_attestation"
    ).detail


def test_controller_preflight_detects_model_drift_and_incomplete_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "secret")
    evidence = _attestation(
        checkpoint="wrong.safetensors",
        missing="vae",
        inventory_errors=({"path": "D:/loras", "error": "access denied"},),
        created_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    with _client(evidence) as client:
        report = check_controller(_settings(tmp_path), client=client)
    assert report.ready is False
    detail = "\n".join(check.detail for check in report.checks if not check.ready)
    assert "filename mismatch" in detail
    assert "vae is missing" in detail
    assert "access denied" in detail
    assert "stale" in detail


def test_controller_preflight_requires_production_primary(tmp_path: Path) -> None:
    report = check_controller(ArtifexSettings())
    assert report.ready is False
    assert report.renderer_id is None
    assert report.checks[0].name == "renderer_configuration"


def test_controller_preflight_checks_local_gguf_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "secret")
    settings = _settings(tmp_path)
    settings.llm.bootstrap.enabled = True
    settings.llm.bootstrap.models_dir = tmp_path / "models"
    with _client(_attestation()) as client:
        missing = check_controller(settings, client=client)
    assert not missing.ready
    assert not next(c for c in missing.checks if c.name == "llm_model_file").ready
    path = settings.llm.bootstrap.model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test gguf")
    with _client(_attestation()) as client:
        present = check_controller(settings, client=client)
    assert present.ready


def test_controller_preflight_cli_exit_code_and_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_check(_settings: ArtifexSettings) -> ControllerPreflight:
        return ControllerPreflight(
            ready=False,
            renderer_id="render-b",
            checks=(PreflightCheck(name="comfyui_lan", ready=False, detail="offline"),),
        )

    monkeypatch.setattr("artifex.cli.check_controller", fake_check)
    result = CliRunner().invoke(app, ["preflight", "--json"])
    assert result.exit_code == 1
    assert '"ready": false' in result.output
    assert "offline" in result.output
