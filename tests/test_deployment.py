from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy import ComfyUIClient, WorkflowTemplateRegistry
from artifex.config.models import ArtifexSettings
from artifex.controller_preflight import ControllerPreflight, PreflightCheck
from artifex.deployment import (
    DeploymentCheck,
    DeploymentReport,
    _request,
    verify_deployment,
)
from artifex.render_node.preflight import RenderNodePreflight
from artifex.windows_tasks import StartupTaskStatus


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.production.checkpoint = "production.safetensors"
    settings.comfyui.base_url = "http://render.test:8188"
    settings.comfyui.output_mode = "api"
    settings.comfyui.default_template = "ilxl_base_v1"
    settings.production.repair_workflow_template = "ilxl_repair_v1"
    settings.comfyui.download_dir = tmp_path / "render-cache"
    settings.comfyui.poll_interval_seconds = 0.001
    return settings


def _controller(*, ready: bool = True) -> ControllerPreflight:
    return ControllerPreflight(
        ready=ready,
        renderer_id="gpu-b",
        checks=(
            PreflightCheck(
                name="comfyui_lan",
                ready=ready,
                detail="reachable" if ready else "connection refused",
            ),
        ),
    )


def _renderer(*, ready: bool = True) -> RenderNodePreflight:
    return RenderNodePreflight(
        ready=ready,
        node_id="gpu-b",
        comfyui_base_url="http://localhost:8188",
        comfyui_version="0.9.0",
        gpu_count=1,
        asset_labels=("production_checkpoint",),
        lora_count=4,
        issues=() if ready else ("ComfyUI unavailable",),
    )


def _task(*, installed: bool) -> StartupTaskStatus:
    return StartupTaskStatus(
        role="controller",
        task_name="Artifex-Controller",
        installed=installed,
        managed=installed,
        state="Ready" if installed else None,
    )


def _png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (64, 64)).save(stream, format="PNG")
    return stream.getvalue()


def _client(
    settings: ArtifexSettings,
    events: list[str],
    *,
    image: bytes | None = None,
    include_image: bool = True,
) -> tuple[ComfyUIClient, httpx.AsyncClient]:
    registry = WorkflowTemplateRegistry.with_packaged_templates()
    node_types: set[str] = set()
    for name in (
        settings.comfyui.default_template,
        settings.production.repair_workflow_template,
    ):
        template = registry.require(name)
        request = _request(settings, output_prefix="test", width=512, height=512)
        node_types.update(template.requirements(request).node_types)

    async def handler(request: httpx.Request) -> httpx.Response:
        events.append(request.url.path)
        if request.url.path == "/object_info":
            return httpx.Response(200, json={key: {} for key in node_types})
        if request.url.path == "/prompt":
            payload = json.loads(request.content.decode())
            assert "prompt" in payload
            assert isinstance(payload["prompt"], dict)
            return httpx.Response(200, json={"prompt_id": "smoke-123", "number": 0})
        if request.url.path == "/history/smoke-123":
            outputs: dict[str, object] = {}
            if include_image:
                outputs = {
                    "10": {
                        "images": [
                            {
                                "filename": "test-smoke.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                }
            return httpx.Response(
                200,
                json={
                    "smoke-123": {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": outputs,
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(
                200,
                content=image if image is not None else _png(),
                headers={"Content-Type": "image/png"},
            )
        raise AssertionError(f"Unexpected ComfyUI request: {request.url}")

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport, base_url=settings.comfyui.base_url
    )
    return ComfyUIClient(settings.comfyui, client=http), http


@pytest.mark.asyncio
async def test_read_only_controller_deployment_checks_both_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.deployment.platform.system", lambda: "Linux")
    settings = _settings(tmp_path)
    events: list[str] = []
    client, http = _client(settings, events)
    async with http:
        result = await verify_deployment(
            settings,
            role="controller",
            controller_check=lambda _: _controller(),
            comfy_client=client,
        )
    assert result.ready
    assert result.smoke is None
    assert events == ["/object_info", "/object_info"]
    assert len([item for item in result.checks if item.name.startswith("workflow:")]) == 2
    assert not settings.comfyui.download_dir.exists()


@pytest.mark.asyncio
async def test_explicit_smoke_submits_real_graph_and_validates_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.deployment.platform.system", lambda: "Linux")
    settings = _settings(tmp_path)
    events: list[str] = []
    client, http = _client(settings, events)
    async with http:
        result = await verify_deployment(
            settings,
            role="controller",
            controller_check=lambda _: _controller(),
            comfy_client=client,
            render_smoke=True,
            width=512,
            height=512,
        )
    assert result.ready
    assert result.smoke is not None
    assert result.smoke.prompt_id == "smoke-123"
    assert result.smoke.bytes > 0
    assert result.smoke.image_path.is_file()
    assert result.smoke.image_path.is_relative_to(
        settings.comfyui.download_dir / "deployment-smoke"
    )
    assert len(result.smoke.image_sha256) == 64
    assert result.smoke.width == 64 and result.smoke.height == 64
    assert "/prompt" in events and "/history/smoke-123" in events and "/view" in events


@pytest.mark.asyncio
async def test_broken_smoke_output_is_failure_not_false_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.deployment.platform.system", lambda: "Linux")
    settings = _settings(tmp_path)
    events: list[str] = []
    client, http = _client(settings, events, image=b"not a PNG")
    async with http:
        result = await verify_deployment(
            settings,
            role="controller",
            controller_check=lambda _: _controller(),
            comfy_client=client,
            render_smoke=True,
        )
    assert not result.ready
    assert result.smoke is None
    assert any(
        item.name == "render_smoke" and not item.ready
        for item in result.checks
    )
    assert not list(settings.comfyui.download_dir.rglob("*.png"))


@pytest.mark.asyncio
async def test_zero_image_outputs_fail_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.deployment.platform.system", lambda: "Linux")
    settings = _settings(tmp_path)
    events: list[str] = []
    client, http = _client(settings, events, include_image=False)
    async with http:
        report = await verify_deployment(
            settings, role="controller",
            controller_check=lambda _: _controller(),
            comfy_client=client, render_smoke=True,
        )
    assert not report.ready
    assert "no output images" in report.checks[-1].detail
    assert "/view" not in events


@pytest.mark.asyncio
async def test_no_gpu_work_submitted_when_prerequisites_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.deployment.platform.system", lambda: "Linux")
    settings = _settings(tmp_path)
    events: list[str] = []
    client, http = _client(settings, events)
    async with http:
        result = await verify_deployment(
            settings, role="controller",
            controller_check=lambda _: _controller(ready=False),
            comfy_client=client, render_smoke=True,
        )
    assert not result.ready
    assert result.smoke is None
    assert "/prompt" not in events
    assert "Skipped" in result.checks[-1].detail


@pytest.mark.asyncio
async def test_renderer_role_checks_local_preflight_without_gpu_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.deployment.platform.system", lambda: "Linux")
    report = await verify_deployment(
        _settings(tmp_path),
        role="renderer",
        renderer_check=lambda _: _renderer(),
    )
    assert report.ready
    assert report.smoke is None
    assert [item.name for item in report.checks] == ["renderer:preflight"]


@pytest.mark.asyncio
async def test_renderer_smoke_is_rejected_before_any_side_effect(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="role controller"):
        await verify_deployment(
            _settings(tmp_path), role="renderer", render_smoke=True
        )


@pytest.mark.asyncio
async def test_smoke_options_require_opt_in(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require --render-smoke"):
        await verify_deployment(
            _settings(tmp_path), role="controller", width=512
        )


@pytest.mark.asyncio
async def test_autostart_is_nonblocking_unless_explicitly_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.deployment.platform.system", lambda: "Windows")
    async def run(required: bool) -> DeploymentReport:
        return await verify_deployment(
            _settings(tmp_path), role="renderer",
            require_autostart=required,
            renderer_check=lambda _: _renderer(),
            startup_check=lambda _: _task(installed=False),
        )

    ordinary = await run(False)
    required = await run(True)
    assert ordinary.ready
    assert not required.ready
    check = next(item for item in required.checks if item.name == "windows_autostart")
    assert check.blocking and not check.ready


@pytest.mark.asyncio
async def test_controller_checker_exception_returns_structured_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.deployment.platform.system", lambda: "Linux")

    def fail(_: ArtifexSettings) -> ControllerPreflight:
        raise TimeoutError("renderer unreachable")

    settings = _settings(tmp_path)
    events: list[str] = []
    client, http = _client(settings, events)
    async with http:
        report = await verify_deployment(
            settings, role="controller", controller_check=fail, comfy_client=client
        )
    assert not report.ready
    assert "renderer unreachable" in report.checks[0].detail


def test_deployment_cli_refuses_missing_config_without_creating_it(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    result = CliRunner().invoke(
        app, ["deployment", "verify", "--config", str(missing), "--json"]
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert not missing.exists()


def test_deployment_cli_rejects_invalid_role() -> None:
    result = CliRunner().invoke(
        app, ["deployment", "verify", "--role", "not-a-role"]
    )
    assert result.exit_code == 1
    assert "must be controller or renderer" in result.output


def test_deployment_cli_preserves_failed_checks_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("production:\n  checkpoint: production.safetensors\n", encoding="utf-8")
    seen: list[dict[str, Any]] = []

    async def fake_verify(_settings: ArtifexSettings, **kwargs: Any) -> DeploymentReport:
        seen.append(kwargs)
        return DeploymentReport(
            role="controller",
            ready=False,
            checks=(
                DeploymentCheck(
                    name="controller:comfyui_lan",
                    ready=False,
                    blocking=True,
                    detail="offline",
                ),
            ),
        )

    monkeypatch.setattr("artifex.cli.verify_deployment", fake_verify)
    result = CliRunner().invoke(
        app,
        [
            "deployment", "verify", "--config", str(config),
            "--require-autostart", "--render-smoke", "--width", "512",
        ],
    )
    assert result.exit_code == 1
    assert '"ready": false' in result.output
    assert "offline" in result.output
    assert seen[0]["require_autostart"] is True
    assert seen[0]["render_smoke"] is True
    assert seen[0]["width"] == 512
