from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy import ComfyUIClient, WorkflowTemplateRegistry
from artifex.comfy.workflow_audit import audit_workflows
from artifex.config.models import ArtifexSettings
from artifex.deployment import _request


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.comfyui.base_url = "http://comfy.test:8188"
    settings.comfyui.default_template = "illust_main_v1"
    settings.production.repair_workflow_template = "illust_main_repair_v1"
    settings.production.checkpoint = "selected.safetensors"
    return settings


def _object_info(settings: ArtifexSettings) -> dict[str, Any]:
    info: dict[str, Any] = {}
    registry = WorkflowTemplateRegistry.with_packaged_templates()
    for name in (
        settings.comfyui.default_template, settings.production.repair_workflow_template
    ):
        requirement = registry.require(name).requirements(
            _request(
                settings,
                output_prefix="test",
                width=settings.production.width,
                height=settings.production.height,
            )
        )
        for node_name in requirement.node_types:
            info.setdefault(node_name, {"input": {"required": {}}})
        for asset in requirement.assets:
            node = info.setdefault(asset.node_class, {"input": {"required": {}}})
            fields = node["input"]["required"]
            if asset.input_name not in fields:
                fields[asset.input_name] = [[asset.value]]
            elif asset.value not in fields[asset.input_name][0]:
                fields[asset.input_name][0].append(asset.value)
    return info


def _client(
    settings: ArtifexSettings, info: dict[str, Any],
    events: list[str]
) -> tuple[ComfyUIClient, httpx.AsyncClient]:
    async def handler(request: httpx.Request) -> httpx.Response:
        events.append(request.method + " " + request.url.path)
        assert request.url.path == "/object_info"
        assert request.method == "GET"
        return httpx.Response(200, json=info)

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=settings.comfyui.base_url,
    )
    return ComfyUIClient(settings.comfyui, client=http), http


@pytest.mark.asyncio
async def test_all_required_nodes_and_model_choices_pass_without_gpu_submit() -> None:
    settings = _settings()
    events: list[str] = []
    client, http = _client(settings, _object_info(settings), events)
    async with http:
        result = await audit_workflows(settings, client=client)
    assert result.ready
    assert len(result.entries) == 2
    assert result.entries[0].required_node_count > 20
    assert result.entries[0].required_asset_count > 0
    assert all(e.ready and not e.unverifiable_assets for e in result.entries)
    assert events == ["GET /object_info"]


@pytest.mark.asyncio
async def test_missing_custom_node_class_is_reported_exactly() -> None:
    settings = _settings()
    info = _object_info(settings)
    missing = next(key for key in info if key != "LoraLoader")
    del info[missing]
    events: list[str] = []
    client, http = _client(settings, info, events)
    async with http:
        report = await audit_workflows(settings, client=client)
    assert not report.ready
    assert any(missing in entry.missing_node_types for entry in report.entries)
    assert any("repositories separately" in note for note in report.guidance)


@pytest.mark.asyncio
async def test_missing_model_from_comfy_enum_reported_with_available_choices() -> None:
    settings = _settings()
    info = _object_info(settings)
    assert "CheckpointLoaderSimple" in info
    info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"] = [
        ["other_checkpoint.safetensors"]
    ]
    events: list[str] = []
    client, http = _client(settings, info, events)
    async with http:
        report = await audit_workflows(settings, client=client)
    assert not report.ready
    matches = [
        asset
        for entry in report.entries
        for asset in entry.missing_assets
        if asset.node_class == "CheckpointLoaderSimple"
    ]
    assert matches
    assert all("other_checkpoint.safetensors" in asset.available for asset in matches)


@pytest.mark.asyncio
async def test_absent_loader_choices_are_unknown_not_false_success() -> None:
    settings = _settings()
    info = _object_info(settings)
    info["CheckpointLoaderSimple"]["input"]["required"] = {}
    events: list[str] = []
    client, http = _client(settings, info, events)
    async with http:
        report = await audit_workflows(settings, client=client)
    assert not report.ready
    unknown = [a for e in report.entries for a in e.unverifiable_assets]
    assert any(a.node_class == "CheckpointLoaderSimple" for a in unknown)
    assert any("unverified" in message for message in report.guidance)


@pytest.mark.asyncio
async def test_custom_node_directory_listing_does_not_guess_class_mapping(
    tmp_path: Path
) -> None:
    settings = _settings()
    folder = tmp_path / "custom_nodes"
    folder.mkdir()
    (folder / "custom-foo").mkdir()
    (folder / ".git").mkdir()
    events: list[str] = []
    client, http = _client(settings, _object_info(settings), events)
    async with http:
        report = await audit_workflows(
            settings, client=client, custom_nodes_root=folder,
        )
    assert report.ready
    assert report.custom_node_folders == ("custom-foo",)


def test_workflow_audit_cli_refuses_broken_endpoint_with_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("production:\n  checkpoint: foo.safetensors\n", encoding="utf-8")

    async def fail(_: ArtifexSettings, **kwargs: Any) -> None:
        raise ValueError("ComfyUI /object_info unreachable")

    monkeypatch.setattr("artifex.cli.audit_workflows", fail)
    result = CliRunner().invoke(
        app, ["onboard", "workflow-audit", "--config", str(config)]
    )
    assert result.exit_code == 1
    assert "unreachable" in result.output
