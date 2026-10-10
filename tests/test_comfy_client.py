from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from artifex.comfy import (
    ComfyErrorKind,
    ComfyOutput,
    ComfyUIClient,
    ComfyUIError,
    WorkflowAssetRequirement,
    WorkflowPatchRequest,
    WorkflowRequirements,
    WorkflowTemplateRegistry,
)
from artifex.config.models import ComfyUiConfig


def _config(**updates: Any) -> ComfyUiConfig:
    values: dict[str, Any] = {
        "base_url": "http://comfy.test",
        "request_attempts": 2,
        "reconnect_backoff_seconds": 0,
        "poll_interval_seconds": 0.001,
        "execution_timeout_seconds": 0.05,
    }
    values.update(updates)
    return ComfyUiConfig(**values)



@pytest.mark.asyncio
async def test_comfyui_gateway_bearer_token_sent_only_when_explicitly_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "approved-private-lan-gateway-secret"
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", token)
    headers: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers.get("Authorization"))
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "auth-123"})
        return httpx.Response(
            200, json={"queue_running": [], "queue_pending": []}
        )

    config = _config(
        submission_fence_path=tmp_path / "auth-fence.sqlite",
        gateway_token_env="ARTIFEX_RENDER_NODE_TOKEN",
    )
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    gateway_client = ComfyUIClient(config, client=async_client)
    assert (await gateway_client.submit({"1": {"class_type": "Test", "inputs": {}}})).prompt_id == "auth-123"
    await gateway_client.queue_snapshot()
    assert headers == ["Bearer " + token] * 2
    await async_client.aclose()

    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN")
    with pytest.raises(ValueError, match="missing"):
        ComfyUIClient(config)
    assert config.gateway_token_env == "ARTIFEX_RENDER_NODE_TOKEN"

@pytest.mark.asyncio
async def test_health_parses_current_comfy_system_stats() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/system_stats"
        return httpx.Response(
            200,
            json={
                "system": {"comfyui_version": "0.9.0"},
                "devices": [{"name": "RTX 3060"}],
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)

    health = await client.health()

    assert health.available is True
    assert health.version == "0.9.0"
    assert health.devices == ("RTX 3060",)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_submit_classifies_invalid_workflow() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/prompt"
        return httpx.Response(
            400,
            json={
                "error": {"type": "prompt_outputs_failed_validation"},
                "node_errors": {"5": {"errors": ["bad sampler"]}},
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)

    with pytest.raises(ComfyUIError) as exc_info:
        await client.submit({"1": {"class_type": "Bad", "inputs": {}}})

    assert exc_info.value.kind is ComfyErrorKind.INVALID_WORKFLOW
    assert exc_info.value.retryable is False
    await http_client.aclose()


@pytest.mark.asyncio
async def test_request_retries_transient_connection_failure() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary", request=request)
        return httpx.Response(200, json={"system": {}, "devices": []})

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(request_attempts=2), client=http_client)

    health = await client.health()

    assert health.available is True
    assert calls == 2
    await http_client.aclose()


@pytest.mark.asyncio
async def test_ambiguous_prompt_post_is_never_retried_as_duplicate_gpu_work() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        assert request.method == "POST"
        assert request.url.path == "/prompt"
        calls += 1
        # Remote ComfyUI could already have queued the GPU job when the
        # response was lost. A second POST would generate a duplicate.
        raise httpx.ReadTimeout("response lost after enqueue", request=request)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(request_attempts=5), client=http_client)
    with pytest.raises(ComfyUIError) as exc:
        await client.submit({"1": {"class_type": "SaveImage", "inputs": {}}})
    assert exc.value.kind is ComfyErrorKind.CONNECTION
    assert not exc.value.retryable
    assert "outcome is unknown" in str(exc.value)
    assert calls == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_prompt_server_error_is_never_automatically_resent() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        assert request.url.path == "/prompt"
        calls += 1
        return httpx.Response(503, text="transient failure")

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(request_attempts=5), client=http_client)
    with pytest.raises(ComfyUIError) as exc:
        await client.submit({"1": {"class_type": "SaveImage", "inputs": {}}})
    assert not exc.value.retryable
    assert calls == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_execute_submits_tracks_and_discovers_outputs() -> None:
    submitted_graph: dict[str, Any] = {}
    history_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal history_calls
        if request.url.path == "/prompt":
            payload = json.loads(request.content)
            submitted_graph.update(payload["prompt"])
            return httpx.Response(
                200,
                json={"prompt_id": "prompt-1", "number": 4, "node_errors": {}},
            )
        if request.url.path == "/history/prompt-1":
            history_calls += 1
            if history_calls == 1:
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "prompt-1": {
                        "status": {
                            "status_str": "success",
                            "completed": True,
                            "messages": [],
                        },
                        "outputs": {
                            "7": {
                                "images": [
                                    {
                                        "filename": "scene_00001_.png",
                                        "subfolder": "ARTIFEX/pack-1",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                },
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)
    template = WorkflowTemplateRegistry.with_packaged_templates().require("ilxl_base_v1")
    patch = WorkflowPatchRequest(
        positive_prompt="amane_kanata",
        negative_prompt="bad_hands",
        checkpoint="model.safetensors",
        seed=7,
        width=1024,
        height=1024,
        output_prefix="ARTIFEX/pack-1/scene-1",
    )

    result = await client.execute(template, patch)

    assert submitted_graph["2"]["inputs"]["text"] == "amane_kanata"
    assert result.completed is True
    assert result.status == "success"
    assert len(result.outputs) == 1
    assert result.outputs[0].filename == "scene_00001_.png"
    assert history_calls == 2
    await http_client.aclose()


@pytest.mark.asyncio
async def test_cancel_uses_queue_delete_and_targeted_interrupt() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(200)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)

    await client.cancel("prompt-1")

    assert calls == [
        ("/queue", {"delete": ["prompt-1"]}),
        ("/interrupt", {"prompt_id": "prompt-1"}),
    ]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_wait_timeout_is_retryable_infrastructure_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/history/prompt-1"
        return httpx.Response(200, json={})

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)

    with pytest.raises(ComfyUIError) as exc_info:
        await client.wait_for_completion("prompt-1", timeout_seconds=0.003)

    assert exc_info.value.kind is ComfyErrorKind.TIMEOUT
    assert exc_info.value.retryable is True
    await http_client.aclose()



@pytest.mark.asyncio
async def test_object_info_validates_required_nodes_and_assets() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info"
        return httpx.Response(
            200,
            json={
                "CheckpointLoaderSimple": {
                    "input": {
                        "required": {
                            "ckpt_name": [
                                ["ilxl.safetensors", "other.safetensors"],
                                {},
                            ]
                        }
                    }
                },
                "LoraLoader": {
                    "input": {
                        "required": {
                            "lora_name": [
                                [
                                    "characters/kanata.safetensors",
                                    "style.safetensors",
                                ],
                                {},
                            ]
                        }
                    }
                },
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)
    requirements = WorkflowRequirements(
        template_id="production",
        source_sha256="a" * 64,
        node_types=("CheckpointLoaderSimple", "LoraLoader"),
        assets=(
            WorkflowAssetRequirement(
                label="checkpoint",
                node_class="CheckpointLoaderSimple",
                input_name="ckpt_name",
                value="ilxl.safetensors",
            ),
            WorkflowAssetRequirement(
                label="lora",
                node_class="LoraLoader",
                input_name="lora_name",
                value="kanata.safetensors",
            ),
        ),
    )

    status = await client.validate_requirements(requirements)

    assert status.ready is True
    assert status.missing_node_types == ()
    assert status.missing_assets == ()
    await http_client.aclose()


@pytest.mark.asyncio
async def test_object_info_reports_missing_custom_node_and_model() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info"
        return httpx.Response(
            200,
            json={
                "CheckpointLoaderSimple": {
                    "input": {
                        "required": {
                            "ckpt_name": [["other.safetensors"], {}]
                        }
                    }
                }
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)
    requirements = WorkflowRequirements(
        template_id="production",
        source_sha256="b" * 64,
        node_types=("CheckpointLoaderSimple", "MissingCustomNode"),
        assets=(
            WorkflowAssetRequirement(
                label="checkpoint",
                node_class="CheckpointLoaderSimple",
                input_name="ckpt_name",
                value="ilxl.safetensors",
            ),
        ),
    )

    status = await client.validate_requirements(requirements)

    assert status.ready is False
    assert status.missing_node_types == ("MissingCustomNode",)
    assert status.missing_assets == ("checkpoint:ilxl.safetensors",)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_free_memory_unloads_models_and_cached_vram() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)

    await client.free_memory()

    assert calls == [
        (
            "/free",
            {
                "unload_models": True,
                "free_memory": True,
            },
        )
    ]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_download_output_fetches_remote_image_through_view_api(
    tmp_path: Path,
) -> None:
    image_bytes = b"remote-image-bytes"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/view"
        assert request.url.params["filename"] == "scene.png"
        assert request.url.params["subfolder"] == "ARTIFEX/pack-1"
        assert request.url.params["type"] == "output"
        return httpx.Response(200, content=image_bytes)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)
    output = ComfyOutput(
        node_id="7",
        filename="scene.png",
        subfolder="ARTIFEX/pack-1",
        output_type="output",
    )

    path = await client.download_output(output, tmp_path / "render-cache")

    assert path.read_bytes() == image_bytes
    assert path.relative_to(tmp_path / "render-cache").as_posix() == (
        "ARTIFEX/pack-1/scene.png"
    )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_download_output_rejects_truncated_image_without_publishing(
    tmp_path: Path,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"short", headers={"Content-Length": "50"}
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://comfy.test",
    )
    client = ComfyUIClient(_config(), client=http_client)
    output = ComfyOutput(
        node_id="1", filename="scene.png", subfolder="ARTIFEX", output_type="output"
    )
    with pytest.raises(ComfyUIError, match="truncated"):
        await client.download_output(output, tmp_path / "cache")
    assert not (tmp_path / "cache" / "ARTIFEX" / "scene.png").exists()
    assert not (tmp_path / "cache" / "ARTIFEX" / "scene.png.part").exists()
    await http_client.aclose()
