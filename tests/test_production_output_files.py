"""Actual ComfyUI completion must not become an unverified PC-A image."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from artifex.comfy.errors import ComfyUIExecutionError, ComfyUIProtocolError
from artifex.comfy.models import ComfyExecutionResult, ComfyOutput, QueueReceipt
from artifex.config.models import ComfyUiConfig, ProductionConfig
from artifex.production.backend import ComfyGenerationBackend
from artifex.production.output_files import resolve_existing_comfy_outputs


def _output(
    filename: str = "scene.png", subfolder: str = "ARTIFEX/pack-1",
) -> ComfyOutput:
    return ComfyOutput(node_id="7", filename=filename, subfolder=subfolder)


def test_shared_comfy_output_accepts_nonempty_nested_image(tmp_path: Path) -> None:
    root = tmp_path / "PC A shared output"
    folder = root / "ARTIFEX" / "pack-1"
    folder.mkdir(parents=True)
    (folder / "scene.png").write_bytes(b"image content")
    assert resolve_existing_comfy_outputs(root, (_output(),)) == (
        (folder / "scene.png").resolve(),
    )
    assert resolve_existing_comfy_outputs(
        root, (_output(subfolder=r"ARTIFEX\pack-1"),)
    ) == ((folder / "scene.png").resolve(),)


@pytest.mark.parametrize("filename,subfolder", [
    ("../secret.png", "ARTIFEX"),
    ("scene.png", "ARTIFEX/../other"),
    ("scene.png", "../outside"),
    ("scene.png", "/outside"),
    ("scene.png", r"C:\Windows"),
    (r"C:\outside\scene.png", ""),
    (r"..\secret.png", ""),
    ("nested/scene.png", ""),
    ("scene.png", r"\\server\secret"),
    ("scene.png", "ARTIFEX/./pack-1"),
    ("scene.png:secret", ""),
    ("scene.png", "ARTIFEX:alternate"),
])
def test_remote_history_cannot_escape_configured_shared_directory(
    tmp_path: Path, filename: str, subfolder: str,
) -> None:
    with pytest.raises(ComfyUIProtocolError, match="unsafe or out-of-root"):
        resolve_existing_comfy_outputs(
            tmp_path, (_output(filename=filename, subfolder=subfolder),)
        )


@pytest.mark.parametrize("kind", ["missing", "zero_bytes", "directory"])
def test_completed_comfy_prompt_without_verified_file_is_not_a_success(
    tmp_path: Path, kind: str,
) -> None:
    folder = tmp_path / "ARTIFEX" / "pack-1"
    folder.mkdir(parents=True)
    image = folder / "scene.png"
    if kind == "zero_bytes":
        image.touch()
    elif kind == "directory":
        image.mkdir()
    with pytest.raises(ComfyUIExecutionError) as raised:
        resolve_existing_comfy_outputs(tmp_path, (_output(),))
    assert not raised.value.retryable
    assert "shared folder" in str(raised.value)


def test_absent_shared_root_is_nonretryable_instead_of_false_success(
    tmp_path: Path,
) -> None:
    with pytest.raises(ComfyUIExecutionError) as raised:
        resolve_existing_comfy_outputs(tmp_path / "unmounted", (_output(),))
    assert not raised.value.retryable
    assert "PC-A" in str(raised.value)


def test_shared_output_symlink_is_never_followed(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private")
    root = tmp_path / "share"
    folder = root / "ARTIFEX" / "pack-1"
    folder.mkdir(parents=True)
    link = folder / "scene.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Windows symlinks require elevation")
    with pytest.raises(ComfyUIProtocolError):
        resolve_existing_comfy_outputs(root, (_output(),))


def _backend(
    root: Path, *,
    mode: str = "filesystem",
    output: ComfyOutput | None = None,
) -> tuple[ComfyGenerationBackend, Any]:
    output = output or _output()

    class CompletedComfy:
        submissions = 0
        downloads = 0

        async def submit(self, graph: dict[str, Any]) -> QueueReceipt:
            self.submissions += 1
            return QueueReceipt(prompt_id="one-gpu-prompt")

        async def wait_for_completion(self, prompt_id: str) -> ComfyExecutionResult:
            assert prompt_id == "one-gpu-prompt"
            return ComfyExecutionResult(
                prompt_id=prompt_id, status="success", completed=True,
                outputs=(output,),
            )

        async def download_output(self, selected: ComfyOutput, destination: Path) -> Path:
            self.downloads += 1
            destination.mkdir(parents=True, exist_ok=True)
            image = destination / "scene.png"
            image.write_bytes(b"API response bytes")
            return image

        async def free_memory(self, **kwargs: Any) -> None:
            pass

    client = CompletedComfy()
    template = SimpleNamespace(
        template_id="illust_main_v1", version=1, model_family="ilxl",
        patch=lambda request: {"1": {"class_type": "SaveImage", "inputs": {}}},
    )
    config = ComfyUiConfig(
        output_mode=mode, output_dir=root,
        download_dir=root / "downloads",
        release_vram_after_attempt=False,
    )
    backend = ComfyGenerationBackend(
        client, SimpleNamespace(require=lambda name: template),
        ProductionConfig(checkpoint="checkpoint.safetensors"), config,
    )
    return backend, client


def _generation_request() -> Any:
    return SimpleNamespace(
        workflow_template_id=None,
        lora_plan=SimpleNamespace(entries=()),
        compiled=SimpleNamespace(
            positive_prompt="one girl", negative_prompt="artifact",
        ),
        seed=100, output_prefix="ARTIFEX/pack-1",
        scene_id="scene-a", attempt_id="attempt-1",
    )


@pytest.mark.asyncio
async def test_real_backend_only_returns_completed_shared_files(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "ARTIFEX" / "pack-1"
    folder.mkdir(parents=True)
    image = folder / "scene.png"
    image.write_bytes(b"saved ComfyUI image")
    backend, client = _backend(tmp_path)
    submitted: list[str] = []
    result = await backend.generate(
        _generation_request(), on_submitted=submitted.append,
    )
    assert result.output_paths == (image.resolve(),)
    assert result.prompt_id == "one-gpu-prompt"
    assert submitted == ["one-gpu-prompt"]
    assert client.submissions == 1
    assert client.downloads == 0


@pytest.mark.asyncio
async def test_real_backend_does_not_report_missing_shared_file_or_resubmit(
    tmp_path: Path,
) -> None:
    backend, client = _backend(tmp_path)
    submitted: list[str] = []
    with pytest.raises(ComfyUIExecutionError) as raised:
        await backend.generate(
            _generation_request(), on_submitted=submitted.append,
        )
    assert not raised.value.retryable
    assert submitted == ["one-gpu-prompt"]
    assert client.submissions == 1
    assert client.downloads == 0


@pytest.mark.asyncio
async def test_real_backend_keeps_remote_api_download_transport(
    tmp_path: Path,
) -> None:
    backend, client = _backend(tmp_path, mode="api")
    result = await backend.generate(
        _generation_request(), on_submitted=lambda _: None,
    )
    assert client.downloads == 1
    assert client.submissions == 1
    assert result.output_paths[0].read_bytes() == b"API response bytes"
