from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config import load_settings
from artifex.config.models import ArtifexSettings
from artifex.onboarding import (
    configure_discovered_renderer,
    discover_controller,
    discover_renderer,
)
from artifex.setup import configure_two_pc


def _portable(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "ComfyUI_windows_portable"
    project = root / "ComfyUI"
    project.mkdir(parents=True)
    (project / "main.py").write_text("# ComfyUI", encoding="utf-8")
    python = root / "python_embeded" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"mock executable")
    models = project / "models"
    checkpoint = models / "checkpoints" / "selected.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    (models / "loras").mkdir()
    return root, python, checkpoint


def test_bounded_discovery_of_portable_comfyui_is_read_only(tmp_path: Path) -> None:
    root, python, checkpoint = _portable(tmp_path)
    first = discover_renderer(root)
    assert first.comfyui_directory == root / "ComfyUI"
    assert first.selected_python == python.resolve()
    assert first.asset_candidates["production_checkpoint"] == (checkpoint.resolve(),)
    assert first.lora_roots == ((root / "ComfyUI" / "models" / "loras").resolve(),)
    assert first.warnings == ()
    assert not (tmp_path / "render-node.yaml").exists()


def test_ambiguous_comfyui_python_never_auto_selected(tmp_path: Path) -> None:
    root, _python, checkpoint = _portable(tmp_path)
    other = root / "ComfyUI" / ".venv" / "Scripts" / "python.exe"
    other.parent.mkdir(parents=True)
    other.touch()
    scan = discover_renderer(root)
    assert scan.selected_python is None
    assert len(scan.python_candidates) == 2
    with pytest.raises(ValueError, match="Cannot safely select"):
        configure_discovered_renderer(
            ArtifexSettings(),
            root=root,
            output_path=tmp_path / "renderer.yaml",
            checkpoint_path=checkpoint,
        )
    assert not (tmp_path / "renderer.yaml").exists()


def test_comfy_root_without_main_py_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "other"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="No main.py"):
        discover_renderer(root)


def test_external_model_path_warning_no_unbounded_scanning(tmp_path: Path) -> None:
    root, _python, _checkpoint = _portable(tmp_path)
    (root / "ComfyUI" / "extra_model_paths.yaml").write_text(
        "external:\n  base_path: E:/models\n", encoding="utf-8"
    )
    report = discover_renderer(root)
    assert any("not auto-scanned" in issue for issue in report.warnings)


def test_onboarding_renders_managed_comfy_and_model_paths(
    tmp_path: Path,
) -> None:
    root, python, checkpoint = _portable(tmp_path)
    other_lora = tmp_path / "other-loras"
    other_lora.mkdir()
    output = tmp_path / "config" / "render-node.yaml"
    result = configure_discovered_renderer(
        ArtifexSettings(),
        root=root,
        output_path=output,
        checkpoint_path=checkpoint,
        node_id="pc-b-rtx3060",
        bind_host="0.0.0.0",
        port=9190,
        comfy_port=8200,
        extra_lora_roots=(other_lora,),
    )
    assert result.comfyui_managed
    assert result.comfyui_executable == python.resolve()
    assert result.bind_host == "0.0.0.0"
    assert result.port == 9190
    saved = load_settings(user_config=output, env={})
    proc = saved.render_agent.comfyui_process
    assert proc.arguments == (
        "main.py", "--listen", "0.0.0.0", "--port", "8200"
    )
    assert saved.render_agent.asset_paths["production_checkpoint"] == checkpoint.resolve()
    assert saved.comfyui.base_url == "http://127.0.0.1:8200"
    assert len(saved.render_agent.lora_roots) == 2
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["render_agent"]["node_id"] == (
        "pc-b-rtx3060"
    )


def test_onboarding_rejects_nonexistent_operator_checkpoint(tmp_path: Path) -> None:
    root, _python, _checkpoint = _portable(tmp_path)
    target = tmp_path / "config.yaml"
    with pytest.raises(FileNotFoundError):
        configure_discovered_renderer(
            ArtifexSettings(),
            root=root,
            output_path=target,
            checkpoint_path=tmp_path / "fake.safetensors",
        )
    assert not target.exists()


def test_onboarding_external_comfy_does_not_take_process_ownership(tmp_path: Path) -> None:
    root, _python, checkpoint = _portable(tmp_path)
    output = tmp_path / "config.yaml"
    result = configure_discovered_renderer(
        ArtifexSettings(),
        root=root,
        output_path=output,
        checkpoint_path=checkpoint,
        external_comfy=True,
    )
    assert not result.comfyui_managed
    config = load_settings(user_config=output, env={})
    assert not config.render_agent.comfyui_process.enabled
    assert config.render_agent.comfyui_process.executable is None


def test_onboarding_update_preserves_other_settings(tmp_path: Path) -> None:
    root, _python, checkpoint = _portable(tmp_path)
    output = tmp_path / "node.yaml"
    configure_discovered_renderer(
        ArtifexSettings(), root=root, output_path=output,
        checkpoint_path=checkpoint,
    )
    saved = yaml.safe_load(output.read_text(encoding="utf-8"))
    saved["discord"] = {"enabled": False}
    output.write_text(yaml.safe_dump(saved), encoding="utf-8")
    updated = configure_discovered_renderer(
        load_settings(user_config=output, env={}),
        root=root, output_path=output,
        checkpoint_path=checkpoint, port=9390, update=True,
    )
    assert updated.port == 9390
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["discord"] == {"enabled": False}
    assert payload["render_agent"]["port"] == 9390


def test_controller_discovers_existing_binary_and_never_exposes_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"binary")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    settings = ArtifexSettings()
    settings.llm.server.executable = str(exe)
    settings.llm.bootstrap.profiles["local-model"] = {
        "source": "local", "path": model,
    }
    # Validate after profile switch so model_path resolves to explicit location.
    from artifex.config.models import LlmModelProfileConfig

    settings.llm.bootstrap.profiles["local-model"] = LlmModelProfileConfig(
        source="local", path=model,
    )
    settings.llm.bootstrap.profile = "local-model"
    settings.render_nodes = settings.render_nodes.model_validate(
        {
            "primary": "b",
            "nodes": {
                "b": {"base_url": "http://192.168.1.50:8188"},
            },
        }
    )
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "private-value")
    info = discover_controller(settings)
    assert info.llama_server_executable == str(exe.resolve())
    assert info.selected_llm_model_exists
    assert info.token_present
    assert "private-value" not in info.model_dump_json()


def test_offline_controller_setup_skips_all_network_requests(tmp_path: Path) -> None:
    def reject(_request: httpx.Request) -> httpx.Response:
        pytest.fail("offline setup must not connect to PC-B")

    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        result = configure_two_pc(
            ArtifexSettings(),
            comfyui_base_url="http://192.168.2.60:8188",
            output_path=tmp_path / "pc-a.yaml",
            production_checkpoint="ilxl.safetensors",
            probe_comfyui=False,
            client=client,
        )
    assert not result.comfyui_probed
    assert result.comfyui_version is None
    assert result.devices == ()
    saved = load_settings(user_config=result.config_path, env={})
    assert saved.render_nodes.primary_node()[1].base_url == "http://192.168.2.60:8188"


def test_offline_setup_cli_creates_yaml_before_renderer_starts(tmp_path: Path) -> None:
    output = tmp_path / "pc-a.yaml"
    result = CliRunner().invoke(
        app,
        [
            "setup", "--output", str(output),
            "--comfy-url", "http://192.168.2.60:8188",
            "--production-checkpoint", "ilxl.safetensors",
            "--offline", "--skip-llm-download",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"comfyui_probed": false' in result.output
    assert output.is_file()


def test_onboard_renderer_cli_detects_python_without_user_entering_binary(
    tmp_path: Path,
) -> None:
    root, _python, checkpoint = _portable(tmp_path)
    output = tmp_path / "node.yaml"
    inspected = CliRunner().invoke(
        app,
        ["onboard", "inspect", "--role", "renderer", "--comfy-root", str(root)],
    )
    assert inspected.exit_code == 0, inspected.output
    assert '"selected_python"' in inspected.output
    saved = CliRunner().invoke(
        app,
        [
            "onboard", "renderer",
            "--comfy-root", str(root),
            "--checkpoint-path", str(checkpoint),
            "--output", str(output),
        ],
    )
    assert saved.exit_code == 0, saved.output
    assert load_settings(user_config=output, env={}).render_agent.comfyui_process.enabled


def test_onboard_inspect_renderer_requires_explicit_root() -> None:
    result = CliRunner().invoke(
        app, ["onboard", "inspect", "--role", "renderer"]
    )
    assert result.exit_code == 1
    assert "--comfy-root is required" in result.output
