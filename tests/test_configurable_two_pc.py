from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config import load_settings
from artifex.config.models import ArtifexSettings
from artifex.setup import configure_two_pc
from artifex.setup_renderer import configure_renderer


def test_renderer_configuration_supports_custom_paths_and_safe_updates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pc-b.yaml"
    assets = {
        "production_checkpoint": tmp_path / "models" / "base.safetensors",
        "vae": tmp_path / "models" / "my_vae.safetensors",
    }
    first = configure_renderer(
        ArtifexSettings(),
        output_path=output,
        node_id="render-b",
        bind_host="192.168.0.20",
        port=9200,
        comfyui_base_url="http://127.0.0.1:8199",
        asset_paths=assets,
        lora_roots=(tmp_path / "loras", tmp_path / "extra-loras"),
    )
    assert first.bind_host == "192.168.0.20"
    original = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert original["render_agent"]["port"] == 9200
    assert len(original["render_agent"]["lora_roots"]) == 2

    saved = load_settings(user_config=output, env={})
    second = configure_renderer(
        saved,
        output_path=output,
        bind_host="192.168.0.30",
        asset_paths={"vae": tmp_path / "models" / "new-vae.safetensors"},
        update=True,
    )
    merged = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert second.bind_host == "192.168.0.30"
    assert merged["render_agent"]["port"] == 9200
    assert merged["render_agent"]["asset_paths"]["production_checkpoint"] == str(
        assets["production_checkpoint"].resolve()
    )
    assert merged["render_agent"]["asset_paths"]["vae"] == str(
        (tmp_path / "models" / "new-vae.safetensors").resolve()
    )
    assert len(merged["render_agent"]["lora_roots"]) == 2
    assert load_settings(user_config=output, env={}).render_agent.port == 9200


def test_renderer_requires_explicit_update_to_change_existing_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "render-node.yaml"
    configure_renderer(ArtifexSettings(), output_path=output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="--update"):
        configure_renderer(
            ArtifexSettings(), output_path=output, port=9010
        )
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_renderer(
            ArtifexSettings(), output_path=output,
            update=True, force=True,
        )


def test_controller_ip_and_model_paths_can_be_changed_without_losing_other_settings(
    tmp_path: Path,
) -> None:
    config = tmp_path / "pc-a.yaml"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/system_stats"
        return httpx.Response(200, json={"system": {}, "devices": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        configure_two_pc(
            ArtifexSettings(),
            comfyui_base_url="http://192.168.0.20:8188",
            output_path=config,
            render_node_id="render-b",
            production_checkpoint="base.safetensors",
            client=client,
        )

    # An operator can add unrelated config without losing it on later IP changes.
    existing = yaml.safe_load(config.read_text(encoding="utf-8"))
    existing["discord"] = {"enabled": False}
    existing["qualification"]["asset_paths"]["semantic_model"] = "D:/semantic"
    config.write_text(yaml.safe_dump(existing), encoding="utf-8")

    saved = load_settings(user_config=config, env={})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        configure_two_pc(
            saved,
            comfyui_base_url="http://192.168.0.40:8188",
            output_path=config,
            render_node_id="render-b",
            attestation_url="http://192.168.0.40:9292",
            production_checkpoint="new-base.safetensors",
            update=True,
            client=client,
        )

    merged = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert merged["discord"] == {"enabled": False}
    assert merged["qualification"]["asset_paths"]["semantic_model"] == "D:/semantic"
    assert merged["production"]["checkpoint"] == "new-base.safetensors"
    assert merged["render_nodes"]["nodes"]["render-b"]["base_url"] == (
        "http://192.168.0.40:8188"
    )
    assert merged["render_nodes"]["nodes"]["render-b"]["attestation_url"] == (
        "http://192.168.0.40:9292"
    )
    assert "llm_model" in merged["qualification"]["asset_paths"]


def test_render_secret_token_does_not_break_strict_config_validation() -> None:
    settings = load_settings(
        env={
            "ARTIFEX_RENDER_NODE_TOKEN": "private-key-not-a-model-setting",
            "ARTIFEX_LLM_BASE_URL": "http://127.0.0.1:9797",
        }
    )
    assert settings.llm.base_url == "http://127.0.0.1:9797"
    assert not hasattr(settings, "render_node_token")


def test_missing_update_file_is_not_silently_created(tmp_path: Path) -> None:
    missing = tmp_path / "not-configured.yaml"
    with pytest.raises(FileNotFoundError, match="cannot update"):
        configure_renderer(
            ArtifexSettings(), output_path=missing, update=True
        )
    assert not missing.exists()


def test_renderer_configure_cli_supports_later_port_and_folder_changes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "render-node.yaml"
    runner = CliRunner()
    created = runner.invoke(
        app,
        [
            "render-node", "configure", "--output", str(output),
            "--node-id", "my-renderer",
            "--bind-host", "192.168.1.8",
            "--port", "9190",
            "--lora-dir", str(tmp_path / "lora-a"),
        ],
    )
    assert created.exit_code == 0, created.output
    updated = runner.invoke(
        app,
        [
            "render-node", "configure", "--output", str(output),
            "--port", "9191",
            "--lora-dir", str(tmp_path / "lora-b"),
            "--update",
        ],
    )
    assert updated.exit_code == 0, updated.output
    saved = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert saved["render_agent"]["port"] == 9191
    assert saved["render_agent"]["node_id"] == "my-renderer"
    assert saved["render_agent"]["bind_host"] == "192.168.1.8"
    assert saved["render_agent"]["lora_roots"] == [
        str((tmp_path / "lora-b").resolve())
    ]


def test_controller_setup_cli_updates_without_reentering_saved_ip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "artifex.setup._probe_comfyui",
        lambda _url, *, client: ("0.9.0", ("RTX 3060",)),
    )
    output = tmp_path / "local.yaml"
    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "setup", "--output", str(output),
            "--comfy-url", "http://192.168.1.20:8188",
            "--render-node-id", "render-b",
            "--skip-llm-download",
        ],
    )
    assert first.exit_code == 0, first.output
    changed = runner.invoke(
        app,
        [
            "setup", "--output", str(output),
            "--production-checkpoint", "my-local-model.safetensors",
            "--render-cache-dir", str(tmp_path / "new-cache"),
            "--skip-llm-download", "--update",
        ],
    )
    assert changed.exit_code == 0, changed.output
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["render_nodes"]["primary"] == "render-b"
    assert payload["render_nodes"]["nodes"]["render-b"]["base_url"] == (
        "http://192.168.1.20:8188"
    )
    assert payload["production"]["checkpoint"] == "my-local-model.safetensors"
    assert payload["render_nodes"]["nodes"]["render-b"]["download_dir"] == str(
        (tmp_path / "new-cache").resolve()
    )
