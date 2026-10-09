from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config import load_settings
from artifex.onboarding_renderer_auto import (
    apply_renderer_auto,
    plan_renderer_auto,
)


def _portable(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "ComfyUI_windows_portable"
    project = root / "ComfyUI"
    project.mkdir(parents=True)
    (project / "main.py").write_text("# existing ComfyUI\n", encoding="utf-8")
    python = root / "python_embeded" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"placeholder")
    checkpoint = project / "models" / "checkpoints" / "only.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    (project / "models" / "loras").mkdir()
    return root, python, checkpoint


def test_unique_discovery_is_preview_only_no_process_ownership(tmp_path: Path) -> None:
    root, exe, checkpoint = _portable(tmp_path)
    plan = plan_renderer_auto(root, node_id="pc-b", bind_host="0.0.0.0")
    assert plan.ready_to_write
    assert plan.checkpoint_selection == "unique_local"
    assert plan.checkpoint_path == checkpoint.resolve()
    assert plan.python_executable == exe.resolve()
    assert plan.protected_gateway_planned
    assert plan.managed_comfy_loopback_only
    assert plan.managed_comfy_launch_planned
    assert plan.process_state_verified is False
    assert plan.task_registered is False
    assert plan.process_started is False
    assert plan.gpu_jobs_submitted is False
    assert plan.production_qualified is False
    assert plan.config_written is False
    assert len(plan.lora_roots) == 1
    assert not (tmp_path / "render-node.yaml").exists()


def test_explicit_apply_prepares_locally_bound_managed_comfy_and_gateway(
    tmp_path: Path,
) -> None:
    root, _, checkpoint = _portable(tmp_path)
    plan = plan_renderer_auto(root)
    path = tmp_path / "config" / "render-node.yaml"
    saved, result = apply_renderer_auto(plan, path)
    assert result.config_path == path
    assert saved.config_written
    assert saved.config_path == path
    assert saved.process_started is False
    settings = load_settings(user_config=path, env={})
    assert settings.render_agent.comfyui_process.enabled
    assert settings.render_agent.gateway.enabled
    assert settings.render_agent.gateway.bind_host == "0.0.0.0"
    assert settings.render_agent.gateway.port == 8191
    assert settings.render_agent.comfyui_process.arguments == (
        "main.py", "--listen", "127.0.0.1", "--port", "8188",
    )
    assert settings.render_agent.comfyui_process.executable is not None
    assert settings.render_agent.asset_paths["production_checkpoint"] == checkpoint.resolve()
    assert settings.render_agent.require_token is True
    assert settings.comfyui.base_url == "http://127.0.0.1:8188"
    with pytest.raises(FileExistsError, match="--update"):
        apply_renderer_auto(plan, path)


def test_update_preserves_unrelated_settings_and_keeps_protection(tmp_path: Path) -> None:
    root, _, _ = _portable(tmp_path)
    plan = plan_renderer_auto(root)
    path = tmp_path / "renderer.yaml"
    apply_renderer_auto(plan, path)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["discord"] = {"enabled": False}
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    updated, _ = apply_renderer_auto(
        plan, path, update=True,
        settings=load_settings(user_config=path, env={}),
    )
    assert updated.config_written
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["discord"]["enabled"] is False
    assert saved["render_agent"]["gateway"]["enabled"] is True


def test_external_comfy_mode_never_adopts_existing_gpu_process(tmp_path: Path) -> None:
    root, _, _ = _portable(tmp_path)
    plan = plan_renderer_auto(root, external_comfy=True)
    assert plan.ready_to_write
    assert not plan.managed_comfy_launch_planned
    assert not plan.protected_gateway_planned
    assert plan.python_executable is None
    target = tmp_path / "external.yaml"
    _, result = apply_renderer_auto(plan, target)
    assert not result.comfyui_managed
    settings = load_settings(user_config=target, env={})
    assert not settings.render_agent.comfyui_process.enabled
    assert not settings.render_agent.gateway.enabled


def test_switch_to_external_mode_disables_existing_protected_gateway(
    tmp_path: Path,
) -> None:
    root, _, _ = _portable(tmp_path)
    target = tmp_path / "renderer.yaml"
    apply_renderer_auto(plan_renderer_auto(root), target)
    previous = load_settings(user_config=target, env={})
    external = plan_renderer_auto(root, external_comfy=True)
    apply_renderer_auto(external, target, update=True, settings=previous)
    changed = load_settings(user_config=target, env={})
    assert not changed.render_agent.comfyui_process.enabled
    assert not changed.render_agent.gateway.enabled


def test_multiple_checkpoints_require_explicit_operator_selection(
    tmp_path: Path,
) -> None:
    root, _, checkpoint = _portable(tmp_path)
    second = checkpoint.parent / "second.ckpt"
    second.write_bytes(b"checkpoint2")
    plan = plan_renderer_auto(root)
    assert plan.checkpoint_selection == "unresolved"
    assert not plan.ready_to_write
    assert plan.checkpoint_path is None
    with pytest.raises(ValueError, match="Unresolved"):
        apply_renderer_auto(plan, tmp_path / "unsafe.yaml")
    explicit = plan_renderer_auto(root, checkpoint_path=second)
    assert explicit.ready_to_write
    assert explicit.checkpoint_selection == "explicit"
    assert explicit.checkpoint_path == second.resolve()
    assert not (tmp_path / "unsafe.yaml").exists()


def test_extra_model_paths_cannot_silently_approve_unique_local_checkpoint(
    tmp_path: Path,
) -> None:
    root, _, checkpoint = _portable(tmp_path)
    (root / "ComfyUI" / "extra_model_paths.yaml").write_text(
        "base_path: E:/other_models\n", encoding="utf-8",
    )
    plan = plan_renderer_auto(root)
    assert not plan.ready_to_write
    assert any("extra_model_paths" in b for b in plan.blockers)
    assert plan_renderer_auto(root, checkpoint_path=checkpoint).ready_to_write


def test_ambiguous_python_blocks_managed_not_external(tmp_path: Path) -> None:
    root, _, _ = _portable(tmp_path)
    other = root / "ComfyUI" / ".venv" / "Scripts" / "python.exe"
    other.parent.mkdir(parents=True)
    other.touch()
    managed = plan_renderer_auto(root)
    assert not managed.ready_to_write
    assert any("Python" in b for b in managed.blockers)
    assert plan_renderer_auto(root, python_executable=other).ready_to_write
    assert plan_renderer_auto(root, external_comfy=True).ready_to_write


@pytest.mark.parametrize(("port", "gateway", "comfy"), [
    (8190, 8190, 8188),
    (8190, 8191, 8190),
    (8190, 8191, 8191),
    (0, 8191, 8188),
    (8190, 65536, 8188),
])
def test_invalid_or_conflicting_ports_rejected(
    tmp_path: Path, port: int, gateway: int, comfy: int,
) -> None:
    root, _, _ = _portable(tmp_path)
    with pytest.raises(ValueError, match="port"):
        plan_renderer_auto(
            root, attestation_port=port, gateway_port=gateway, comfy_port=comfy,
        )


def test_changed_comfy_asset_identity_blocks_stale_apply(tmp_path: Path) -> None:
    root, _, checkpoint = _portable(tmp_path)
    plan = plan_renderer_auto(root)
    checkpoint.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        apply_renderer_auto(plan, tmp_path / "renderer.yaml")
    assert not (tmp_path / "renderer.yaml").exists()


def test_symlinked_output_path_is_not_written(tmp_path: Path) -> None:
    root, _, _ = _portable(tmp_path)
    plan = plan_renderer_auto(root)
    outside = tmp_path / "outside.yaml"
    outside.write_text("{}")
    linked = tmp_path / "config.yaml"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Cannot create symlink")
    with pytest.raises(ValueError, match="symlink"):
        apply_renderer_auto(plan, linked)
    assert outside.read_text() == "{}"


def test_cli_preview_apply_and_ambiguous_exit(tmp_path: Path) -> None:
    root, _, checkpoint = _portable(tmp_path)
    output = tmp_path / "created.yaml"
    args = [
        "onboard", "renderer-auto", "--comfy-root", str(root),
        "--output", str(output), "--json",
    ]
    preview = CliRunner().invoke(app, args)
    assert preview.exit_code == 0, preview.output
    assert json.loads(preview.output)["config_written"] is False
    assert not output.exists()
    apply = CliRunner().invoke(app, args + ["--apply"])
    assert apply.exit_code == 0, apply.output
    assert json.loads(apply.output)["config_written"] is True
    assert output.is_file()
    again = CliRunner().invoke(app, args + ["--apply"])
    assert again.exit_code == 1
    checkpoint.parent.joinpath("second.safetensors").write_bytes(b"second")
    ambiguous = CliRunner().invoke(app, args)
    assert ambiguous.exit_code == 1
    assert json.loads(ambiguous.output)["ready_to_write"] is False


def test_cli_update_requires_apply_without_filesystem_change(tmp_path: Path) -> None:
    root, _, _ = _portable(tmp_path)
    target = tmp_path / "updated.yaml"
    result = CliRunner().invoke(app, [
        "onboard", "renderer-auto", "--comfy-root", str(root),
        "--output", str(target), "--update",
    ])
    assert result.exit_code == 1
    assert "--update requires --apply" in result.output
    assert not target.exists()


def test_explicit_checkpoint_requires_existing_file(tmp_path: Path) -> None:
    root, _, _ = _portable(tmp_path)
    with pytest.raises(FileNotFoundError):
        plan_renderer_auto(root, checkpoint_path=tmp_path / "unknown.safetensors")
