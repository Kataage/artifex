from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.first_run import compile_first_run_guide


def _portable(tmp_path: Path, *, ambiguous: bool = False) -> Path:
    root = tmp_path / "ComfyUI_windows_portable"
    comfy = root / "ComfyUI"
    comfy.mkdir(parents=True)
    (comfy / "main.py").write_text("# controlled fake\n", encoding="utf-8")
    embedded = root / "python_embeded" / "python.exe"
    embedded.parent.mkdir()
    embedded.write_bytes(b"stand-in Python")
    ckpt = comfy / "models" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "selected.safetensors").write_bytes(b"fake")
    if ambiguous:
        (ckpt / "different.safetensors").write_bytes(b"fake")
    (comfy / "models" / "loras").mkdir()
    return root


def test_first_run_pc_a_missing_yaml_does_not_start_or_download_anything(
    tmp_path: Path,
) -> None:
    target = tmp_path / "config" / "local.yaml"
    result = compile_first_run_guide(
        role="controller", config=target, system_name="Windows",
    )
    assert result.status == "needs_input"
    assert not result.configuration_exists
    assert result.needed_inputs == (
        "PC-B authenticated attestation LAN URL (do not guess IP/port)",
        "Shared token environment variable on PC-A and PC-B",
    )
    assert result.next_safe_preview_argv == (
        "uv", "run", "artifex", "onboard", "inspect",
        "--role", "controller", "--json",
    )
    assert result.optional_config_write_argv is None
    assert result.controller_discovery is not None
    assert result.authenticated_pair_preview is None
    assert result.checked_live_pc_b is False
    assert not target.exists()
    assert not (tmp_path / "data").exists()
    assert not result.gpu_jobs_submitted
    assert not result.service_started
    assert not result.production_qualified


def test_first_run_pc_b_recognizes_unambiguous_local_comfy_without_any_write(
    tmp_path: Path,
) -> None:
    comfy_root = _portable(tmp_path)
    config = tmp_path / "config" / "render-node.yaml"
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    result = compile_first_run_guide(
        role="renderer", config=config, comfy_root=comfy_root,
        system_name="Windows",
    )
    assert result.status == "preview_ready"
    assert result.renderer_plan is not None
    assert result.renderer_plan.ready_to_write
    assert result.renderer_plan.checkpoint_selection == "unique_local"
    assert result.next_safe_preview_argv is not None
    assert "--apply" not in result.next_safe_preview_argv
    assert result.optional_config_write_argv is not None
    assert "--apply" in result.optional_config_write_argv
    assert "--update" not in result.optional_config_write_argv
    assert not config.exists()
    assert before == sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert not result.service_started


def test_first_run_renderer_refuses_ambiguous_checkpoint_or_missing_root(
    tmp_path: Path,
) -> None:
    root = _portable(tmp_path, ambiguous=True)
    target = tmp_path / "render-node.yaml"
    ambiguous = compile_first_run_guide(
        role="renderer", config=target, comfy_root=root, system_name="Windows",
    )
    assert ambiguous.status == "needs_review"
    assert ambiguous.renderer_plan is not None
    assert ambiguous.renderer_plan.checkpoint_selection == "unresolved"
    assert ambiguous.optional_config_write_argv is None
    assert ambiguous.blockers
    no_root = compile_first_run_guide(
        role="renderer", config=target, system_name="Windows",
    )
    assert no_root.status == "needs_input"
    assert no_root.renderer_plan is None
    assert no_root.optional_config_write_argv is None
    assert "ComfyUI" in no_root.needed_inputs[0]
    assert not target.exists()


def test_first_run_pc_b_existing_yaml_does_not_adopt_or_restart_comfy(
    tmp_path: Path,
) -> None:
    target = tmp_path / "render-node.yaml"
    target.write_text("{}\n", encoding="utf-8")
    report = compile_first_run_guide(
        role="renderer", config=target, system_name="Windows",
    )
    assert report.configuration_exists
    assert report.status == "configured"
    assert report.next_safe_preview_argv == (
        "uv", "run", "artifex", "onboard", "renderer-safety",
        "--config", str(target), "--json",
    )
    assert report.optional_config_write_argv is None
    assert target.read_text(encoding="utf-8") == "{}\n"
    assert not report.checked_live_pc_b


def test_first_run_pc_a_existing_yaml_shows_read_only_overview_only(
    tmp_path: Path,
) -> None:
    target = tmp_path / "local.yaml"
    target.write_text("{}\n", encoding="utf-8")
    report = compile_first_run_guide(
        role="controller", config=target, system_name="Windows",
    )
    assert report.configuration_exists
    assert report.status == "configured"
    assert report.next_safe_preview_argv == (
        "uv", "run", "artifex", "qualify", "overview",
        "--config", str(target), "--json",
    )
    assert report.optional_config_write_argv is None
    assert not report.checked_live_pc_b


def _attestation() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "node_id": "gpu-b",
        "created_at": datetime.now(UTC).isoformat(),
        "hostname": "renderer-pc-b",
        "os": {"system": "Windows"},
        "nvidia_gpus": [{"name": "RTX 3060"}],
        "comfyui_base_url": "http://127.0.0.1:8188",
        "assets": [{
            "label": "production_checkpoint",
            "path": "D:\\AI\\ComfyUI\\models\\checkpoints\\model.safetensors",
            "sha256": "a" * 64,
            "bytes": 420,
        }],
        "loras": [],
        "inventory_errors": [],
    }


def test_authenticated_pc_a_first_run_queries_only_protected_get_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    token = "a-test-bearer-value-with-at-least-24-characters"
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", token)
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer " + token
        requests.append(str(request.url))
        if request.url.path == "/v1/attestation":
            assert request.url.params["fresh"] == "1"
            return httpx.Response(200, json=_attestation())
        if request.url.path == "/system_stats":
            assert request.url.host == "192.168.50.8"
            assert request.url.port == 8191
            return httpx.Response(
                200, json={"system": {"comfyui_version": "0.9.0"}},
            )
        pytest.fail("Unexpected endpoint: first-run may not queue GPU work")

    target = tmp_path / "local.yaml"
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        report = compile_first_run_guide(
            role="controller", config=target,
            attestation_url="http://192.168.50.8:8190",
            client=http, system_name="Windows",
        )
    assert report.authenticated_pair_preview is not None
    assert report.authenticated_pair_preview.authenticated
    assert report.checked_live_pc_b
    assert report.status == "preview_ready"
    assert len(requests) == 2
    assert not any("/prompt" in x for x in requests)
    assert report.optional_config_write_argv is not None
    assert "--apply" in report.optional_config_write_argv
    assert "--update" not in report.optional_config_write_argv
    assert "--apply" not in (report.next_safe_preview_argv or ())
    assert token not in report.model_dump_json()
    assert not target.exists()
    assert not report.service_started
    assert not report.config_written


def test_existing_pc_a_config_update_is_only_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ARTIFEX_RENDER_NODE_TOKEN", "a-test-bearer-value-with-at-least-24-characters",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/attestation":
            return httpx.Response(200, json=_attestation())
        return httpx.Response(200, json={"system": {"comfyui_version": "0.9.0"}})

    config = tmp_path / "local.yaml"
    config.write_text("{}\n", encoding="utf-8")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        report = compile_first_run_guide(
            role="controller", config=config,
            attestation_url="http://192.168.50.8:8190",
            client=http, system_name="Windows",
        )
    assert report.optional_config_write_argv is not None
    assert "--update" in report.optional_config_write_argv
    assert config.read_text(encoding="utf-8") == "{}\n"


def test_first_run_rejects_symlink_inputs_and_cross_role_arguments(
    tmp_path: Path,
) -> None:
    target = tmp_path / "real.yaml"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.yaml"
    try:
        alias.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        compile_first_run_guide(role="controller", config=alias)
    with pytest.raises(ValueError, match="only valid"):
        compile_first_run_guide(
            role="controller", config=target, comfy_root=tmp_path,
        )
    with pytest.raises(ValueError, match="only valid"):
        compile_first_run_guide(
            role="renderer", config=target,
            attestation_url="http://192.168.50.8:8190",
        )


def test_cli_first_run_creates_only_exclusive_report_and_never_writes_config(
    tmp_path: Path,
) -> None:
    config = tmp_path / "absent" / "local.yaml"
    report = tmp_path / "reports" / "first-run.json"
    cli = CliRunner()
    args = [
        "onboard", "first-run", "--role", "controller",
        "--config", str(config), "--report-path", str(report), "--json",
    ]
    first = cli.invoke(app, args)
    assert first.exit_code == 0, first.output
    parsed = json.loads(first.stdout)
    assert parsed["status"] == "needs_input"
    assert parsed["config_written"] is False
    assert parsed["gpu_jobs_submitted"] is False
    assert parsed["subprocess_commands_executed"] is False
    assert json.loads(report.read_text(encoding="utf-8")) == parsed
    assert not config.exists()
    second = cli.invoke(app, args)
    assert second.exit_code == 1
    assert "FileExistsError" in second.output
    assert json.loads(report.read_text(encoding="utf-8")) == parsed
