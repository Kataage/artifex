from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.native_dependencies import (
    _torch_backend,
    check_native_dependencies,
)


def _renderer(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "ComfyUI_windows_portable"
    comfy = root / "ComfyUI"
    comfy.mkdir(parents=True)
    (comfy / "main.py").touch()
    (comfy / "requirements.txt").write_text("torch\n", encoding="utf-8")
    python = root / "python_embeded" / "python.exe"
    python.parent.mkdir()
    python.touch()
    return root, python


def _environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    nvidia_ok: bool = True,
    uv_ok: bool = True,
) -> None:
    from artifex import native_dependencies as deps

    monkeypatch.setattr(deps.platform, "system", lambda: "Windows")

    def fake_which(program: str) -> str | None:
        if program == "nvidia-smi":
            return "C:/Windows/nvidia-smi.exe" if nvidia_ok else None
        if program == "uv":
            return "C:/tools/uv.exe" if uv_ok else None
        return None

    monkeypatch.setattr(deps.shutil, "which", fake_which)
    monkeypatch.setattr(
        deps, "_command",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, "RTX 3060, 600.0, 12288\n", ""
        ),
    )


def test_readonly_renderer_dependency_inspection_does_not_import_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _python = _renderer(tmp_path)
    _environment(monkeypatch)
    settings = ArtifexSettings()
    calls: list[str] = []

    def no_torch(python: Path) -> tuple[bool, str]:
        calls.append(str(python))
        return True, ""

    monkeypatch.setattr("artifex.native_dependencies._torch_backend", no_torch)
    report = check_native_dependencies(
        settings, role="renderer", comfy_root=root, probe_torch=False
    )
    assert report.ready
    assert report.platform == "Windows"
    assert not calls
    assert not any(check.name == "torch_cuda" for check in report.checks)


def test_explicit_torch_cuda_probe_checks_real_selected_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, python = _renderer(tmp_path)
    _environment(monkeypatch)
    reported: list[Path] = []

    def fake_torch(executable: Path) -> tuple[bool, str]:
        reported.append(executable)
        return True, "PyTorch=2.7.0; built CUDA=12.8; CUDA available=True"

    monkeypatch.setattr("artifex.native_dependencies._torch_backend", fake_torch)
    report = check_native_dependencies(
        ArtifexSettings(), role="renderer",
        comfy_root=root, probe_torch=True,
    )
    assert report.ready
    assert reported == [python.resolve()]
    assert any(check.name == "torch_cuda" and check.ready for check in report.checks)


def test_cuda_probe_failure_is_reported_and_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _python = _renderer(tmp_path)
    _environment(monkeypatch)
    monkeypatch.setattr(
        "artifex.native_dependencies._torch_backend",
        lambda _: (False, "torch installed without CUDA"),
    )
    report = check_native_dependencies(
        ArtifexSettings(), role="renderer",
        comfy_root=root, probe_torch=True,
    )
    assert not report.ready
    assert any(
        check.name == "torch_cuda" and check.blocking and not check.ready
        for check in report.checks
    )


def test_missing_nvidia_driver_fails_renderer_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _python = _renderer(tmp_path)
    _environment(monkeypatch, nvidia_ok=False)
    report = check_native_dependencies(
        ArtifexSettings(), role="renderer", comfy_root=root
    )
    assert not report.ready
    assert next(c for c in report.checks if c.name == "nvidia_driver").blocking


def test_controller_managed_llama_binary_is_blocking_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _environment(monkeypatch)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF fake")
    from artifex.config.models import LlmModelProfileConfig

    settings = ArtifexSettings()
    settings.llm.bootstrap.profiles["local"] = LlmModelProfileConfig(
        source="local", path=model
    )
    settings.llm.bootstrap.profile = "local"
    settings.llm.server.enabled = False
    report = check_native_dependencies(settings, role="controller")
    binary = next(c for c in report.checks if c.name == "llama_server")
    assert not binary.ready and not binary.blocking
    settings.llm.server.enabled = True
    report = check_native_dependencies(settings, role="controller")
    assert not report.ready
    assert next(c for c in report.checks if c.name == "llama_server").blocking


def test_torch_backend_probes_json_without_pip_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from artifex import native_dependencies as deps

    captured: list[list[str]] = []
    python = tmp_path / "python.exe"
    python.touch()

    def fake_command(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append(args)
        return subprocess.CompletedProcess(
            args, 0,
            json.dumps(
                {
                    "torch": "2.7.0+cu128",
                    "torch_cuda": "12.8",
                    "cuda_available": True,
                    "devices": ["RTX 3060"],
                }
            ),
            "",
        )

    monkeypatch.setattr(deps, "_command", fake_command)
    okay, detail = _torch_backend(python)
    assert okay
    assert "CUDA available=True" in detail
    assert captured[0][0] == str(python)
    assert captured[0][1:3] == ["-I", "-c"]
    assert "pip install" not in captured[0][3]


def test_torch_probe_bad_json_and_import_failure_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from artifex import native_dependencies as deps
    path = tmp_path / "python.exe"
    path.touch()
    monkeypatch.setattr(
        deps, "_command",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "invalid", ""),
    )
    valid, detail = _torch_backend(path)
    assert not valid and "invalid probe JSON" in detail
    monkeypatch.setattr(
        deps, "_command",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, json.dumps({"error": "ModuleNotFoundError: torch"}), ""
        ),
    )
    valid, detail = _torch_backend(path)
    assert not valid and "ModuleNotFoundError" in detail


def test_cli_dependency_diagnostics_emits_structured_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _python = _renderer(tmp_path)
    _environment(monkeypatch)
    result = CliRunner().invoke(
        app, [
            "onboard", "dependencies",
            "--role", "renderer",
            "--comfy-root", str(root),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"role": "renderer"' in result.output
    assert '"nvidia_driver"' in result.output


def test_cli_dependency_requires_explicit_renderer_root() -> None:
    result = CliRunner().invoke(
        app, ["onboard", "dependencies", "--role", "renderer"]
    )
    assert result.exit_code == 1
    assert "require --comfy-root" in result.output
