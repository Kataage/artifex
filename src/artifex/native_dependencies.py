from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.onboarding import discover_renderer

DependencyRole = Literal["controller", "renderer"]


class DependencyCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ready: bool
    blocking: bool
    detail: str


class NativeDependencyReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: DependencyRole
    ready: bool
    platform: str
    checks: tuple[DependencyCheck, ...]


_TORCH_CHECK = """
import json
try:
    import torch
    cuda_ok = bool(torch.cuda.is_available())
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_ok else []
    print(json.dumps({
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cuda_available': cuda_ok,
        'devices': names,
    }))
except Exception as exc:
    print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)[:500]}))
"""


def _command(args: list[str], *, timeout: float = 12.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, capture_output=True, text=True, check=False, shell=False,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if platform.system() == "Windows" else 0,
    )


def _nvidia_driver() -> tuple[bool, str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False, "nvidia-smi was not found on PATH"
    try:
        result = _command(
            [exe, "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"nvidia-smi query failed: {exc}"
    if result.returncode:
        return False, f"nvidia-smi reported an error: {result.stderr.strip()[:500]}"
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if not rows:
        return False, "nvidia-smi found no NVIDIA GPU"
    # This only describes driver/GPU presence, not CUDA/PyTorch compatibility.
    return True, "; ".join(rows[:8])


def _torch_backend(executable: Path) -> tuple[bool, str]:
    try:
        result = _command([str(executable), "-I", "-c", _TORCH_CHECK], timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"ComfyUI Python/Torch probe failed: {exc}"
    if result.returncode:
        return False, (
            f"ComfyUI Python exited {result.returncode}: "
            f"{result.stderr.strip()[-600:]}"
        )
    try:
        raw: Any = json.loads(result.stdout.strip())
        if not isinstance(raw, dict):
            raise TypeError("Torch probe is not an object")
    except (TypeError, ValueError) as exc:
        return False, f"ComfyUI Python returned invalid probe JSON: {exc}"
    if raw.get("error"):
        return False, f"Torch import/cuda probe failed: {raw['error']}"
    available = raw.get("cuda_available") is True
    return available, (
        f"PyTorch={raw.get('torch')}; built CUDA={raw.get('torch_cuda')}; "
        f"CUDA available={available}; devices={raw.get('devices')}"
    )


def check_native_dependencies(
    settings: ArtifexSettings,
    *,
    role: DependencyRole,
    comfy_root: Path | None = None,
    probe_torch: bool = False,
) -> NativeDependencyReport:
    """Read-only, bounded checks. Never downloads, pip-installs, or edits settings."""
    checks: list[DependencyCheck] = []
    windows = platform.system() == "Windows"
    checks.append(
        DependencyCheck(
            name="windows",
            ready=windows,
            blocking=True,
            detail=f"native Windows required for target deployment, current={platform.system()}",
        )
    )
    uv = shutil.which("uv")
    checks.append(
        DependencyCheck(
            name="uv", ready=uv is not None, blocking=True,
            detail=f"uv found at {uv}" if uv else "uv missing from PATH",
        )
    )
    gpu_ok, gpu_detail = _nvidia_driver()
    checks.append(
        DependencyCheck(
            name="nvidia_driver", ready=gpu_ok,
            blocking=role == "renderer",
            detail=gpu_detail,
        )
    )
    if role == "controller":
        checks.append(
            DependencyCheck(
                name="python", ready=sys.version_info >= (3, 12),
                blocking=True,
                detail=f"Artifex Python {sys.version.split()[0]} at {sys.executable}",
            )
        )
        if settings.llm.backend == "llama_cpp":
            binary_name = settings.llm.server.executable
            resolved = Path(binary_name).expanduser()
            path = (
                str(resolved.resolve(strict=True))
                if resolved.is_file() else shutil.which(binary_name)
            )
            checks.append(
                DependencyCheck(
                    name="llama_server", ready=path is not None,
                    blocking=settings.llm.server.enabled,
                    detail=(
                        f"llama-server binary present: {path}"
                        if path else
                        "llama-server missing; use onboard llama-assets/llama-install "
                        "or configure an existing binary"
                    ),
                )
            )
            model = settings.llm.bootstrap.model_path().expanduser()
            checks.append(
                DependencyCheck(
                    name="gguf_model", ready=model.is_file(),
                    blocking=settings.llm.bootstrap.enabled,
                    detail=(
                        f"Selected GGUF exists: {model}"
                        if model.is_file()
                        else "Selected GGUF missing: run artifex llm bootstrap "
                        "(automatic model bootstrap remains separate)"
                    ),
                )
            )
    else:
        if comfy_root is None:
            raise ValueError("Renderer dependency checks require --comfy-root")
        discovered = discover_renderer(comfy_root)
        checks.append(
            DependencyCheck(
                name="comfy_main_py",
                ready=discovered.main_py_exists,
                blocking=True,
                detail=str(discovered.comfyui_directory / "main.py"),
            )
        )
        chosen = settings.render_agent.comfyui_process.executable
        if chosen is not None:
            executable = chosen.expanduser().resolve(strict=False)
        else:
            executable = discovered.selected_python
        available = executable is not None and executable.is_file()
        checks.append(
            DependencyCheck(
                name="comfy_python",
                ready=available,
                blocking=settings.render_agent.comfyui_process.enabled,
                detail=(
                    f"ComfyUI Python found: {executable}"
                    if available else
                    "ComfyUI Python is missing/ambiguous: set --comfy-exe explicitly, "
                    "or use an externally managed ComfyUI"
                ),
            )
        )
        requirements = discovered.comfyui_directory / "requirements.txt"
        checks.append(
            DependencyCheck(
                name="comfy_requirements",
                ready=requirements.is_file(),
                blocking=False,
                detail=(
                    f"ComfyUI requirements present: {requirements}"
                    if requirements.is_file() else
                    "ComfyUI requirements.txt is absent; portable distributions may "
                    "bundle dependencies instead"
                ),
            )
        )
        if probe_torch:
            if not available or executable is None:
                checks.append(
                    DependencyCheck(
                        name="torch_cuda", ready=False, blocking=True,
                        detail="Cannot probe PyTorch without a valid selected Python",
                    )
                )
            else:
                torch_ok, details = _torch_backend(executable)
                checks.append(
                    DependencyCheck(
                        name="torch_cuda", ready=torch_ok, blocking=True,
                        detail=details,
                    )
                )
    return NativeDependencyReport(
        role=role, ready=all(c.ready for c in checks if c.blocking),
        platform=platform.system(), checks=tuple(checks),
    )
