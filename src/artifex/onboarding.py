from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.setup_renderer import RendererSetupResult, configure_renderer

_ASSET_FOLDERS = {
    "production_checkpoint": "checkpoints",
    "vae": "vae",
    "upscale_model": "upscale_models",
}
_ASSET_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth"}
_MAX_CANDIDATES = 40


class RendererDiscovery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_root: Path
    comfyui_directory: Path
    main_py_exists: bool
    python_candidates: tuple[Path, ...]
    selected_python: Path | None
    asset_candidates: dict[str, tuple[Path, ...]]
    lora_roots: tuple[Path, ...]
    warnings: tuple[str, ...]


class ControllerDiscovery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    os_name: str
    python_executable: Path
    uv_executable: Path | None
    llama_server_executable: str | None
    selected_llm_model_path: Path
    selected_llm_model_exists: bool
    selected_model_profile: str
    token_env_name: str | None
    token_present: bool
    warnings: tuple[str, ...]


def _candidate_folders(root: Path) -> tuple[Path, ...]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"ComfyUI root must be a directory: {root}")
    return (root, root / "ComfyUI")


def _resolve_comfy_dir(root: Path) -> Path:
    candidates = [
        location for location in _candidate_folders(root)
        if (location / "main.py").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No main.py found at {root} or {root / 'ComfyUI'}; "
            "pass the ComfyUI project or portable root explicitly"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous ComfyUI roots ({', '.join(str(p) for p in candidates)}); "
            "pass the exact project directory instead"
        )
    return candidates[0]


def _local_python_candidates(comfy_dir: Path) -> tuple[Path, ...]:
    parent = comfy_dir.parent
    locations = (
        parent / "python_embeded" / "python.exe",
        parent / "python_embedded" / "python.exe",
        parent / ".venv" / "Scripts" / "python.exe",
        parent / "venv" / "Scripts" / "python.exe",
        comfy_dir / ".venv" / "Scripts" / "python.exe",
        comfy_dir / "venv" / "Scripts" / "python.exe",
        parent / ".venv" / "bin" / "python",
        parent / "venv" / "bin" / "python",
        comfy_dir / ".venv" / "bin" / "python",
        comfy_dir / "venv" / "bin" / "python",
    )
    found: list[Path] = []
    seen: set[Path] = set()
    for item in locations:
        if item.is_file():
            resolved = item.resolve(strict=True)
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
    return tuple(found)


def _assets_in(folder: Path) -> tuple[Path, ...]:
    if not folder.is_dir():
        return ()
    files = [
        path.resolve(strict=False)
        for path in folder.iterdir()
        if path.is_file() and path.suffix.casefold() in _ASSET_SUFFIXES
    ]
    return tuple(sorted(files, key=lambda value: value.name.casefold())[:_MAX_CANDIDATES])


def discover_renderer(root: Path) -> RendererDiscovery:
    """Bounded, read-only discovery; never scans unrelated drives or guesses models."""
    comfy = _resolve_comfy_dir(root)
    python_candidates = _local_python_candidates(comfy)
    models_root = comfy / "models"
    assets = {
        label: _assets_in(models_root / folder)
        for label, folder in _ASSET_FOLDERS.items()
    }
    lora_dir = models_root / "loras"
    roots = (lora_dir.resolve(strict=False),) if lora_dir.is_dir() else ()
    issues: list[str] = []
    if not python_candidates:
        issues.append(
            "No local Python environment found; provide --comfy-exe or use "
            "--external-comfy for an independently launched server"
        )
    elif len(python_candidates) > 1:
        issues.append(
            "Multiple local Python installations found; choose --comfy-exe explicitly"
        )
    if not assets["production_checkpoint"]:
        issues.append(
            "No checkpoint in models/checkpoints; provide a full --checkpoint-path "
            "if using external model paths"
        )
    if not roots:
        issues.append(
            "models/loras directory is absent; additional LoRA roots must be specified manually"
        )
    if (comfy / "extra_model_paths.yaml").is_file():
        issues.append(
            "extra_model_paths.yaml is present; external model paths are not auto-scanned "
            "and must be supplied explicitly"
        )
    selected = python_candidates[0] if len(python_candidates) == 1 else None
    return RendererDiscovery(
        requested_root=root.expanduser().resolve(strict=False),
        comfyui_directory=comfy,
        main_py_exists=True,
        python_candidates=python_candidates,
        selected_python=selected,
        asset_candidates=assets,
        lora_roots=roots,
        warnings=tuple(issues),
    )


def discover_controller(settings: ArtifexSettings) -> ControllerDiscovery:
    """Read local binary/model presence only; never download or expose secret values."""
    executable = settings.llm.server.executable
    candidate = Path(executable).expanduser()
    selected: str | None = None
    if candidate.is_file():
        selected = str(candidate.resolve(strict=True))
    else:
        selected = shutil.which(executable)
    token_env = None
    primary = settings.render_nodes.primary_node()
    if primary is not None:
        token_env = primary[1].attestation_token_env
    model = settings.llm.bootstrap.model_path().expanduser().resolve(strict=False)
    uv = shutil.which("uv")
    warnings: list[str] = []
    if selected is None:
        warnings.append("llama-server binary not detected; configure --llama-server-exe")
    if settings.llm.backend == "llama_cpp" and not model.is_file():
        warnings.append(
            "Selected local GGUF not found; use artifex llm bootstrap after "
            "configuring llm.bootstrap and before unattended operation"
        )
    if token_env and not os.environ.get(token_env):
        warnings.append(
            f"Controller does not have render-node bearer environment variable {token_env}"
        )
    if uv is None:
        warnings.append("uv executable not detected on PATH")
    return ControllerDiscovery(
        os_name=platform.system(),
        python_executable=Path(sys.executable).resolve(strict=False),
        uv_executable=Path(uv) if uv is not None else None,
        llama_server_executable=selected,
        selected_llm_model_path=model,
        selected_llm_model_exists=model.is_file(),
        selected_model_profile=settings.llm.bootstrap.profile,
        token_env_name=token_env,
        token_present=bool(token_env and os.environ.get(token_env)),
        warnings=tuple(warnings),
    )


def configure_discovered_renderer(
    settings: ArtifexSettings,
    *,
    root: Path,
    output_path: Path,
    checkpoint_path: Path,
    node_id: str | None = None,
    bind_host: str | None = None,
    port: int | None = None,
    comfy_port: int = 8188,
    external_comfy: bool = False,
    python_executable: Path | None = None,
    refiner_path: Path | None = None,
    vae_path: Path | None = None,
    upscaler_path: Path | None = None,
    extra_lora_roots: tuple[Path, ...] = (),
    update: bool = False,
    force: bool = False,
) -> RendererSetupResult:
    """Apply only operator-approved checkpoint and explicit ComfyUI directory.

    Does not launch services, download assets, register startup tasks or assume
    which model from the scanned checkpoint list is desired.
    """
    if not 1 <= comfy_port <= 65535:
        raise ValueError("ComfyUI port must be between 1 and 65535")
    result = discover_renderer(root)
    chosen_python: Path | None = None
    if not external_comfy:
        chosen_python = (
            python_executable.expanduser().resolve(strict=True)
            if python_executable is not None else result.selected_python
        )
        if chosen_python is None:
            raise ValueError(
                "Cannot safely select ComfyUI Python executable. "
                "Provide --comfy-exe or --external-comfy"
            )
        if not chosen_python.is_file():
            raise FileNotFoundError(f"ComfyUI Python executable is missing: {chosen_python}")
    elif python_executable is not None:
        raise ValueError("--comfy-exe conflicts with --external-comfy")

    checkpoint = checkpoint_path.expanduser().resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.suffix.casefold() not in {".safetensors", ".ckpt"}:
        raise ValueError("--checkpoint-path must name an existing checkpoint file")
    assets = {"production_checkpoint": checkpoint}
    for label, selected in (
        ("refiner_checkpoint", refiner_path),
        ("vae", vae_path),
        ("upscale_model", upscaler_path),
    ):
        if selected is not None:
            resolved = selected.expanduser().resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f"Selected {label} is not a file: {resolved}")
            assets[label] = resolved

    roots = list(result.lora_roots)
    for value in extra_lora_roots:
        root_path = value.expanduser().resolve(strict=True)
        if not root_path.is_dir():
            raise ValueError(f"LoRA root must be a directory: {root_path}")
        if root_path not in roots:
            roots.append(root_path)

    return configure_renderer(
        settings,
        output_path=output_path,
        node_id=node_id,
        bind_host=bind_host,
        port=port,
        comfyui_base_url=f"http://127.0.0.1:{comfy_port}",
        asset_paths=assets,
        lora_roots=tuple(roots),
        comfy_executable=chosen_python,
        comfy_working_directory=result.comfyui_directory if chosen_python is not None else None,
        comfy_arguments=(
            ("main.py", "--listen", "0.0.0.0", "--port", str(comfy_port))
            if chosen_python is not None else None
        ),
        disable_comfy_management=external_comfy,
        update=update,
        force=force,
    )
