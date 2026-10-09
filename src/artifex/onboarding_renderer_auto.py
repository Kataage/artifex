"""Preview-first PC-B renderer configuration from bounded local ComfyUI discovery.

The output only prepares YAML. There is NO task install, GPU job, process
adoption, model download or authority to start/restart a live ComfyUI.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.onboarding import configure_discovered_renderer, discover_renderer
from artifex.setup_renderer import RendererSetupResult

_CHECKPOINT_SUFFIXES = frozenset({".safetensors", ".ckpt"})


class RendererAutoPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    comfy_root: Path
    comfy_directory: Path
    python_executable: Path | None
    checkpoint_path: Path | None
    checkpoint_selection: Literal["explicit", "unique_local", "unresolved"]
    checkpoint_candidates: tuple[Path, ...]
    lora_roots: tuple[Path, ...]
    external_comfy: bool
    node_id: str
    attestation_bind_host: str
    attestation_port: int
    gateway_port: int
    comfy_port: int
    protected_gateway_planned: bool
    managed_comfy_launch_planned: bool
    managed_comfy_loopback_only: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    ready_to_write: bool
    config_written: bool = False
    config_path: Path | None = None
    process_state_verified: Literal[False] = False
    task_registered: Literal[False] = False
    process_started: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    production_qualified: Literal[False] = False


def plan_renderer_auto(
    comfy_root: Path,
    *,
    checkpoint_path: Path | None = None,
    python_executable: Path | None = None,
    external_comfy: bool = False,
    node_id: str = "renderer",
    bind_host: str = "0.0.0.0",
    attestation_port: int = 8190,
    gateway_port: int = 8191,
    comfy_port: int = 8188,
) -> RendererAutoPlan:
    """Select a checkpoint only if exactly one local candidate is unambiguous."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", node_id):
        raise ValueError("Renderer node ID must be an unambiguous short identifier")
    if not all(1 <= port <= 65535 for port in
               (attestation_port, gateway_port, comfy_port)):
        raise ValueError("All ports must be between 1 and 65535")
    if len({attestation_port, gateway_port, comfy_port}) != 3:
        raise ValueError("ComfyUI, attestation, and gateway ports must be different")
    if not bind_host.strip():
        raise ValueError("Attestation bind host is required")
    if python_executable is not None and external_comfy:
        raise ValueError("An external ComfyUI cannot also select managed Python")
    discovery = discover_renderer(comfy_root)
    blockers: list[str] = []
    notes = list(discovery.warnings)
    candidates = tuple(
        p for p in discovery.asset_candidates.get("production_checkpoint", ())
        if p.suffix.casefold() in _CHECKPOINT_SUFFIXES
    )
    extra_paths = (discovery.comfyui_directory / "extra_model_paths.yaml").is_file()
    selection: Literal["explicit", "unique_local", "unresolved"] = "unresolved"
    selected: Path | None = None
    if checkpoint_path is not None:
        selected = checkpoint_path.expanduser().resolve(strict=True)
        if not selected.is_file() or selected.suffix.casefold() not in _CHECKPOINT_SUFFIXES:
            raise ValueError("Explicit checkpoint must be an existing .safetensors or .ckpt")
        selection = "explicit"
    elif not extra_paths and len(candidates) == 1:
        selected = candidates[0]
        selection = "unique_local"
        notes.append(
            "Single local checkpoint is a proposal; --apply explicitly accepts this choice."
        )
    elif extra_paths:
        blockers.append(
            "extra_model_paths.yaml may supply other checkpoints; select --checkpoint-path explicitly"
        )
    else:
        blockers.append(
            "Zero or multiple local checkpoints; select --checkpoint-path explicitly"
        )
    chosen_python: Path | None = None
    if not external_comfy:
        chosen_python = (
            python_executable.expanduser().resolve(strict=True)
            if python_executable is not None else discovery.selected_python
        )
        if chosen_python is None:
            blockers.append("Ambiguous or absent Python; specify --comfy-exe")
        elif not chosen_python.is_file():
            blockers.append("Configured ComfyUI Python executable is not a file")
    else:
        notes.append(
            "External ComfyUI mode cannot qualify protected ownership or gateway."
        )
    notes.append(
        "Existing ComfyUI/process occupancy is NOT verified; no process is started or adopted."
    )
    return RendererAutoPlan(
        comfy_root=comfy_root.expanduser().resolve(strict=True),
        comfy_directory=discovery.comfyui_directory,
        python_executable=chosen_python, checkpoint_path=selected,
        checkpoint_selection=selection, checkpoint_candidates=candidates,
        lora_roots=discovery.lora_roots, external_comfy=external_comfy,
        node_id=node_id, attestation_bind_host=bind_host,
        attestation_port=attestation_port, gateway_port=gateway_port,
        comfy_port=comfy_port, protected_gateway_planned=not external_comfy,
        managed_comfy_launch_planned=not external_comfy,
        managed_comfy_loopback_only=not external_comfy,
        blockers=tuple(blockers), warnings=tuple(dict.fromkeys(notes)),
        ready_to_write=not blockers,
    )


def apply_renderer_auto(
    plan: RendererAutoPlan,
    output: Path,
    *,
    update: bool = False,
    settings: ArtifexSettings | None = None,
) -> tuple[RendererAutoPlan, RendererSetupResult]:
    """Apply an explicit preview, rechecking source identities against a race."""
    if not plan.ready_to_write or plan.checkpoint_path is None:
        raise ValueError("Unresolved checkpoint/Python selection; config not written")
    target = output.expanduser().absolute()
    if any(part.is_symlink() for part in (target, *target.parents)):
        raise ValueError("Refusing a symlink in renderer output path")
    if not update and target.exists():
        raise FileExistsError("Config exists; explicit --update required")
    if update and not target.is_file():
        raise FileNotFoundError("Cannot update missing PC-B config")
    fresh = plan_renderer_auto(
        plan.comfy_root, checkpoint_path=plan.checkpoint_path,
        python_executable=plan.python_executable,
        external_comfy=plan.external_comfy, node_id=plan.node_id,
        bind_host=plan.attestation_bind_host,
        attestation_port=plan.attestation_port,
        gateway_port=plan.gateway_port, comfy_port=plan.comfy_port,
    )
    if (
        not fresh.ready_to_write
        or fresh.comfy_directory != plan.comfy_directory
        or fresh.checkpoint_path != plan.checkpoint_path
        or fresh.python_executable != plan.python_executable
        or fresh.lora_roots != plan.lora_roots
    ):
        raise ValueError("ComfyUI discovery changed after preview; no config written")
    result = configure_discovered_renderer(
        settings or ArtifexSettings(), root=plan.comfy_root,
        output_path=target, checkpoint_path=plan.checkpoint_path,
        node_id=plan.node_id, bind_host=plan.attestation_bind_host,
        port=plan.attestation_port, comfy_port=plan.comfy_port,
        external_comfy=plan.external_comfy,
        python_executable=plan.python_executable,
        comfy_bind_host="127.0.0.1",
        protected_gateway=not plan.external_comfy,
        gateway_port=plan.gateway_port,
        update=update, force=False,
    )
    return (
        plan.model_copy(update={"config_written": True, "config_path": result.config_path}),
        result,
    )
