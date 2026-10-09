from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from artifex.config import load_settings
from artifex.config.models import ArtifexSettings
from artifex.configuration_edit import write_override
from artifex.setup import _normalize_base_url


class RendererSetupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_path: Path
    node_id: str
    bind_host: str
    port: int
    comfyui_base_url: str
    asset_paths: dict[str, str]
    lora_roots: tuple[str, ...]
    token_env: str
    comfyui_managed: bool
    comfyui_executable: Path | None
    comfyui_working_directory: Path | None


def configure_renderer(
    settings: ArtifexSettings,
    *,
    output_path: Path,
    node_id: str | None = None,
    bind_host: str | None = None,
    port: int | None = None,
    comfyui_base_url: str | None = None,
    asset_paths: Mapping[str, Path] | None = None,
    lora_roots: tuple[Path, ...] | None = None,
    token_env: str | None = None,
    comfy_executable: Path | None = None,
    comfy_working_directory: Path | None = None,
    comfy_arguments: tuple[str, ...] | None = None,
    disable_comfy_management: bool = False,
    protected_gateway: bool = False,
    gateway_port: int = 8191,
    update: bool = False,
    force: bool = False,
) -> RendererSetupResult:
    """Create/edit PC-B config without probing or downloading any models."""
    selected_node = (node_id or settings.render_agent.node_id).strip()
    selected_host = (bind_host or settings.render_agent.bind_host).strip()
    selected_port = port if port is not None else settings.render_agent.port
    selected_token_env = token_env or settings.render_agent.token_env
    if not selected_node or not selected_host:
        raise ValueError("render node ID and bind host must be nonempty")
    if not selected_token_env:
        raise ValueError("render-node token environment variable name is required")
    if not 1 <= selected_port <= 65535:
        raise ValueError("render-node port must be between 1 and 65535")
    comfy_url = _normalize_base_url(
        comfyui_base_url or settings.comfyui.base_url
    )
    changed_assets = {
        name: str(path.expanduser().resolve(strict=False))
        for name, path in (asset_paths or {}).items()
    }
    roots = (
        lora_roots
        if lora_roots is not None
        else settings.render_agent.lora_roots
    )
    process_options: dict[str, object] = {}
    if disable_comfy_management and comfy_executable is not None:
        raise ValueError("--disable-comfy-management conflicts with --comfy-exe")
    if protected_gateway and (disable_comfy_management or comfy_executable is None):
        raise ValueError("Protected gateway requires a configured managed ComfyUI executable")
    if protected_gateway and (
        gateway_port in {selected_port, urlsplit(comfy_url).port or 80}
        or not 1 <= gateway_port <= 65535
    ):
        raise ValueError("Gateway, attestation and ComfyUI ports must be distinct")
    if comfy_executable is not None:
        process_options["enabled"] = True
        process_options["executable"] = str(
            comfy_executable.expanduser().resolve(strict=False)
        )
    if comfy_working_directory is not None:
        process_options["working_directory"] = str(
            comfy_working_directory.expanduser().resolve(strict=False)
        )
    if comfy_arguments is not None:
        process_options["arguments"] = list(comfy_arguments)
    if disable_comfy_management:
        process_options["enabled"] = False
    payload: dict[str, object] = {
        "render_agent": {
            "node_id": selected_node,
            "bind_host": selected_host,
            "port": selected_port,
            "require_token": True,
            "token_env": selected_token_env,
            **({"comfyui_process": process_options} if process_options else {}),
            **(
                {"gateway": {"enabled": True, "bind_host": "0.0.0.0",
                             "port": gateway_port}}
                if protected_gateway else {}
            ),
            **({"asset_paths": changed_assets} if changed_assets else {}),
            **(
                {
                    "lora_roots": [
                        str(root.expanduser().resolve(strict=False))
                        for root in roots
                    ]
                }
                if roots or lora_roots is not None
                else {}
            ),
        },
        "comfyui": {"base_url": comfy_url},
    }
    target = write_override(output_path, payload, update=update, force=force)
    configured = load_settings(user_config=target, env={})
    return RendererSetupResult(
        config_path=target,
        node_id=configured.render_agent.node_id,
        bind_host=configured.render_agent.bind_host,
        port=configured.render_agent.port,
        comfyui_base_url=configured.comfyui.base_url,
        asset_paths={
            name: str(path)
            for name, path in configured.render_agent.asset_paths.items()
        },
        lora_roots=tuple(
            str(root) for root in configured.render_agent.lora_roots
        ),
        token_env=configured.render_agent.token_env or selected_token_env,
        comfyui_managed=configured.render_agent.comfyui_process.enabled,
        comfyui_executable=configured.render_agent.comfyui_process.executable,
        comfyui_working_directory=(
            configured.render_agent.comfyui_process.working_directory
        ),
    )
