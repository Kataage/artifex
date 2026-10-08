from __future__ import annotations

from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict

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
    payload: dict[str, object] = {
        "render_agent": {
            "node_id": selected_node,
            "bind_host": selected_host,
            "port": selected_port,
            "require_token": True,
            "token_env": selected_token_env,
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
    return RendererSetupResult(
        config_path=target,
        node_id=selected_node,
        bind_host=selected_host,
        port=selected_port,
        comfyui_base_url=comfy_url,
        asset_paths=changed_assets,
        lora_roots=tuple(str(root) for root in roots),
        token_env=selected_token_env,
    )
