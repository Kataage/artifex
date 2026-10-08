from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.configuration_edit import write_override


class SetupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_path: Path
    render_node_id: str
    comfyui_base_url: str
    comfyui_version: str | None
    devices: tuple[str, ...]
    comfyui_probed: bool
    attestation_url: str
    llm_profile: str
    llm_model_path: Path


def _normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("ComfyUI URL must be an http(s) URL with a host")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("ComfyUI URL must not contain a path, query, or fragment")
    return value


def _default_attestation_url(comfyui_base_url: str) -> str:
    parsed = urlsplit(comfyui_base_url)
    assert parsed.hostname is not None
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}:8190"


def _probe_comfyui(
    base_url: str,
    *,
    client: httpx.Client,
) -> tuple[str | None, tuple[str, ...]]:
    response = client.get(f"{base_url}/system_stats")
    response.raise_for_status()
    payload = response.json()
    system = payload.get("system", {})
    version = system.get("comfyui_version") if isinstance(system, dict) else None
    devices_payload = payload.get("devices", ())
    devices: list[str] = []
    if isinstance(devices_payload, list):
        for item in devices_payload:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    devices.append(name.strip())
    return (
        str(version) if version is not None else None,
        tuple(devices),
    )


def configure_two_pc(
    settings: ArtifexSettings,
    *,
    comfyui_base_url: str,
    output_path: Path,
    render_node_id: str = "renderer",
    attestation_url: str | None = None,
    production_checkpoint: str | None = None,
    semantic_model_path: Path | None = None,
    llm_base_url: str | None = None,
    llama_server_executable: str | None = None,
    llama_server_device: str | None = None,
    llama_server_gpu_layers: str | None = None,
    attestation_token_env: str = "ARTIFEX_RENDER_NODE_TOKEN",
    update: bool = False,
    force: bool = False,
    client: httpx.Client | None = None,
    probe_comfyui: bool = True,
) -> SetupResult:
    node_id = render_node_id.strip()
    if not node_id:
        raise ValueError("render node id must not be empty")

    base_url = _normalize_base_url(comfyui_base_url)
    attestation = _normalize_base_url(
        attestation_url or _default_attestation_url(base_url)
    )

    version: str | None = None
    devices: tuple[str, ...] = ()
    if probe_comfyui:
        owns_client = client is None
        http = client or httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        try:
            version, devices = _probe_comfyui(base_url, client=http)
        finally:
            if owns_client:
                http.close()

    target = output_path.expanduser().resolve(strict=False)
    llm_path = settings.llm.bootstrap.model_path().expanduser().resolve(strict=False)
    payload: dict[str, Any] = {
        "llm": {
            "bootstrap": {
                "enabled": settings.llm.bootstrap.enabled,
                "auto_download": settings.llm.bootstrap.auto_download,
                "profile": settings.llm.bootstrap.profile,
                "models_dir": str(
                    settings.llm.bootstrap.models_dir.expanduser().resolve(strict=False)
                ),
            }
        },
        "render_nodes": {
            "primary": node_id,
            "nodes": {
                node_id: {
                    "type": "comfyui",
                    "enabled": True,
                    "base_url": base_url,
                    "output_mode": "api",
                    "download_dir": str(
                        settings.comfyui.download_dir.expanduser().resolve(strict=False)
                    ),
                    "attestation_url": attestation,
                    "attestation_token_env": attestation_token_env,
                }
            },
        },
        "qualification": {
            "asset_paths": {
                "llm_model": str(llm_path),
            }
        },
    }

    if production_checkpoint is not None:
        if not production_checkpoint.strip():
            raise ValueError("production checkpoint must not be empty")
        payload["production"] = {"checkpoint": production_checkpoint.strip()}
    if semantic_model_path is not None:
        payload["qualification"]["asset_paths"]["semantic_model"] = str(
            semantic_model_path.expanduser().resolve(strict=False)
        )
    if llm_base_url is not None:
        payload["llm"]["base_url"] = _normalize_base_url(llm_base_url)
    server: dict[str, Any] = {}
    if llama_server_executable is not None:
        if not llama_server_executable.strip():
            raise ValueError("llama-server executable must not be empty")
        server.update(enabled=True, executable=llama_server_executable.strip())
    if llama_server_device is not None:
        if not llama_server_device.strip():
            raise ValueError("llama-server device must not be empty")
        server["device"] = llama_server_device.strip()
    if llama_server_gpu_layers is not None:
        if llama_server_gpu_layers in {"auto", "all"}:
            server["gpu_layers"] = llama_server_gpu_layers
        else:
            try:
                layers = int(llama_server_gpu_layers)
            except ValueError as exc:
                raise ValueError("llama-server gpu-layers must be auto, all or an integer") from exc
            if layers < 0:
                raise ValueError("llama-server gpu-layers must be nonnegative")
            server["gpu_layers"] = layers
    if server:
        payload["llm"]["server"] = server

    write_override(target, payload, update=update, force=force)

    return SetupResult(
        config_path=target,
        render_node_id=node_id,
        comfyui_base_url=base_url,
        comfyui_version=version,
        devices=devices,
        comfyui_probed=probe_comfyui,
        attestation_url=attestation,
        llm_profile=settings.llm.bootstrap.profile,
        llm_model_path=llm_path,
    )
