from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings


class SetupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_path: Path
    render_node_id: str
    comfyui_base_url: str
    comfyui_version: str | None
    devices: tuple[str, ...]
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
    force: bool = False,
    client: httpx.Client | None = None,
) -> SetupResult:
    node_id = render_node_id.strip()
    if not node_id:
        raise ValueError("render node id must not be empty")

    base_url = _normalize_base_url(comfyui_base_url)
    attestation = _normalize_base_url(
        attestation_url or _default_attestation_url(base_url)
    )

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
    payload = {
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
                    "attestation_token_env": "ARTIFEX_RENDER_NODE_TOKEN",
                }
            },
        },
        "qualification": {
            "asset_paths": {
                "llm_model": str(llm_path),
            }
        },
    }

    if target.exists() and not force:
        # Re-running setup after a download/network failure is safe, but never
        # silently replace an operator-modified configuration.
        existing = yaml.safe_load(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"configuration already exists with different settings: {target}; "
                "rerun with --force to replace it"
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(target)

    return SetupResult(
        config_path=target,
        render_node_id=node_id,
        comfyui_base_url=base_url,
        comfyui_version=version,
        devices=devices,
        attestation_url=attestation,
        llm_profile=settings.llm.bootstrap.profile,
        llm_model_path=llm_path,
    )
