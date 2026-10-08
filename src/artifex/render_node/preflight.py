from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.render_node.attestation import build_attestation
from artifex.render_node.models import RenderNodeAttestation

REQUIRED_RENDER_ASSETS = frozenset(
    {"production_checkpoint", "refiner_checkpoint", "vae", "upscale_model"}
)


class RenderNodePreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    node_id: str
    comfyui_base_url: str
    comfyui_version: str | None
    gpu_count: int
    asset_labels: tuple[str, ...]
    lora_count: int
    issues: tuple[str, ...]


def check_render_node(
    settings: ArtifexSettings,
    *,
    client: httpx.Client | None = None,
    attestation: RenderNodeAttestation | None = None,
) -> RenderNodePreflight:
    """Read-only renderer checks before starting two-host qualification.

    This checks renderer-local state only. PC-A must separately verify that
    the ComfyUI and attestation ports are reachable over the trusted LAN.
    """
    config = settings.render_agent
    evidence = attestation if attestation is not None else build_attestation(settings)
    issues: list[str] = []
    labels = tuple(sorted(item.label for item in evidence.assets))
    missing = sorted(REQUIRED_RENDER_ASSETS.difference(labels))

    if evidence.os.get("system") != "Windows":
        issues.append("native Windows is required for target-machine qualification")
    if not evidence.nvidia_gpus:
        issues.append("NVIDIA GPU was not detected by nvidia-smi")
    if missing:
        issues.append("missing renderer asset_paths: " + ", ".join(missing))
    for error in evidence.inventory_errors:
        issues.append(
            "asset/LoRA inventory: "
            + error.get("path", "?")
            + ": "
            + error.get("error", "unknown failure")
        )
    if not config.lora_roots:
        issues.append("render_agent.lora_roots is not configured")
    if not config.require_token:
        issues.append("render_agent.require_token must be true")
    token_name = config.token_env
    if not token_name or not os.environ.get(token_name):
        issues.append("render-node bearer token environment variable is not set")
    if config.bind_host.casefold() in {"localhost", "127.0.0.1", "::1"}:
        issues.append(
            "render_agent.bind_host is loopback-only; PC-A cannot reach it"
        )

    version: str | None = None
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))
    try:
        try:
            response = http.get(settings.comfyui.base_url.rstrip("/") + "/system_stats")
            response.raise_for_status()
            payload: Any = response.json()
            if not isinstance(payload, dict):
                raise TypeError("ComfyUI returned a non-object response")
            system = payload.get("system")
            if isinstance(system, dict):
                raw_version = system.get("comfyui_version")
                if raw_version is not None:
                    version = str(raw_version)
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            issues.append(f"local ComfyUI /system_stats is unavailable: {exc}")
    finally:
        if owns_client:
            http.close()

    return RenderNodePreflight(
        ready=not issues,
        node_id=evidence.node_id,
        comfyui_base_url=settings.comfyui.base_url,
        comfyui_version=version,
        gpu_count=len(evidence.nvidia_gpus),
        asset_labels=labels,
        lora_count=len(evidence.loras),
        issues=tuple(issues),
    )
