from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.render_node import fetch_render_attestation
from artifex.render_node.models import RenderNodeAttestation


class PreflightCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ready: bool
    detail: str


class ControllerPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    renderer_id: str | None
    checks: tuple[PreflightCheck, ...]


def _asset_name(value: str) -> str:
    # Renderer reports Windows-native paths even when tests run on Linux.
    return value.replace("\\", "/").rstrip("/").rsplit("/", maxsplit=1)[-1]


def _renderer_checks(
    settings: ArtifexSettings,
    node_id: str,
    evidence: RenderNodeAttestation,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    age = (datetime.now(UTC) - evidence.created_at).total_seconds()
    fresh = -300 <= age <= 300
    checks.append(
        PreflightCheck(
            name="render_attestation_freshness",
            ready=fresh,
            detail=(
                f"Renderer attestation age: {age:.0f}s"
                if fresh
                else f"Renderer attestation stale or clock-skewed ({age:.0f}s)"
            ),
        )
    )
    windows = evidence.os.get("system") == "Windows"
    checks.append(
        PreflightCheck(
            name="render_os",
            ready=windows,
            detail=f"Render node {node_id} OS: {evidence.os.get('system', 'unknown')}",
        )
    )
    checks.append(
        PreflightCheck(
            name="render_gpu",
            ready=bool(evidence.nvidia_gpus),
            detail=f"Renderer NVIDIA GPUs: {len(evidence.nvidia_gpus)}",
        )
    )
    errors = evidence.inventory_errors
    checks.append(
        PreflightCheck(
            name="render_inventory",
            ready=not errors,
            detail=(
                f"{len(evidence.loras)} LoRAs; complete inventory"
                if not errors
                else (
                    f"{len(evidence.loras)} LoRAs; {len(errors)} inventory errors: "
                    + "; ".join(
                        f"{item.get('path', '?')}: {item.get('error', 'unknown')}"
                        for item in errors[:3]
                    )
                )
            ),
        )
    )

    expected_assets = {
        "production_checkpoint": settings.production.checkpoint,
        "refiner_checkpoint": settings.comfyui.refiner_checkpoint,
        "vae": settings.comfyui.vae,
        "upscale_model": settings.comfyui.upscale_model,
    }
    available = {item.label: item for item in evidence.assets}
    for label, configured in expected_assets.items():
        remote = available.get(label)
        ready = bool(configured and remote and _asset_name(remote.path).casefold() ==
                     _asset_name(configured).casefold())
        if not configured:
            detail = f"{label} is not configured on PC-A"
        elif remote is None:
            detail = f"{label} is missing from authenticated PC-B inventory"
        elif not ready:
            detail = (
                f"{label} filename mismatch: controller={_asset_name(configured)}, "
                f"renderer={_asset_name(remote.path)}"
            )
        else:
            detail = f"{label}: {_asset_name(configured)} (remote SHA-256 recorded)"
        checks.append(
            PreflightCheck(name=f"asset_{label}", ready=ready, detail=detail)
        )
    return checks


def check_controller(
    settings: ArtifexSettings,
    *,
    client: httpx.Client | None = None,
    attestation: RenderNodeAttestation | None = None,
) -> ControllerPreflight:
    """Read-only two-PC network and model agreement test from PC-A.

    This is not the production qualification ladder; it never submits jobs,
    downloads models, mutates user config or bypasses authenticated attestation.
    """
    checks: list[PreflightCheck] = []
    primary = settings.render_nodes.primary_node()
    if primary is None:
        checks.append(
            PreflightCheck(
                name="renderer_configuration",
                ready=False,
                detail="No primary render node configured; run artifex setup first",
            )
        )
        return ControllerPreflight(ready=False, renderer_id=None, checks=tuple(checks))

    node_id, node = primary
    configured = bool(node.attestation_url)
    checks.append(
        PreflightCheck(
            name="renderer_configuration",
            ready=configured and node.output_mode == "api",
            detail=(
                f"primary={node_id}; API image transport; attestation configured"
                if configured and node.output_mode == "api"
                else "Primary renderer requires authenticated attestation and output_mode=api"
            ),
        )
    )

    token_env = node.attestation_token_env
    token_present = bool(token_env and os.environ.get(token_env))
    checks.append(
        PreflightCheck(
            name="render_token",
            ready=token_present,
            detail=(
                f"Authentication token is set via {token_env}"
                if token_present
                else f"Missing authentication token environment variable {token_env or '(unset)'}"
            ),
        )
    )

    owns_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(10.0, connect=5.0),
        follow_redirects=False,
    )
    try:
        try:
            response = http.get(node.base_url.rstrip("/") + "/system_stats")
            response.raise_for_status()
            payload: Any = response.json()
            if not isinstance(payload, dict):
                raise TypeError("ComfyUI system_stats must be an object")
            system = payload.get("system", {})
            if not isinstance(system, dict):
                raise TypeError("ComfyUI system_stats.system must be an object")
            checks.append(
                PreflightCheck(
                    name="comfyui_lan",
                    ready=True,
                    detail=(
                        f"PC-B ComfyUI reachable at {node.base_url}; "
                        f"version={system.get('comfyui_version', 'unknown')}"
                    ),
                )
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            checks.append(
                PreflightCheck(
                    name="comfyui_lan",
                    ready=False,
                    detail=f"PC-A cannot reach ComfyUI {node.base_url}: {exc}",
                )
            )

        remote: RenderNodeAttestation | None = attestation
        if configured and token_present and remote is None:
            try:
                remote = fetch_render_attestation(
                    node_id, node, client=http, timeout_seconds=120.0
                )
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                checks.append(
                    PreflightCheck(
                        name="render_attestation",
                        ready=False,
                        detail=(
                            f"Authenticated PC-B render-node unavailable: {exc}; "
                            "verify LAN address, firewall, service and shared token"
                        ),
                    )
                )
        if remote is not None:
            matched = remote.node_id == node_id
            checks.append(
                PreflightCheck(
                    name="render_attestation",
                    ready=matched and configured and token_present,
                    detail=(
                        f"Authenticated renderer identity: {remote.node_id}"
                        if matched
                        else f"Renderer identity mismatch: expected {node_id}, got {remote.node_id}"
                    ),
                )
            )
            if matched and configured and token_present:
                checks.extend(_renderer_checks(settings, node_id, remote))

        try:
            response = http.get(settings.llm.base_url.rstrip("/") + "/health")
            if response.status_code == 404:
                response = http.get(settings.llm.base_url.rstrip("/") + "/v1/models")
            response.raise_for_status()
            checks.append(
                PreflightCheck(
                    name="llm_lan",
                    ready=True,
                    detail=f"Configured LLM endpoint responded at {settings.llm.base_url}",
                )
            )
        except (httpx.HTTPError, ValueError) as exc:
            checks.append(
                PreflightCheck(
                    name="llm_lan",
                    ready=False,
                    detail=f"LLM server unavailable at {settings.llm.base_url}: {exc}",
                )
            )
    finally:
        if owns_client:
            http.close()

    if settings.llm.backend == "llama_cpp" and settings.llm.bootstrap.enabled:
        model_path = settings.llm.bootstrap.model_path().expanduser()
        exists = model_path.is_file()
        checks.append(
            PreflightCheck(
                name="llm_model_file",
                ready=exists,
                detail=(
                    f"Local GGUF found: {model_path}"
                    if exists
                    else (
                        f"Local GGUF missing: {model_path}; "
                        "run artifex llm bootstrap or update the selected profile"
                    )
                ),
            )
        )
    return ControllerPreflight(
        ready=all(check.ready for check in checks),
        renderer_id=node_id,
        checks=tuple(checks),
    )
