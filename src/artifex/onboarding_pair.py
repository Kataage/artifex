"""Provision a PC-A connection from authenticated live PC-B facts.

All discovery is read-only. Writes are an explicit, no-clobber --apply step.
Never launch or adopt a ComfyUI process, bypass its gateway, or infer ownership
from a port match alone.
"""
from __future__ import annotations

import ipaddress
import os
import re
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings
from artifex.render_node.models import RenderNodeAttestation
from artifex.setup import configure_two_pc

_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_ASSET = re.compile(r"^[^\\/:*?\"<>|\x00-\x1f]{1,255}\.(?:safetensors|ckpt)$", re.IGNORECASE)
_MAX_JSON = 2 * 1024 * 1024


class PairConfigurationPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    attestation_url: str
    renderer_gateway_url: str
    hostname: str
    attestation_age_seconds: float = Field(ge=-30, le=300)
    checkpoint_filename: str
    checkpoint_sha256: str
    gpu_count: int
    gateway_reachable: bool
    authenticated: bool
    configuration_written: bool = False
    output_path: str | None = None
    services_mutated: bool = False
    gpu_jobs_submitted: bool = False
    production_qualified: bool = False


def _validated_lan_url(value: str) -> tuple[str, str, str]:
    from artifex.setup import _normalize_base_url

    normalized = _normalize_base_url(value)
    parsed = urlsplit(normalized)
    if (
        parsed.username is not None or parsed.password is not None
        or parsed.port is None or parsed.hostname is None
        or parsed.path not in ("", "/")
    ):
        raise ValueError("PC-B URL must contain a host and port with no credentials or path")
    host = parsed.hostname
    if host.lower() in {"localhost", "0.0.0.0", "::", "host.docker.internal"}:
        raise ValueError("PC-B URL must use an actual LAN host, not localhost or wildcard")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host) or "." not in host:
            raise ValueError("Use an explicit PC-B LAN IP or resolvable FQDN") from None
    else:
        if literal.is_loopback or literal.is_unspecified or literal.is_multicast:
            raise ValueError("PC-B URL must use the actual remote host")
    host_for_url = f"[{host}]" if ":" in host else host
    return normalized, parsed.scheme, host_for_url


def _checkpoint_from_attestation(attestation: RenderNodeAttestation) -> tuple[str, str]:
    options = [asset for asset in attestation.assets if asset.label == "production_checkpoint"]
    if len(options) != 1:
        raise ValueError("PC-B must attest exactly one configured production checkpoint")
    candidate = options[0]
    name = PureWindowsPath(candidate.path.replace("/", "\\")).name
    if not _ASSET.fullmatch(name) or name in (".", ".."):
        raise ValueError("PC-B production checkpoint filename is unsafe or unsupported")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", candidate.sha256):
        raise ValueError("PC-B checkpoint requires a SHA-256 digest")
    return name, candidate.sha256.lower()


def discover_controller_pair(
    attestation_url: str,
    *,
    gateway_port: int = 8191,
    token_env: str = "ARTIFEX_RENDER_NODE_TOKEN",
    now: datetime | None = None,
    client: httpx.Client | None = None,
) -> PairConfigurationPreview:
    """GET authenticated fresh attestation and protected gateway health only."""
    url, scheme, host = _validated_lan_url(attestation_url)
    if not 1 <= gateway_port <= 65535:
        raise ValueError("Gateway port must be 1..65535")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", token_env):
        raise ValueError("Invalid token environment variable name")
    token = os.environ.get(token_env, "")
    if len(token) < 24:
        raise ValueError("An existing shared PC-B token of >=24 characters is required")
    gateway_url = f"{scheme}://{host}:{gateway_port}"
    owns_client = client is None
    http = client or httpx.Client(
        follow_redirects=False, trust_env=False,
        timeout=httpx.Timeout(12.0, connect=4.0),
    )
    headers = {"Authorization": "Bearer " + token}
    try:
        response = http.get(
            url + "/v1/attestation", params={"fresh": "1"},
            headers=headers, follow_redirects=False,
        )
        response.raise_for_status()
        if len(response.content) > _MAX_JSON:
            raise ValueError("PC-B attestation exceeds bounded response size")
        attest = RenderNodeAttestation.model_validate(response.json())
        if not _ID.fullmatch(attest.node_id):
            raise ValueError("Invalid PC-B node identifier")
        if attest.created_at.tzinfo is None:
            raise ValueError("PC-B attestation lacks timezone")
        current = now or datetime.now(UTC)
        age = (current - attest.created_at).total_seconds()
        if age < -30 or age > 300:
            raise ValueError("PC-B attestation is stale or timestamp is in the future")
        checkpoint, sha = _checkpoint_from_attestation(attest)
        # Never use the upstream URL advertised by PC-B: it is generally
        # 127.0.0.1:8188 and bypasses protected Artifex gateway admission.
        response = http.get(
            gateway_url + "/system_stats", headers=headers,
            follow_redirects=False,
        )
        response.raise_for_status()
        if len(response.content) > _MAX_JSON:
            raise ValueError("PC-B gateway health response is oversized")
        stats = response.json()
        if not isinstance(stats, dict) or not isinstance(stats.get("system"), dict):
            raise TypeError("PC-B protected gateway did not return ComfyUI system_stats")
        if not stats["system"].get("comfyui_version"):
            raise ValueError("PC-B protected gateway did not identify running ComfyUI")
        return PairConfigurationPreview(
            node_id=attest.node_id,
            attestation_url=url,
            renderer_gateway_url=gateway_url,
            hostname=attest.hostname,
            attestation_age_seconds=age,
            checkpoint_filename=checkpoint,
            checkpoint_sha256=sha,
            gpu_count=len(attest.nvidia_gpus),
            gateway_reachable=True,
            authenticated=True,
        )
    finally:
        if owns_client:
            http.close()


def apply_controller_pair(
    preview: PairConfigurationPreview,
    settings: ArtifexSettings,
    output: Path,
    *,
    update: bool = False,
    token_env: str = "ARTIFEX_RENDER_NODE_TOKEN",
) -> PairConfigurationPreview:
    """Write validated PC-A override only after preview; no model downloads."""
    if not (preview.authenticated and preview.gateway_reachable):
        raise ValueError("Live authenticated PC-B gateway proof required")
    target = output.expanduser().absolute()
    if any(part.is_symlink() for part in (target, *target.parents)):
        raise ValueError("Symlinked controller configuration path refused")
    if not update and target.exists():
        raise FileExistsError("Existing PC-A configuration requires --update")
    if update and not target.is_file():
        raise FileNotFoundError("Cannot update missing controller configuration")
    # The write helper validates the effective YAML, recursively merges only
    # explicitly provided keys for --update and never restarts either service.
    result = configure_two_pc(
        settings,
        comfyui_base_url=preview.renderer_gateway_url,
        output_path=target,
        render_node_id=preview.node_id,
        attestation_url=preview.attestation_url,
        production_checkpoint=preview.checkpoint_filename,
        attestation_token_env=token_env,
        update=update,
        probe_comfyui=False,
    )
    return preview.model_copy(update={
        "configuration_written": True, "output_path": str(result.config_path),
    })
