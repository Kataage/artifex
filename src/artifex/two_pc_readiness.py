from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.deployment import DeploymentReport, verify_deployment
from artifex.native_dependencies import (
    NativeDependencyReport,
    check_native_dependencies,
)
from artifex.render_node import (
    RenderNodeAttestation,
    RenderNodePreflight,
    build_attestation,
    check_render_node,
    fetch_render_attestation,
)

_REQUIRED_LABELS = frozenset(
    {"production_checkpoint", "refiner_checkpoint", "vae", "upscale_model"}
)


class RendererEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    created_at: datetime
    hostname: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    gpu_count: int = Field(ge=0)
    asset_sha256: dict[str, str]
    asset_bytes: dict[str, int]
    lora_count: int = Field(ge=0)
    torch_probed: bool
    native: NativeDependencyReport
    preflight: RenderNodePreflight


class PairCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ready: bool
    detail: str
    next_action: str | None = None


class PairReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    preflight_ready: bool
    actual_render_verified: bool = False
    checks: tuple[PairCheck, ...]
    snapshot_at: datetime
    checked_at: datetime
    controller_native: NativeDependencyReport
    controller_deployment: DeploymentReport


def collect_renderer_evidence(
    settings: ArtifexSettings,
    *,
    comfy_root: Path,
    probe_torch: bool = True,
    native_check: Callable[..., NativeDependencyReport] = check_native_dependencies,
    attestation_builder: Callable[[ArtifexSettings], RenderNodeAttestation] = (
        build_attestation
    ),
    renderer_check: Callable[..., RenderNodePreflight] = check_render_node,
) -> RendererEvidence:
    """Collect local PC-B evidence without changing packages, settings or services.

    Inventory hashing follows the existing renderer attestation path and can
    read large model files. No model is loaded and no GPU image is queued.
    """
    native = native_check(
        settings, role="renderer", comfy_root=comfy_root, probe_torch=probe_torch
    )
    evidence = attestation_builder(settings)
    preflight = renderer_check(settings, attestation=evidence)
    return RendererEvidence(
        created_at=evidence.created_at,
        hostname=evidence.hostname,
        node_id=evidence.node_id,
        gpu_count=len(evidence.nvidia_gpus),
        asset_sha256={item.label: item.sha256 for item in evidence.assets},
        asset_bytes={item.label: item.bytes for item in evidence.assets},
        lora_count=len(evidence.loras),
        torch_probed=probe_torch,
        native=native,
        preflight=preflight,
    )


def read_renderer_evidence(path: Path) -> RendererEvidence:
    """Small strict external report: do not accept arbitrary oversized JSON."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Renderer evidence must be a regular, non-symlinked JSON file")
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Renderer evidence exceeds the 1 MiB size limit")
    return RendererEvidence.model_validate_json(path.read_bytes())


def _check(
    name: str, ready: bool, detail: str, action: str | None = None
) -> PairCheck:
    return PairCheck(
        name=name, ready=ready, detail=detail,
        next_action=None if ready else action,
    )


async def verify_pair_readiness(
    settings: ArtifexSettings,
    renderer: RendererEvidence,
    *,
    max_age_minutes: int = 60,
    local_hostname: str,
    native_check: Callable[..., NativeDependencyReport] = check_native_dependencies,
    deployment_check: Callable[..., Awaitable[DeploymentReport]] = verify_deployment,
    attestation_fetch: Callable[
        [str, RenderNodeConfig], RenderNodeAttestation
    ] | None = None,
) -> PairReadiness:
    """PC-A checks local dependencies, live network/services, then pairs PC-B.

    The imported JSON is not trusted proof. Independently refresh authenticated
    PC-B attestation and compare GPU, host, node identity and actual model bytes.
    No packages downloaded, no jobs queued, no qualification stages marked.
    """
    if not 1 <= max_age_minutes <= 1440:
        raise ValueError("max_age_minutes must be between 1 and 1440")
    current = datetime.now(UTC)
    checks: list[PairCheck] = []
    when = renderer.created_at
    age = (current - when).total_seconds() if when.tzinfo else float("inf")
    fresh = -300 <= age <= max_age_minutes * 60
    checks.append(_check(
        "renderer_evidence_freshness", fresh,
        f"PC-B report age={age:.0f}s; allowed maximum={max_age_minutes}min",
        "Collect a new PC-B report and verify both system clocks.",
    ))
    separate = local_hostname.casefold() != renderer.hostname.casefold()
    checks.append(_check(
        "separate_hosts", separate,
        f"PC-A={local_hostname}; PC-B={renderer.hostname}",
        "Use a PC-B report from the other physical computer.",
    ))
    checks.append(_check(
        "renderer_local_native", renderer.native.ready,
        "PC-B native Windows/uv/GPU/Python prerequisites",
        "On PC-B run dependency diagnostics with --probe-torch and repair failures.",
    ))
    checks.append(_check(
        "renderer_torch_probe",
        renderer.torch_probed
        and any(
            item.name == "torch_cuda" and item.ready and item.blocking
            for item in renderer.native.checks
        ),
        "PC-B ComfyUI Python Torch CUDA availability was explicitly checked",
        "Run deployment pair-export on PC-B with --probe-torch.",
    ))
    checks.append(_check(
        "renderer_preflight", renderer.preflight.ready,
        "; ".join(renderer.preflight.issues)
        or "PC-B ComfyUI, driver, LoRA inventory and local token checked",
        "Inspect PC-B renderer preflight for missing model paths, token or LAN bind.",
    ))
    controller_native = await asyncio.to_thread(native_check, settings, role="controller")
    checks.extend(
        _check(
            f"controller_native:{item.name}", item.ready,
            item.detail,
            "Run 'artifex onboard dependencies --role controller' on PC-A.",
        )
        for item in controller_native.checks if item.blocking
    )
    deployment = await deployment_check(
        settings, role="controller", render_smoke=False
    )
    checks.extend(
        _check(
            f"controller_deployment:{item.name}", item.ready,
            item.detail,
            "Run 'artifex deployment verify --role controller' on PC-A and fix this check.",
        )
        for item in deployment.checks if item.blocking
    )
    primary = settings.render_nodes.primary_node()
    checks.append(_check(
        "primary_renderer_configured", primary is not None,
        f"PC-A primary node: {primary[0] if primary else '(missing)'}",
        "Configure an enabled PC-B primary render node on PC-A.",
    ))
    remote: RenderNodeAttestation | None = None
    if primary is not None:
        node_id, node = primary
        checks.append(_check(
            "paired_renderer_id", node_id == renderer.node_id
            and renderer.preflight.node_id == renderer.node_id,
            f"PC-A={node_id}; PC-B={renderer.node_id}; "
            f"preflight={renderer.preflight.node_id}",
            "Use matching render node identifiers in both PC configs.",
        ))
        try:
            if attestation_fetch is not None:
                remote = await asyncio.to_thread(attestation_fetch, node_id, node)
            else:
                remote = await asyncio.to_thread(
                    fetch_render_attestation, node_id, node, fresh=True,
                    timeout_seconds=20,
                )
        except Exception as exc:  # noqa: BLE001 - report network failures
            checks.append(_check(
                "live_authenticated_renderer", False,
                f"PC-A could not refresh authenticated PC-B attestation: {exc}",
                "Check PC-B attestation service, firewall, LAN route and bearer token.",
            ))
    if remote is not None:
        remote_age = (current - remote.created_at).total_seconds()
        checks.append(_check(
            "live_authenticated_renderer",
            remote.node_id == renderer.node_id
            and remote.hostname.casefold() == renderer.hostname.casefold()
            and remote.os.get("system") == "Windows"
            and -300 <= remote_age <= 300,
            f"Remote node={remote.node_id}; host={remote.hostname}; "
            f"OS={remote.os.get('system')}; age={remote_age:.0f}s",
            "Refresh the renderer service/evidence; verify clock and node identity.",
        ))
        checks.append(_check(
            "gpu_count_agreement",
            renderer.gpu_count > 0 and len(remote.nvidia_gpus) == renderer.gpu_count,
            f"PC-B reported {renderer.gpu_count} GPU(s); remote has "
            f"{len(remote.nvidia_gpus)}",
            "Inspect NVIDIA driver and select the correct renderer PC.",
        ))
        actual = {item.label: item for item in remote.assets}
        for label in sorted(_REQUIRED_LABELS):
            digest = renderer.asset_sha256.get(label)
            remote_asset = actual.get(label)
            same = bool(
                digest and remote_asset and digest == remote_asset.sha256
                and renderer.asset_bytes.get(label) == remote_asset.bytes
            )
            checks.append(_check(
                f"asset_agreement:{label}", same,
                f"PC-B snapshot and authenticated fresh attestation "
                f"{'match' if same else 'differ/missing'} for {label}",
                "Verify PC-B model paths and recapture evidence after model changes.",
            ))
        checks.append(_check(
            "lora_count_agreement", renderer.lora_count == len(remote.loras),
            f"PC-B reported {renderer.lora_count}; remote has {len(remote.loras)} LoRA(s)",
            "Refresh renderer LoRA inventory and recapture PC-B evidence.",
        ))
        checks.append(_check(
            "remote_asset_inventory_clean", not remote.inventory_errors,
            f"Authenticated PC-B inventory errors={len(remote.inventory_errors)}",
            "Fix invalid or unreadable model and LoRA paths on PC-B.",
        ))
    return PairReadiness(
        preflight_ready=all(item.ready for item in checks),
        checks=tuple(checks),
        snapshot_at=renderer.created_at,
        checked_at=current,
        controller_native=controller_native,
        controller_deployment=deployment,
    )
