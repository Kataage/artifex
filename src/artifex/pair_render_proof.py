from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.deployment import DeploymentReport, RenderSmokeEvidence, verify_deployment
from artifex.render_node import RenderNodeAttestation, fetch_render_attestation
from artifex.two_pc_readiness import (
    PairReadiness,
    RendererEvidence,
    verify_pair_readiness,
)

_REQUIRED_ASSETS = frozenset(
    {"production_checkpoint", "refiner_checkpoint", "vae", "upscale_model"}
)


class PairRenderProof(BaseModel):
    """One explicitly requested image test; never a full production qualification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    started_at: datetime
    finished_at: datetime
    controller_host: str
    renderer_host: str
    renderer_id: str
    preflight_ready: bool
    render_attempted: bool
    actual_render_verified: bool
    asset_stability_verified: bool
    ready_for_qualification: bool
    production_qualified: bool = False
    failures: tuple[str, ...]
    preflight: PairReadiness
    deployment: DeploymentReport | None = None
    output: RenderSmokeEvidence | None = None


def _post_render_integrity(
    snapshot: RendererEvidence, remote: RenderNodeAttestation
) -> tuple[str, ...]:
    failures: list[str] = []
    checked_at = datetime.now(UTC)
    age = (checked_at - remote.created_at).total_seconds()
    if not -300 <= age <= 300:
        failures.append("Post-render attestation was stale or clock-skewed")
    if remote.node_id != snapshot.node_id or (
        remote.hostname.casefold() != snapshot.hostname.casefold()
    ):
        failures.append("PC-B host or node changed during image generation")
    if remote.os.get("system") != "Windows":
        failures.append("Post-render attestation no longer reports Windows")
    if not remote.nvidia_gpus or len(remote.nvidia_gpus) != snapshot.gpu_count:
        failures.append("PC-B GPU inventory changed during image generation")
    actual = {item.label: item for item in remote.assets}
    for label in sorted(_REQUIRED_ASSETS):
        item = actual.get(label)
        if (
            item is None
            or snapshot.asset_sha256.get(label) != item.sha256
            or snapshot.asset_bytes.get(label) != item.bytes
        ):
            failures.append(f"PC-B {label} file digest/size changed during image generation")
    if remote.inventory_errors:
        failures.append("PC-B model/LoRA inventory returned errors after render")
    if snapshot.lora_count != len(remote.loras):
        failures.append("PC-B LoRA inventory count changed during image generation")
    return tuple(failures)


async def run_pair_render_proof(
    settings: ArtifexSettings,
    evidence: RendererEvidence,
    *,
    output_dir: Path | None = None,
    width: int | None = None,
    height: int | None = None,
    max_age_minutes: int = 60,
    controller_host: str | None = None,
    preflight_check: Callable[..., Awaitable[PairReadiness]] = verify_pair_readiness,
    deployment_check: Callable[..., Awaitable[DeploymentReport]] = verify_deployment,
    attestation_fetch: Callable[
        [str, RenderNodeConfig], RenderNodeAttestation
    ] | None = None,
) -> PairRenderProof:
    """Gate ONE real GPU render on full PC-A/B preflight, then check model stability.

    The preflight must pass before any render job can be queued. The existing
    smoke implementation verifies ComfyUI history, downloads and image bytes.
    Always request a fresh remote attestation afterward, including on a
    failed render, and do not call this a production qualification.
    """
    if (width is not None and not 64 <= width <= 8192) or (
        height is not None and not 64 <= height <= 8192
    ):
        raise ValueError("Render dimensions must be between 64 and 8192")
    started = datetime.now(UTC)
    host = controller_host or socket.gethostname()
    preflight = await preflight_check(
        settings, evidence,
        local_hostname=host,
        max_age_minutes=max_age_minutes,
    )
    failures: list[str] = []
    deployment: DeploymentReport | None = None
    image: RenderSmokeEvidence | None = None
    attempted = False
    stable = False
    try:
        primary = settings.render_nodes.primary_node()
    except (KeyError, ValueError):
        # A runtime-edited primary must not bypass the preflight gate or crash
        # before writing the failed evidence file.
        primary = None
    if not preflight.preflight_ready:
        failures.append("Two-PC readiness preflight did not pass; GPU render not started")
    elif primary is None or primary[0] != evidence.node_id:
        failures.append("Primary PC-B renderer identity changed since preflight")
    else:
        attempted = True
        try:
            deployment = await deployment_check(
                settings, role="controller", render_smoke=True,
                output_dir=output_dir, width=width, height=height,
            )
            image = deployment.smoke
            if not deployment.ready or image is None or not any(
                item.name == "render_smoke" and item.ready and item.blocking
                for item in deployment.checks
            ):
                failures.append(
                    "Real ComfyUI image generation, history verification or "
                    "download/image validation did not pass"
                )
                image = None
        except Exception as exc:  # noqa: BLE001 - retain an actionable failed proof
            failures.append(
                f"ComfyUI real image generation raised {type(exc).__name__}: {exc}"
            )
        try:
            node_id, node = primary
            if attestation_fetch is None:
                remote = await asyncio.to_thread(
                    fetch_render_attestation,
                    node_id, node, fresh=True, timeout_seconds=30,
                )
            else:
                remote = await asyncio.to_thread(attestation_fetch, node_id, node)
            after = _post_render_integrity(evidence, remote)
            stable = not after
            failures.extend(after)
        except Exception as exc:  # noqa: BLE001 - a missing post-hash blocks success
            failures.append(
                f"Cannot verify post-render PC-B asset integrity: "
                f"{type(exc).__name__}: {exc}"
            )
    verified = bool(
        attempted and deployment is not None and deployment.ready and image is not None
    )
    return PairRenderProof(
        started_at=started,
        finished_at=datetime.now(UTC),
        controller_host=host,
        renderer_host=evidence.hostname,
        renderer_id=evidence.node_id,
        preflight_ready=preflight.preflight_ready,
        render_attempted=attempted,
        actual_render_verified=verified,
        asset_stability_verified=stable,
        ready_for_qualification=preflight.preflight_ready and verified
        and stable and not failures,
        failures=tuple(failures),
        preflight=preflight,
        deployment=deployment,
        output=image,
    )
