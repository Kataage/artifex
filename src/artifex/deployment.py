from __future__ import annotations

import asyncio
import hashlib
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from artifex.comfy import (
    ComfyUIClient,
    WorkflowPatchRequest,
    WorkflowTemplateRegistry,
)
from artifex.config.models import ArtifexSettings
from artifex.controller_preflight import ControllerPreflight, check_controller
from artifex.render_node.preflight import RenderNodePreflight, check_render_node
from artifex.windows_tasks import StartupRole, StartupTaskStatus, task_status

DeploymentRole = Literal["controller", "renderer"]


class DeploymentCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ready: bool
    blocking: bool
    detail: str


class RenderSmokeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str
    prompt_id: str
    image_path: Path
    image_sha256: str = Field(min_length=64, max_length=64)
    width: int
    height: int
    bytes: int


class DeploymentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: DeploymentRole
    ready: bool
    checks: tuple[DeploymentCheck, ...]
    smoke: RenderSmokeEvidence | None = None


def _request(
    settings: ArtifexSettings,
    *,
    output_prefix: str,
    width: int,
    height: int,
) -> WorkflowPatchRequest:
    cfg = settings.comfyui
    return WorkflowPatchRequest(
        positive_prompt="solo, 1girl, simple background, deployment verification",
        negative_prompt="low quality, blurry, artifacts",
        checkpoint=settings.production.checkpoint or "__CHECKPOINT_NOT_CONFIGURED__",
        refiner_checkpoint=cfg.refiner_checkpoint,
        vae=cfg.vae,
        upscale_model=cfg.upscale_model,
        seed=42,
        width=width,
        height=height,
        batch_size=1,
        output_prefix=output_prefix,
        base_steps=cfg.base_steps,
        base_cfg=cfg.base_cfg,
        base_sampler=cfg.base_sampler,
        base_scheduler=cfg.base_scheduler,
        base_denoise=cfg.base_denoise,
        refiner_steps=cfg.refiner_steps,
        refiner_cfg=cfg.refiner_cfg,
        refiner_sampler=cfg.refiner_sampler,
        refiner_scheduler=cfg.refiner_scheduler,
        refiner_denoise=cfg.refiner_denoise,
        upscale_steps=cfg.upscale_steps,
        upscale_cfg=cfg.upscale_cfg,
        upscale_sampler=cfg.upscale_sampler,
        upscale_scheduler=cfg.upscale_scheduler,
        upscale_denoise=cfg.upscale_denoise,
    )


async def _workflow_checks(
    settings: ArtifexSettings, client: ComfyUIClient
) -> tuple[DeploymentCheck, ...]:
    registry = WorkflowTemplateRegistry.with_packaged_templates()
    results: list[DeploymentCheck] = []
    ids = (
        settings.comfyui.default_template,
        settings.production.repair_workflow_template,
    )
    for template_id in dict.fromkeys(ids):
        try:
            template = registry.require(template_id)
            request = _request(
                settings,
                output_prefix="ARTIFEX/deployment/readiness",
                width=settings.production.width,
                height=settings.production.height,
            )
            outcome = await client.validate_requirements(template.requirements(request))
            results.append(
                DeploymentCheck(
                    name=f"workflow:{template_id}",
                    ready=outcome.ready,
                    blocking=True,
                    detail=outcome.detail,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report failed dependency, not success
            results.append(
                DeploymentCheck(
                    name=f"workflow:{template_id}",
                    ready=False,
                    blocking=True,
                    detail=f"Workflow dependency verification failed: {exc}",
                )
            )
    return tuple(results)


async def _render_smoke(
    settings: ArtifexSettings,
    client: ComfyUIClient,
    *,
    output_dir: Path | None = None,
    width: int | None = None,
    height: int | None = None,
) -> RenderSmokeEvidence:
    """Explicit opt-in GPU workflow, /history completion and /view image integrity.

    This intentionally does not create a production Pack or qualification pass.
    The test uses the actual configured workflow and does not mutate its template.
    """
    if not settings.production.checkpoint:
        raise ValueError("production checkpoint must be configured for render smoke")
    if settings.comfyui.output_mode != "api":
        raise ValueError("two-PC render smoke requires comfyui.output_mode=api")

    template_id = settings.comfyui.default_template
    template = WorkflowTemplateRegistry.with_packaged_templates().require(template_id)
    run_id = uuid4().hex
    prefix = f"ARTIFEX/deployment-smoke/{run_id}"
    request = _request(
        settings,
        output_prefix=prefix,
        width=width if width is not None else settings.production.width,
        height=height if height is not None else settings.production.height,
    )
    result = await client.execute(
        template,
        request,
        timeout_seconds=settings.comfyui.execution_timeout_seconds,
    )
    if not result.completed:
        raise RuntimeError(
            f"ComfyUI did not complete smoke prompt {result.prompt_id}: {result.status}"
        )
    images = [
        output
        for output in result.outputs
        if Path(output.filename).suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
        and output.output_type == "output"
    ]
    if not images:
        raise RuntimeError(
            f"ComfyUI prompt {result.prompt_id} completed but returned no output images"
        )

    root = (
        output_dir.expanduser().resolve(strict=False)
        if output_dir is not None
        else (settings.comfyui.download_dir / "deployment-smoke").expanduser().resolve(
            strict=False
        )
    )
    # Each verification lives in its own folder. Never overwrite an operator image.
    destination = root / run_id
    downloaded = await client.download_output(images[0], destination)
    digest = hashlib.sha256()
    size = 0
    with downloaded.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    try:
        with Image.open(downloaded) as image:
            image.verify()
        with Image.open(downloaded) as image:
            image_width, image_height = image.size
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        downloaded.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded ComfyUI image failed full Pillow verification: {exc}"
        ) from exc

    return RenderSmokeEvidence(
        template_id=template_id,
        prompt_id=result.prompt_id,
        image_path=downloaded,
        image_sha256=digest.hexdigest(),
        width=image_width,
        height=image_height,
        bytes=size,
    )


async def verify_deployment(
    settings: ArtifexSettings,
    *,
    role: DeploymentRole,
    require_autostart: bool = False,
    render_smoke: bool = False,
    output_dir: Path | None = None,
    width: int | None = None,
    height: int | None = None,
    controller_check: Callable[[ArtifexSettings], ControllerPreflight] = check_controller,
    renderer_check: Callable[[ArtifexSettings], RenderNodePreflight] = check_render_node,
    startup_check: Callable[[StartupRole], StartupTaskStatus] = task_status,
    comfy_client: ComfyUIClient | None = None,
) -> DeploymentReport:
    """Aggregate existing checks and optionally prove a real rendered image.

    Read-only by default. Startup-task registration, model downloads, daemon
    spawning and qualification stage recordings are intentionally out of scope.
    """
    if role == "renderer" and render_smoke:
        raise ValueError("render smoke runs from PC-A (role controller) only")
    if (width is not None or height is not None or output_dir is not None) and not render_smoke:
        raise ValueError("--width/--height/--output-dir require --render-smoke")

    checks: list[DeploymentCheck] = []
    smoke: RenderSmokeEvidence | None = None

    if role == "controller":
        try:
            report = await asyncio.to_thread(controller_check, settings)
            checks.extend(
                DeploymentCheck(
                    name=f"controller:{check.name}",
                    ready=check.ready,
                    blocking=True,
                    detail=check.detail,
                )
                for check in report.checks
            )
        except Exception as exc:  # noqa: BLE001 - connectivity may be fully offline
            checks.append(
                DeploymentCheck(
                    name="controller:preflight",
                    ready=False,
                    blocking=True,
                    detail=f"Controller preflight failed: {exc}",
                )
            )
    else:
        try:
            report_renderer = await asyncio.to_thread(renderer_check, settings)
            checks.append(
                DeploymentCheck(
                    name="renderer:preflight",
                    ready=report_renderer.ready,
                    blocking=True,
                    detail=(
                        f"GPU count={report_renderer.gpu_count}; "
                        f"LoRA count={report_renderer.lora_count}; "
                        + ("; ".join(report_renderer.issues) or "renderer checks passed")
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep a structured failure
            checks.append(
                DeploymentCheck(
                    name="renderer:preflight",
                    ready=False,
                    blocking=True,
                    detail=f"Renderer preflight failed: {exc}",
                )
            )

    if platform.system() == "Windows":
        try:
            task = await asyncio.to_thread(startup_check, role)
            checks.append(
                DeploymentCheck(
                    name="windows_autostart",
                    ready=task.installed and task.managed,
                    blocking=require_autostart,
                    detail=(
                        f"{task.task_name}: installed={task.installed}, "
                        f"Artifex-owned={task.managed}, state={task.state}"
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - not a deployment crash
            checks.append(
                DeploymentCheck(
                    name="windows_autostart",
                    ready=False,
                    blocking=require_autostart,
                    detail=f"Could not inspect Windows scheduled task: {exc}",
                )
            )
    elif require_autostart:
        checks.append(
            DeploymentCheck(
                name="windows_autostart",
                ready=False,
                blocking=True,
                detail="Windows autostart requires a native Windows host",
            )
        )

    owns_client = comfy_client is None
    client: ComfyUIClient | None = comfy_client
    if role == "controller":
        client = client or ComfyUIClient(settings.comfyui)
        try:
            checks.extend(await _workflow_checks(settings, client))
            if render_smoke:
                if any(not check.ready for check in checks if check.blocking):
                    checks.append(
                        DeploymentCheck(
                            name="render_smoke",
                            ready=False,
                            blocking=True,
                            detail="Skipped: blocking deployment prerequisites did not pass",
                        )
                    )
                else:
                    try:
                        smoke = await _render_smoke(
                            settings, client,
                            output_dir=output_dir, width=width, height=height,
                        )
                        checks.append(
                            DeploymentCheck(
                                name="render_smoke",
                                ready=True,
                                blocking=True,
                                detail=(
                                    f"Real ComfyUI image saved and validated: "
                                    f"{smoke.image_path} (SHA-256 {smoke.image_sha256})"
                                ),
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        checks.append(
                            DeploymentCheck(
                                name="render_smoke",
                                ready=False,
                                blocking=True,
                                detail=f"Image generation or transport verification failed: {exc}",
                            )
                        )
        finally:
            if owns_client:
                await client.aclose()
    return DeploymentReport(
        role=role,
        ready=all(item.ready for item in checks if item.blocking),
        checks=tuple(checks),
        smoke=smoke,
    )
