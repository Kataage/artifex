from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from artifex.comfy import (
    ComfyUIClient,
    ComfyUIError,
    ComfyUIExecutionError,
    WorkflowLoRA,
    WorkflowPatchRequest,
    WorkflowTemplateRegistry,
)
from artifex.config.models import ComfyUiConfig, ProductionConfig
from artifex.loras import LoRAPlan
from artifex.prompts import CompiledPrompt


class ProductionBackendModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerationRequest(ProductionBackendModel):
    attempt_id: str
    scene_id: str
    compiled: CompiledPrompt
    lora_plan: LoRAPlan
    seed: int = Field(ge=0)
    output_prefix: str
    workflow_template_id: str | None = None


class GeneratedBatch(ProductionBackendModel):
    prompt_id: str
    output_paths: tuple[Path, ...]
    outputs: tuple[dict[str, Any], ...] = ()


class GenerationBackend(Protocol):
    def provenance(self) -> Mapping[str, Any]: ...

    async def generate(
        self,
        request: GenerationRequest,
        *,
        on_submitted: Callable[[str], None],
    ) -> GeneratedBatch: ...


class ComfyGenerationBackend:
    def __init__(
        self,
        client: ComfyUIClient,
        templates: WorkflowTemplateRegistry,
        production: ProductionConfig,
        comfy_config: ComfyUiConfig,
    ) -> None:
        self._client = client
        self._templates = templates
        self._default_template_id = comfy_config.default_template
        self._template = templates.require(self._default_template_id)
        self._production = production
        self._comfy_config = comfy_config

    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": "comfyui",
            "workflow_template": self._template.template_id,
            "workflow_version": self._template.version,
            "model_family": self._template.model_family,
            "checkpoint": self._production.checkpoint,
            "width": self._production.width,
            "height": self._production.height,
            "batch_size": self._production.batch_size,
        }

    async def generate(
        self,
        request: GenerationRequest,
        *,
        on_submitted: Callable[[str], None],
    ) -> GeneratedBatch:
        template = self._templates.require(
            request.workflow_template_id or self._default_template_id
        )
        checkpoint = self._production.checkpoint
        if not checkpoint:
            raise RuntimeError("production.checkpoint must be configured")
        output_dir = self._comfy_config.output_dir
        if output_dir is None:
            raise RuntimeError("comfyui.output_dir must be configured for evaluation/archive")

        loras = tuple(
            WorkflowLoRA(
                name=Path(entry.path).name,
                weight_model=entry.weight,
                weight_clip=entry.weight,
            )
            for entry in request.lora_plan.entries
        )
        patch = WorkflowPatchRequest(
            positive_prompt=request.compiled.positive_prompt,
            negative_prompt=request.compiled.negative_prompt,
            checkpoint=checkpoint,
            refiner_checkpoint=self._comfy_config.refiner_checkpoint,
            vae=self._comfy_config.vae,
            upscale_model=self._comfy_config.upscale_model,
            seed=request.seed,
            width=self._production.width,
            height=self._production.height,
            batch_size=self._production.batch_size,
            output_prefix=request.output_prefix,
            base_steps=self._comfy_config.base_steps,
            base_cfg=self._comfy_config.base_cfg,
            base_sampler=self._comfy_config.base_sampler,
            base_scheduler=self._comfy_config.base_scheduler,
            base_denoise=self._comfy_config.base_denoise,
            refiner_steps=self._comfy_config.refiner_steps,
            refiner_cfg=self._comfy_config.refiner_cfg,
            refiner_sampler=self._comfy_config.refiner_sampler,
            refiner_scheduler=self._comfy_config.refiner_scheduler,
            refiner_denoise=self._comfy_config.refiner_denoise,
            upscale_steps=self._comfy_config.upscale_steps,
            upscale_cfg=self._comfy_config.upscale_cfg,
            upscale_sampler=self._comfy_config.upscale_sampler,
            upscale_scheduler=self._comfy_config.upscale_scheduler,
            upscale_denoise=self._comfy_config.upscale_denoise,
            loras=loras,
        )
        try:
            graph = template.patch(patch)
            receipt = await self._client.submit(graph)
            on_submitted(receipt.prompt_id)
            result = await self._client.wait_for_completion(receipt.prompt_id)
            if not result.outputs:
                raise ComfyUIExecutionError(
                    f"ComfyUI completed without image outputs: {receipt.prompt_id}",
                    retryable=False,
                )

            root = output_dir.expanduser().resolve()
            paths = tuple(
                root / output.subfolder / output.filename
                for output in result.outputs
                if output.output_type == "output"
            )
            if not paths:
                raise ComfyUIExecutionError(
                    f"ComfyUI returned no output-type images: {receipt.prompt_id}",
                    retryable=False,
                )
            return GeneratedBatch(
                prompt_id=receipt.prompt_id,
                output_paths=paths,
                outputs=tuple(
                    output.model_dump(mode="json")
                    for output in result.outputs
                ),
            )
        except Exception:
            if self._comfy_config.release_vram_on_error:
                await self._release_vram_best_effort()
            raise
        finally:
            if self._comfy_config.release_vram_after_attempt:
                await self._release_vram_best_effort()
