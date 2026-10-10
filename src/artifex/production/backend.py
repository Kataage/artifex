from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from artifex.comfy import (
    ComfyOutput,
    ComfyUIClient,
    ComfyUIError,
    ComfyUIExecutionError,
    ComfyUIProtocolError,
    WorkflowLoRA,
    WorkflowPatchRequest,
    WorkflowTemplateRegistry,
)
from artifex.config.models import ComfyUiConfig, ProductionConfig
from artifex.loras import LoRAPlan
from artifex.production.output_files import resolve_existing_comfy_outputs
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
    # Reuse the persisted ComfyUI prompt after a transport error. Never
    # re-submit an already accepted GPU job just to wait/download again.
    resume_prompt_id: str | None = Field(default=None, min_length=1)


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
            "render_node_id": self._comfy_config.render_node_id,
            "base_url": self._comfy_config.base_url,
            "output_mode": self._comfy_config.output_mode,
            "workflow_template": self._template.template_id,
            "workflow_version": self._template.version,
            "model_family": self._template.model_family,
            "checkpoint": self._production.checkpoint,
            "width": self._production.width,
            "height": self._production.height,
            "batch_size": self._production.batch_size,
            "base_sampler": self._comfy_config.base_sampler,
            "base_scheduler": self._comfy_config.base_scheduler,
            "base_steps": self._comfy_config.base_steps,
            "base_cfg": self._comfy_config.base_cfg,
            "refiner_checkpoint": self._comfy_config.refiner_checkpoint,
            "refiner_sampler": self._comfy_config.refiner_sampler,
            "refiner_scheduler": self._comfy_config.refiner_scheduler,
            "refiner_steps": self._comfy_config.refiner_steps,
            "upscale_model": self._comfy_config.upscale_model,
            "release_vram_after_attempt": (
                self._comfy_config.release_vram_after_attempt
            ),
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
        if self._comfy_config.output_mode == "filesystem" and output_dir is None:
            raise RuntimeError(
                "comfyui.output_dir must be configured when output_mode=filesystem"
            )

        loras = tuple(
            WorkflowLoRA(
                name=entry.asset_name or Path(entry.path).name,
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
        # A timeout after /prompt does not mean ComfyUI stopped rendering.
        # Never send /free to a potentially active or ambiguously submitted
        # job; only release cached models once remote completion is confirmed.
        remote_completed = False
        try:
            graph = template.patch(patch)
            if request.resume_prompt_id is not None:
                # A receipt was already durably recorded by the coordinator.
                # Resuming must not enter /prompt or change ComfyUI's queue.
                prompt_id = request.resume_prompt_id
            else:
                receipt = await self._client.submit(graph)
                on_submitted(receipt.prompt_id)
                prompt_id = receipt.prompt_id
            result = await self._client.wait_for_completion(prompt_id)
            remote_completed = result.completed
            if not result.outputs:
                raise ComfyUIExecutionError(
                    f"ComfyUI completed without image outputs: {prompt_id}",
                    retryable=False,
                )

            paths = await self._materialize_outputs(
                result.outputs,
                scene_id=request.scene_id,
                attempt_id=request.attempt_id,
            )
            return GeneratedBatch(
                prompt_id=prompt_id,
                output_paths=paths,
                outputs=tuple(
                    output.model_dump(mode="json")
                    for output in result.outputs
                ),
            )
        except Exception:
            if (
                remote_completed
                and self._comfy_config.release_vram_on_error
                and not self._comfy_config.release_vram_after_attempt
            ):
                await self._release_vram_best_effort()
            raise
        finally:
            if remote_completed and self._comfy_config.release_vram_after_attempt:
                await self._release_vram_best_effort()

    async def recover_completed(
        self,
        *,
        prompt_id: str,
        scene_id: str,
        attempt_id: str,
        outputs: list[dict[str, Any]],
    ) -> GeneratedBatch:
        """Recover confirmed ComfyUI history outputs, never issuing /prompt.

        RecoveryManager has already checked the completed remote prompt and
        persisted its output references. This step materializes the same
        output images after PC-A restart, using the normal API/shared-folder
        transport and path safety checks.
        """
        if not prompt_id:
            raise ComfyUIProtocolError("completed recovery lacks prompt ID")
        try:
            saved = tuple(ComfyOutput.model_validate(item) for item in outputs)
        except (ValueError, TypeError, ValidationError) as exc:
            raise ComfyUIProtocolError(
                "recovered ComfyUI output references are malformed"
            ) from exc
        paths = await self._materialize_outputs(
            saved, scene_id=scene_id, attempt_id=attempt_id,
        )
        return GeneratedBatch(
            prompt_id=prompt_id,
            output_paths=paths,
            outputs=tuple(output.model_dump(mode="json") for output in saved),
        )

    async def _materialize_outputs(
        self,
        outputs: tuple[ComfyOutput, ...],
        *,
        scene_id: str,
        attempt_id: str,
    ) -> tuple[Path, ...]:
        selected = tuple(
            item for item in outputs if item.output_type == "output"
        )
        if not selected:
            raise ComfyUIExecutionError(
                "completed ComfyUI prompt has no output-type images",
                retryable=False,
            )
        if self._comfy_config.output_mode == "api":
            destination = (
                self._comfy_config.download_dir
                / self._comfy_config.render_node_id
                / scene_id
                / attempt_id
            )
            return tuple([
                await self._client.download_output(item, destination)
                for item in selected
            ])
        root = self._comfy_config.output_dir
        if root is None:
            raise ComfyUIExecutionError(
                "shared ComfyUI output_dir is not configured",
                retryable=False,
            )
        return resolve_existing_comfy_outputs(root, selected)

    async def _release_vram_best_effort(self) -> None:
        try:
            await self._client.free_memory(
                unload_models=True,
                free_memory=True,
            )
        except ComfyUIError:
            return
