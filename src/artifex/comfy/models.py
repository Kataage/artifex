from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ComfyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComfyHealth(ComfyModel):
    available: bool
    version: str | None = None
    devices: tuple[str, ...] = ()
    detail: str | None = None


class WorkflowLoRA(ComfyModel):
    name: str = Field(min_length=1)
    weight_model: float = Field(default=1.0, ge=-4.0, le=4.0)
    weight_clip: float = Field(default=1.0, ge=-4.0, le=4.0)


class WorkflowPatchRequest(ComfyModel):
    positive_prompt: str
    negative_prompt: str
    checkpoint: str = Field(min_length=1)
    refiner_checkpoint: str = Field(default="anime-refiner-beta1.1.safetensors", min_length=1)
    vae: str = Field(default="pppanimixVAE_ilxl.safetensors", min_length=1)
    upscale_model: str = Field(default="4xRealisticrescaler_100000G.pt", min_length=1)
    seed: int = Field(ge=0)
    width: int = Field(default=1024, ge=64, le=8192)
    height: int = Field(default=1024, ge=64, le=8192)
    batch_size: int = Field(default=1, ge=1, le=64)
    output_prefix: str = Field(min_length=1)
    base_steps: int = Field(default=48, ge=1, le=200)
    base_cfg: float = Field(default=5.5, ge=0, le=30)
    base_sampler: str = Field(default="dpmpp_3m_sde_gpu", min_length=1)
    base_scheduler: str = Field(default="karras", min_length=1)
    base_denoise: float = Field(default=1.0, ge=0, le=1)
    refiner_steps: int = Field(default=24, ge=1, le=200)
    refiner_cfg: float = Field(default=5.0, ge=0, le=30)
    refiner_sampler: str = Field(default="dpmpp_3m_sde_gpu", min_length=1)
    refiner_scheduler: str = Field(default="karras", min_length=1)
    refiner_denoise: float = Field(default=0.10, ge=0, le=1)
    upscale_steps: int = Field(default=15, ge=1, le=200)
    upscale_cfg: float = Field(default=5.5, ge=0, le=30)
    upscale_sampler: str = Field(default="euler_ancestral", min_length=1)
    upscale_scheduler: str = Field(default="karras", min_length=1)
    upscale_denoise: float = Field(default=0.50, ge=0, le=1)
    loras: tuple[WorkflowLoRA, ...] = ()


class QueueReceipt(ComfyModel):
    prompt_id: str
    queue_number: float | None = None


class ComfyOutput(ComfyModel):
    node_id: str
    filename: str
    subfolder: str = ""
    output_type: str = "output"


class ComfyExecutionResult(ComfyModel):
    prompt_id: str
    completed: bool
    status: str
    outputs: tuple[ComfyOutput, ...]
    raw_status: dict[str, Any] = Field(default_factory=dict)
