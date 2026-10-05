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
    seed: int = Field(ge=0)
    width: int = Field(default=1024, ge=64, le=8192)
    height: int = Field(default=1024, ge=64, le=8192)
    batch_size: int = Field(default=1, ge=1, le=64)
    output_prefix: str = Field(min_length=1)
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
