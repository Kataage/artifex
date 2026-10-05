from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PromptProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_id: str
    adapter_version: str
    model_family: str
    character_ids: tuple[str, ...]
    lora_ids: tuple[str, ...]
    lora_weights: tuple[float, ...]


class CompiledPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    positive_prompt: str
    negative_prompt: str
    positive_tags: tuple[str, ...]
    negative_tags: tuple[str, ...]
    provenance: PromptProvenance
