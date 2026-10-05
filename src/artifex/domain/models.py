from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from artifex.domain.enums import LoRAPolicy, LoRAState


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CharacterProfile(DomainModel):
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    enabled: bool = True
    canonical_tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ("ilxl",)
    lora_policy: LoRAPolicy = LoRAPolicy.OPTIONAL
    preferred_lora_ids: tuple[str, ...] = ()
    readiness: float = Field(default=0.0, ge=0.0, le=1.0)
    policy_profile: str | None = None
    last_used_at: datetime | None = None
    total_generated: int = Field(default=0, ge=0)


class LoRAProfile(DomainModel):
    id: str = Field(min_length=1)
    path: Path
    state: LoRAState = LoRAState.DISCOVERED
    lora_type: str = "character"
    target_character_ids: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ()
    trigger_tags: tuple[str, ...] = ()
    recommended_weight: float = 1.0
    validated_min_weight: float | None = None
    validated_max_weight: float | None = None
    readiness: float = Field(default=0.0, ge=0.0, le=1.0)
    checksum: str | None = None
