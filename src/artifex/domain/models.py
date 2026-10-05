from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.domain.enums import CharacterStatus, LoRAPolicy, LoRAState


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CharacterOutfit(DomainModel):
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    canonical_tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    preferred_lora_ids: tuple[str, ...] = ()
    generation_notes: tuple[str, ...] = ()


class CharacterSource(DomainModel):
    kind: str = Field(min_length=1)
    url: str = Field(min_length=1)
    checked_at: datetime | None = None
    note: str | None = None


class CharacterProfile(DomainModel):
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    enabled: bool = True
    status: CharacterStatus = CharacterStatus.ACTIVE
    branch: str | None = None
    group: str | None = None
    generation: str | None = None
    canonical_tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ("ilxl",)
    lora_policy: LoRAPolicy = LoRAPolicy.OPTIONAL
    preferred_lora_ids: tuple[str, ...] = ()
    readiness: float = Field(default=0.0, ge=0.0, le=1.0)
    policy_profile: str | None = None
    outfits: tuple[CharacterOutfit, ...] = ()
    reference_image_dirs: tuple[Path, ...] = ()
    provenance: tuple[CharacterSource, ...] = ()
    generation_notes: tuple[str, ...] = ()
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
    recommended_weight: float = Field(default=1.0, ge=-4.0, le=4.0)
    validated_min_weight: float | None = Field(default=None, ge=-4.0, le=4.0)
    validated_max_weight: float | None = Field(default=None, ge=-4.0, le=4.0)
    identity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    flexibility_score: float | None = Field(default=None, ge=0.0, le=1.0)
    readiness: float = Field(default=0.0, ge=0.0, le=1.0)
    checksum: str | None = None
    incompatible_lora_ids: tuple[str, ...] = ()
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_weight_range(self) -> LoRAProfile:
        if (
            self.validated_min_weight is not None
            and self.validated_max_weight is not None
            and self.validated_min_weight > self.validated_max_weight
        ):
            raise ValueError("validated_min_weight must be <= validated_max_weight")
        return self
