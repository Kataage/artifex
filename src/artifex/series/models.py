from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.planner.models import PackFormat


class SeriesStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class SeriesProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    status: SeriesStatus = SeriesStatus.ACTIVE
    current_episode: int = Field(default=0, ge=0)
    prior_pack_ids: tuple[str, ...] = ()
    character_ids: tuple[str, ...] = Field(min_length=1)
    continuity_state: dict[str, Any] = Field(default_factory=dict)
    bible: tuple[str, ...] = ()
    rolling_summary: str = ""
    recent_episode_summaries: tuple[str, ...] = ()
    unresolved_hooks: tuple[str, ...] = ()
    preferred_format: PackFormat | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_unique_values(self) -> SeriesProfile:
        if len(self.character_ids) != len(set(self.character_ids)):
            raise ValueError("series character_ids must be unique")
        if len(self.prior_pack_ids) != len(set(self.prior_pack_ids)):
            raise ValueError("series prior_pack_ids must be unique")
        return self
