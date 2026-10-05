from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EditorialModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PackInventoryState(StrEnum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    CONSUMED = "consumed"
    EXPIRED = "expired"


class EditorialLane(StrEnum):
    STANDALONE = "standalone"
    SERIES = "series"


class SeriesPlanKind(StrEnum):
    START = "start"
    CONTINUE = "continue"
    BONUS = "bonus"


class InventoryItem(EditorialModel):
    pack_id: str
    state: PackInventoryState
    reserved_at: datetime | None = None
    consumed_at: datetime | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime


class InventoryCounts(EditorialModel):
    available: int = Field(ge=0)
    reserved: int = Field(ge=0)
    consumed: int = Field(ge=0)
    expired: int = Field(ge=0)


class EditorialPlanDecision(EditorialModel):
    decision_id: str | None
    lane: EditorialLane
    reason: str
    series_id: str | None = None
    concept_id: str | None = None
    series_plan_kind: SeriesPlanKind | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
