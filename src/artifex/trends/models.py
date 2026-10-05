from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrendModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RawTrendSignal(TrendModel):
    provider: str = Field(min_length=1, max_length=100)
    external_id: str | None = Field(default=None, max_length=300)
    topic: str = Field(min_length=1, max_length=300)
    strength: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    observed_at: datetime
    expires_at: datetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TrendSourceReference(TrendModel):
    provider: str
    external_id: str | None = None
    strength: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    observed_at: datetime


class NormalizedTrendSignal(TrendModel):
    id: str
    normalized_topic: str
    topic: str
    strength: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    observed_at: datetime
    expires_at: datetime
    sources: tuple[TrendSourceReference, ...]
    payload: dict[str, Any] = Field(default_factory=dict)
