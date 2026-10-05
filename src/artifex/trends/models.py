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



class RawSeasonalEvent(TrendModel):
    provider: str = Field(min_length=1, max_length=100)
    external_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    relevance: float = Field(ge=0, le=1)
    starts_at: datetime
    ends_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class SeasonalEventRecord(TrendModel):
    id: str
    provider: str
    external_id: str
    title: str
    relevance: float = Field(ge=0, le=1)
    starts_at: datetime
    ends_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class SignalSourceHealth(TrendModel):
    provider: str
    kind: str
    state: str
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None
    consecutive_failures: int = Field(default=0, ge=0)
    last_count: int = Field(default=0, ge=0)
