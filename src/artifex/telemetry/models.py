from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TelemetryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EventSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class OperationalEvent(TelemetryModel):
    id: int
    event_type: str
    severity: EventSeverity
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DailyMetrics(TelemetryModel):
    since: datetime
    until: datetime
    finalized_packs: int
    failed_packs: int
    blocked_packs: int
    generation_attempts: int
    accepted_evaluations: int
    review_evaluations: int
    rejected_evaluations: int
    open_reviews: int
    errors: int
