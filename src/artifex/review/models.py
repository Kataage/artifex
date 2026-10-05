from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReviewState(StrEnum):
    OPEN = "open"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETRY = "retry"
    SKIPPED = "skipped"


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    subject_type: str
    subject_id: str
    state: ReviewState
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    resolved_at: datetime | None = None
