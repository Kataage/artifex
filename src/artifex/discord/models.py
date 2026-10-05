from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DiscordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CommandName(StrEnum):
    STATUS = "status"
    PAUSE = "pause"
    RESUME = "resume"
    CURRENT = "current"
    QUEUE = "queue"
    RECENT = "recent"
    APPROVE = "approve"
    REJECT = "reject"
    RETRY = "retry"
    SKIP = "skip"
    ALTERNATE = "alternate"
    NEXT = "next"
    CHARACTER = "character"
    SERIES = "series"
    SIGNALS = "signals"
    DETAILS = "details"


class OperatorContext(DiscordModel):
    user_id: int = Field(ge=1)
    role_ids: tuple[int, ...] = ()


class CommandRequest(DiscordModel):
    name: CommandName
    args: tuple[str, ...] = ()


class CommandResponse(DiscordModel):
    ok: bool
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    ephemeral: bool = True


class NotificationKind(StrEnum):
    COMPLETION = "completion"
    REVIEW = "review"
    ERROR = "error"
    BACKEND = "backend"
    DISK = "disk"
    DAILY_SUMMARY = "daily_summary"


class Notification(DiscordModel):
    kind: NotificationKind
    title: str
    body: str
    review_id: str | None = None
