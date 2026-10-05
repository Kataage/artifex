from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SchedulerAction(StrEnum):
    IDLE = "idle"
    RECOVER_PACK = "recover_pack"
    REPLENISH_IDEAS = "replenish_ideas"
    PLAN_PACK = "plan_pack"
    RUN_PACK = "run_pack"


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    ideas: int
    planned: int
    completed_available: int


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    action: SchedulerAction
    reason: str
    pack_id: str | None = None
