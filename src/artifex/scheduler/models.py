from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SchedulerAction(StrEnum):
    IDLE = "idle"
    RECOVER_PACK = "recover_pack"
    REPLENISH_IDEAS = "replenish_ideas"
    PLAN_PACK = "plan_pack"
    PLAN_SERIES_PACK = "plan_series_pack"
    RUN_PACK = "run_pack"


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    ideas: int
    planned: int
    completed_available: int
    completed_reserved: int = 0
    completed_consumed: int = 0
    completed_expired: int = 0


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    action: SchedulerAction
    reason: str
    pack_id: str | None = None
    series_id: str | None = None
    concept_id: str | None = None
    editorial_decision_id: str | None = None
    series_plan_kind: str | None = None
