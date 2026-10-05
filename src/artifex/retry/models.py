from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RetryAction(StrEnum):
    ALTERNATE_LORA = "alternate_lora"
    ADJUST_LORA_WEIGHT = "adjust_lora_weight"
    REVISE_PROMPT = "revise_prompt"
    CHANGE_SEED = "change_seed"
    REPAIR_WORKFLOW = "repair_workflow"
    VARY_SCENE = "vary_scene"
    INFRASTRUCTURE_RETRY = "infrastructure_retry"
    BLOCK_MISSING_ASSET = "block_missing_asset"
    OPERATOR_REVIEW = "operator_review"


class RetryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    should_retry: bool
    actions: tuple[RetryAction, ...] = ()
    reason: str
    next_retry_number: int | None = None
