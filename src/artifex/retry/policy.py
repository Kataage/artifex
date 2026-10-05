from __future__ import annotations

from artifex.comfy.errors import ComfyUIError
from artifex.config.models import ProductionConfig
from artifex.domain import ResultState
from artifex.evaluation.models import EvaluationResult
from artifex.retry.models import RetryAction, RetryDecision


class RetryPolicy:
    def __init__(self, config: ProductionConfig) -> None:
        self._config = config

    def for_evaluation(
        self,
        result: EvaluationResult,
        *,
        production_retries_used: int,
    ) -> RetryDecision:
        if result.state is ResultState.ACCEPTED:
            return RetryDecision(
                should_retry=False,
                reason="evaluation accepted",
            )

        actions = self._actions_for_reasons(result.reasons)
        if not actions and result.state is ResultState.REVIEW:
            return RetryDecision(
                should_retry=False,
                actions=(RetryAction.OPERATOR_REVIEW,),
                reason="review requires operator decision",
            )

        if production_retries_used >= self._config.retry_limit:
            return RetryDecision(
                should_retry=False,
                actions=(RetryAction.OPERATOR_REVIEW,),
                reason="production retry budget exhausted",
            )

        if not actions:
            actions = (RetryAction.CHANGE_SEED,)

        return RetryDecision(
            should_retry=True,
            actions=actions,
            reason="reason-aware production retry",
            next_retry_number=production_retries_used + 1,
        )

    def for_infrastructure(
        self,
        error: ComfyUIError,
        *,
        infrastructure_retries_used: int,
    ) -> RetryDecision:
        if not error.retryable:
            return RetryDecision(
                should_retry=False,
                actions=(RetryAction.OPERATOR_REVIEW,),
                reason=f"non-retryable infrastructure failure: {error.kind.value}",
            )
        if infrastructure_retries_used >= self._config.infrastructure_retry_limit:
            return RetryDecision(
                should_retry=False,
                actions=(RetryAction.OPERATOR_REVIEW,),
                reason="infrastructure retry budget exhausted",
            )
        return RetryDecision(
            should_retry=True,
            actions=(RetryAction.INFRASTRUCTURE_RETRY,),
            reason=f"retryable infrastructure failure: {error.kind.value}",
            next_retry_number=infrastructure_retries_used + 1,
        )

    @staticmethod
    def for_missing_asset(detail: str) -> RetryDecision:
        return RetryDecision(
            should_retry=False,
            actions=(RetryAction.BLOCK_MISSING_ASSET,),
            reason=f"missing asset: {detail}",
        )

    @staticmethod
    def _actions_for_reasons(reasons: tuple[str, ...]) -> tuple[RetryAction, ...]:
        actions: list[RetryAction] = []

        def add(*values: RetryAction) -> None:
            for value in values:
                if value not in actions:
                    actions.append(value)

        for reason in reasons:
            if reason.startswith("identity_"):
                add(
                    RetryAction.ALTERNATE_LORA,
                    RetryAction.ADJUST_LORA_WEIGHT,
                    RetryAction.REVISE_PROMPT,
                )
            elif reason.startswith("alignment_"):
                add(RetryAction.REVISE_PROMPT)
            elif reason.startswith(("face_quality_", "technical_quality_")):
                add(RetryAction.CHANGE_SEED, RetryAction.REPAIR_WORKFLOW)
            elif reason.startswith("continuity_"):
                add(RetryAction.REVISE_PROMPT)
            elif "similarity" in reason:
                add(RetryAction.VARY_SCENE, RetryAction.CHANGE_SEED)
            elif "integrity" in reason:
                add(RetryAction.REPAIR_WORKFLOW)
            elif reason.startswith("aggregate_"):
                add(RetryAction.CHANGE_SEED)

        return tuple(actions)
