from __future__ import annotations

from artifex.comfy.errors import ComfyErrorKind, ComfyUIError
from artifex.config.models import ProductionConfig
from artifex.domain import ResultState
from artifex.evaluation.models import EvaluationResult, EvaluationScores
from artifex.retry import RetryAction, RetryPolicy


def _result(
    state: ResultState,
    reasons: tuple[str, ...],
) -> EvaluationResult:
    return EvaluationResult(
        attempt_id="attempt-1",
        state=state,
        scores=EvaluationScores(
            identity=0.5,
            alignment=0.8,
            face_quality=0.8,
            technical_quality=0.8,
            aesthetic=0.8,
            image_similarity=0.0,
            novelty=1.0,
            continuity=0.8,
            integrity=1.0,
            aggregate=0.8,
        ),
        reasons=reasons,
    )


def test_identity_failure_routes_to_lora_and_prompt_changes() -> None:
    decision = RetryPolicy(ProductionConfig()).for_evaluation(
        _result(ResultState.REJECTED, ("identity_hard_failure",)),
        production_retries_used=0,
    )

    assert decision.should_retry is True
    assert decision.actions == (
        RetryAction.ALTERNATE_LORA,
        RetryAction.ADJUST_LORA_WEIGHT,
        RetryAction.REVISE_PROMPT,
    )
    assert decision.next_retry_number == 1


def test_production_retry_budget_is_bounded() -> None:
    decision = RetryPolicy(ProductionConfig(retry_limit=3)).for_evaluation(
        _result(ResultState.REJECTED, ("technical_quality_needs_review",)),
        production_retries_used=3,
    )

    assert decision.should_retry is False
    assert decision.actions == (RetryAction.OPERATOR_REVIEW,)


def test_infrastructure_retry_has_separate_budget() -> None:
    policy = RetryPolicy(
        ProductionConfig(retry_limit=0, infrastructure_retry_limit=2)
    )
    error = ComfyUIError(
        ComfyErrorKind.CONNECTION,
        "offline",
        retryable=True,
    )

    decision = policy.for_infrastructure(error, infrastructure_retries_used=0)

    assert decision.should_retry is True
    assert decision.actions == (RetryAction.INFRASTRUCTURE_RETRY,)
    assert decision.next_retry_number == 1


def test_missing_asset_is_blocked_not_blindly_retried() -> None:
    decision = RetryPolicy.for_missing_asset("required LoRA absent")

    assert decision.should_retry is False
    assert decision.actions == (RetryAction.BLOCK_MISSING_ASSET,)
