from __future__ import annotations

from artifex.config.models import EvaluationConfig
from artifex.domain import ResultState
from artifex.evaluation.models import (
    EvaluationContext,
    EvaluationResult,
    EvaluationScores,
)
from artifex.evaluation.provider import EvaluationProvider
from artifex.evaluation.similarity import SimilarityService


class EvaluationEngine:
    @property
    def config(self) -> EvaluationConfig:
        return self._config

    def __init__(
        self,
        provider: EvaluationProvider,
        similarity: SimilarityService,
        config: EvaluationConfig,
    ) -> None:
        self._provider = provider
        self._similarity = similarity
        self._config = config

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        raw = await self._provider.evaluate(context)
        duplicate_references = (
            context.duplicate_reference_image_paths
            or context.reference_image_paths
        )
        novelty_references = (
            context.novelty_reference_image_paths
            or duplicate_references
        )
        image_similarity = await self._similarity.max_image_similarity(
            context.image_path,
            duplicate_references,
        )
        novelty_similarity = await self._similarity.max_image_similarity(
            context.image_path,
            novelty_references,
        )
        identity_reference = await self._similarity.identity_similarity(
            context.image_path,
            context.identity_reference_image_paths,
        )
        novelty = 1.0 - novelty_similarity
        weights = self._config.weights
        aggregate = (
            raw.identity * weights.identity
            + raw.alignment * weights.alignment
            + raw.face_quality * weights.face_quality
            + raw.technical_quality * weights.technical_quality
            + raw.aesthetic * weights.aesthetic
            + novelty * weights.novelty
            + raw.continuity * weights.continuity
            + raw.integrity * weights.integrity
        )

        scores = EvaluationScores(
            identity=raw.identity,
            identity_reference=identity_reference.aggregate,
            alignment=raw.alignment,
            face_quality=raw.face_quality,
            technical_quality=raw.technical_quality,
            aesthetic=raw.aesthetic,
            image_similarity=image_similarity,
            novelty_similarity=novelty_similarity,
            novelty=novelty,
            continuity=raw.continuity,
            integrity=raw.integrity,
            aggregate=max(0.0, min(1.0, aggregate)),
        )
        state, reasons = self._classify(
            scores,
            raw.reasons,
            identity_reference_missing=(
                self._config.identity_reference_required
                and (
                    identity_reference.aggregate is None
                    or bool(identity_reference.missing_character_ids)
                )
            ),
        )
        return EvaluationResult(
            attempt_id=context.attempt_id,
            state=state,
            scores=scores,
            reasons=reasons,
            content_rating=raw.content_rating,
            content_labels=tuple(
                dict.fromkeys(
                    label.strip().casefold()
                    for label in raw.content_labels
                    if label.strip()
                )
            ),
        )

    def _classify(
        self,
        scores: EvaluationScores,
        provider_reasons: tuple[str, ...],
        *,
        identity_reference_missing: bool = False,
    ) -> tuple[ResultState, tuple[str, ...]]:
        reasons = list(provider_reasons)

        if scores.identity < self._config.identity_hard_min:
            reasons.append("identity_hard_failure")
        if (
            scores.identity_reference is not None
            and scores.identity_reference < self._config.identity_reference_hard_min
        ):
            reasons.append("identity_reference_hard_failure")
        if scores.integrity < self._config.integrity_hard_min:
            reasons.append("output_integrity_failure")
        if scores.image_similarity >= self._config.similarity_hard_max:
            reasons.append("image_similarity_hard_failure")

        if any(
            reason in reasons
            for reason in (
                "identity_hard_failure",
                "identity_reference_hard_failure",
                "output_integrity_failure",
                "image_similarity_hard_failure",
            )
        ):
            return ResultState.REJECTED, tuple(dict.fromkeys(reasons))

        if scores.identity < self._config.identity_accept_min:
            reasons.append("identity_needs_review")
        if identity_reference_missing:
            reasons.append("identity_reference_missing")
        elif scores.identity_reference < self._config.identity_reference_accept_min:
            reasons.append("identity_reference_needs_review")
        if scores.alignment < self._config.alignment_review_min:
            reasons.append("alignment_needs_review")
        if scores.face_quality < self._config.face_review_min:
            reasons.append("face_quality_needs_review")
        if scores.technical_quality < self._config.technical_review_min:
            reasons.append("technical_quality_needs_review")
        if scores.continuity < self._config.continuity_review_min:
            reasons.append("continuity_needs_review")

        signal_review = (
            any(reason.endswith("_needs_review") for reason in reasons)
            or "identity_reference_missing" in reasons
        )
        if (
            scores.aggregate >= self._config.accepted_score_min
            and not signal_review
        ):
            return ResultState.ACCEPTED, tuple(dict.fromkeys(reasons))

        if scores.aggregate >= self._config.review_score_min:
            if not reasons:
                reasons.append("aggregate_below_accept")
            return ResultState.REVIEW, tuple(dict.fromkeys(reasons))

        reasons.append("aggregate_below_review")
        return ResultState.REJECTED, tuple(dict.fromkeys(reasons))
