from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.evaluation.semantic import EmbeddingModelDescriptor
from artifex.evaluation.similarity import SimilarityService


class CalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageCalibrationPair(CalibrationModel):
    left: Path
    right: Path
    note: str | None = None


class TextCalibrationPair(CalibrationModel):
    left: str = Field(min_length=1)
    right: str = Field(min_length=1)
    note: str | None = None


class SemanticCalibrationManifest(CalibrationModel):
    corpus_id: str = Field(min_length=1)
    profile_id: str = "siglip2-hololive-ilxl-v1"
    identity_positive_pairs: tuple[ImageCalibrationPair, ...] = ()
    identity_negative_pairs: tuple[ImageCalibrationPair, ...] = ()
    duplicate_positive_pairs: tuple[ImageCalibrationPair, ...] = ()
    duplicate_negative_pairs: tuple[ImageCalibrationPair, ...] = ()
    text_paraphrase_pairs: tuple[TextCalibrationPair, ...] = ()
    text_distinct_pairs: tuple[TextCalibrationPair, ...] = ()
    minimum_pairs_per_class: int = Field(default=4, ge=2)
    minimum_balanced_accuracy: float = Field(default=0.90, ge=0.5, le=1.0)

    @model_validator(mode="after")
    def require_all_classes(self) -> SemanticCalibrationManifest:
        groups = {
            "identity_positive_pairs": self.identity_positive_pairs,
            "identity_negative_pairs": self.identity_negative_pairs,
            "duplicate_positive_pairs": self.duplicate_positive_pairs,
            "duplicate_negative_pairs": self.duplicate_negative_pairs,
            "text_paraphrase_pairs": self.text_paraphrase_pairs,
            "text_distinct_pairs": self.text_distinct_pairs,
        }
        missing = [
            name
            for name, values in groups.items()
            if len(values) < self.minimum_pairs_per_class
        ]
        if missing:
            raise ValueError(
                "semantic calibration corpus is undersized: " + ", ".join(missing)
            )
        return self


class ThresholdCalibration(CalibrationModel):
    purpose: Literal["identity", "duplicate", "text_paraphrase"]
    threshold: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    positive_count: int = Field(ge=1)
    negative_count: int = Field(ge=1)
    positive_min: float = Field(ge=0, le=1)
    positive_mean: float = Field(ge=0, le=1)
    negative_max: float = Field(ge=0, le=1)
    negative_mean: float = Field(ge=0, le=1)
    margin: float
    validated: bool


class SemanticCalibrationProfile(CalibrationModel):
    profile_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    generated_at: datetime
    identity_hard_min: float = Field(ge=0, le=1)
    identity_accept_min: float = Field(ge=0, le=1)
    similarity_hard_max: float = Field(ge=0, le=1)
    planner_hard_similarity_threshold: float = Field(ge=0, le=1)
    identity: ThresholdCalibration
    duplicate: ThresholdCalibration
    text_paraphrase: ThresholdCalibration
    validated: bool


def load_calibration_manifest(path: Path) -> SemanticCalibrationManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = SemanticCalibrationManifest.model_validate(payload)
    root = path.parent

    def image_pair(pair: ImageCalibrationPair) -> ImageCalibrationPair:
        left = pair.left if pair.left.is_absolute() else root / pair.left
        right = pair.right if pair.right.is_absolute() else root / pair.right
        return pair.model_copy(update={"left": left, "right": right})

    resolved = manifest.model_copy(
        update={
            "identity_positive_pairs": tuple(
                image_pair(pair) for pair in manifest.identity_positive_pairs
            ),
            "identity_negative_pairs": tuple(
                image_pair(pair) for pair in manifest.identity_negative_pairs
            ),
            "duplicate_positive_pairs": tuple(
                image_pair(pair) for pair in manifest.duplicate_positive_pairs
            ),
            "duplicate_negative_pairs": tuple(
                image_pair(pair) for pair in manifest.duplicate_negative_pairs
            ),
        }
    )
    for pair in (
        *resolved.identity_positive_pairs,
        *resolved.identity_negative_pairs,
        *resolved.duplicate_positive_pairs,
        *resolved.duplicate_negative_pairs,
    ):
        if not pair.left.is_file() or not pair.right.is_file():
            raise FileNotFoundError(
                f"calibration image pair is missing: {pair.left} / {pair.right}"
            )
    return resolved


def load_calibration_profile(path: Path) -> SemanticCalibrationProfile:
    return SemanticCalibrationProfile.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def save_calibration_profile(
    profile: SemanticCalibrationProfile,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        profile.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _balanced_accuracy(
    positives: tuple[float, ...],
    negatives: tuple[float, ...],
    threshold: float,
) -> float:
    true_positive = sum(value >= threshold for value in positives) / len(positives)
    true_negative = sum(value < threshold for value in negatives) / len(negatives)
    return (true_positive + true_negative) / 2.0


def _calibrate_threshold(
    purpose: Literal["identity", "duplicate", "text_paraphrase"],
    positives: tuple[float, ...],
    negatives: tuple[float, ...],
    minimum_balanced_accuracy: float,
) -> ThresholdCalibration:
    if not positives or not negatives:
        raise ValueError(f"{purpose} calibration requires both classes")

    values = sorted({*positives, *negatives})
    candidates: list[float] = [0.0, 1.0]
    for left, right in zip(values, values[1:], strict=False):
        candidates.append((left + right) / 2.0)
    candidates.extend(values)

    scored = [
        (_balanced_accuracy(positives, negatives, threshold), threshold)
        for threshold in candidates
    ]
    best_accuracy = max(score for score, _ in scored)
    # Prefer the highest threshold among ties to reduce false-positive duplicate/identity
    # matches; identity's hard gate is softened separately below.
    threshold = max(
        threshold
        for score, threshold in scored
        if abs(score - best_accuracy) <= 1e-12
    )
    positive_min = min(positives)
    negative_max = max(negatives)
    return ThresholdCalibration(
        purpose=purpose,
        threshold=threshold,
        balanced_accuracy=best_accuracy,
        positive_count=len(positives),
        negative_count=len(negatives),
        positive_min=positive_min,
        positive_mean=fmean(positives),
        negative_max=negative_max,
        negative_mean=fmean(negatives),
        margin=positive_min - negative_max,
        validated=best_accuracy >= minimum_balanced_accuracy,
    )


class SemanticCalibrator:
    def __init__(
        self,
        similarity: SimilarityService,
        descriptor_getter: Callable[[], EmbeddingModelDescriptor],
    ) -> None:
        self._similarity = similarity
        self._descriptor_getter = descriptor_getter

    async def _image_scores(
        self,
        pairs: tuple[ImageCalibrationPair, ...],
    ) -> tuple[float, ...]:
        scores: list[float] = []
        for pair in pairs:
            scores.append(
                await self._similarity.max_image_similarity(pair.left, (pair.right,))
            )
        return tuple(scores)

    async def _text_scores(
        self,
        pairs: tuple[TextCalibrationPair, ...],
    ) -> tuple[float, ...]:
        scores: list[float] = []
        for pair in pairs:
            scores.append(
                await self._similarity.max_text_similarity(pair.left, (pair.right,))
            )
        return tuple(scores)

    async def calibrate(
        self,
        manifest: SemanticCalibrationManifest,
    ) -> SemanticCalibrationProfile:
        identity_positive = await self._image_scores(manifest.identity_positive_pairs)
        identity_negative = await self._image_scores(manifest.identity_negative_pairs)
        duplicate_positive = await self._image_scores(manifest.duplicate_positive_pairs)
        duplicate_negative = await self._image_scores(manifest.duplicate_negative_pairs)
        text_positive = await self._text_scores(manifest.text_paraphrase_pairs)
        text_negative = await self._text_scores(manifest.text_distinct_pairs)

        identity = _calibrate_threshold(
            "identity",
            identity_positive,
            identity_negative,
            manifest.minimum_balanced_accuracy,
        )
        duplicate = _calibrate_threshold(
            "duplicate",
            duplicate_positive,
            duplicate_negative,
            manifest.minimum_balanced_accuracy,
        )
        text_paraphrase = _calibrate_threshold(
            "text_paraphrase",
            text_positive,
            text_negative,
            manifest.minimum_balanced_accuracy,
        )
        descriptor = self._descriptor_getter()

        # Hard identity rejection is deliberately below the classification threshold.
        # Accept-level identity and duplicate/text thresholds use measured operating points.
        identity_hard_min = max(0.0, min(identity.threshold, identity.positive_min) - 0.05)
        validated = identity.validated and duplicate.validated and text_paraphrase.validated
        return SemanticCalibrationProfile(
            profile_id=manifest.profile_id,
            corpus_id=manifest.corpus_id,
            provider=descriptor.provider,
            model=descriptor.model,
            revision=descriptor.revision,
            generated_at=datetime.now(UTC),
            identity_hard_min=identity_hard_min,
            identity_accept_min=identity.threshold,
            similarity_hard_max=duplicate.threshold,
            planner_hard_similarity_threshold=text_paraphrase.threshold,
            identity=identity,
            duplicate=duplicate,
            text_paraphrase=text_paraphrase,
            validated=validated,
        )
