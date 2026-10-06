from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from artifex.domain import ContentRating, ResultState


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationContext(EvaluationModel):
    attempt_id: str
    scene_id: str
    image_path: Path
    character_ids: tuple[str, ...] = Field(min_length=1)
    positive_prompt: str
    negative_prompt: str
    reference_image_paths: tuple[Path, ...] = ()
    adjacent_image_paths: tuple[Path, ...] = ()


class RawEvaluationSignals(EvaluationModel):
    identity: float = Field(ge=0, le=1)
    alignment: float = Field(ge=0, le=1)
    face_quality: float = Field(ge=0, le=1)
    technical_quality: float = Field(ge=0, le=1)
    aesthetic: float = Field(ge=0, le=1)
    continuity: float = Field(ge=0, le=1)
    integrity: float = Field(ge=0, le=1)
    reasons: tuple[str, ...] = ()
    content_rating: ContentRating = ContentRating.GENERAL
    content_labels: tuple[str, ...] = ()


class EvaluationScores(EvaluationModel):
    identity: float = Field(ge=0, le=1)
    alignment: float = Field(ge=0, le=1)
    face_quality: float = Field(ge=0, le=1)
    technical_quality: float = Field(ge=0, le=1)
    aesthetic: float = Field(ge=0, le=1)
    image_similarity: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    continuity: float = Field(ge=0, le=1)
    integrity: float = Field(ge=0, le=1)
    aggregate: float = Field(ge=0, le=1)


class EvaluationResult(EvaluationModel):
    attempt_id: str
    state: ResultState
    scores: EvaluationScores
    reasons: tuple[str, ...] = ()
    content_rating: ContentRating = ContentRating.GENERAL
    content_labels: tuple[str, ...] = ()
