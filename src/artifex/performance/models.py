from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PerformanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PerformanceMetrics(PerformanceModel):
    views: int | None = Field(default=None, ge=0)
    likes: int | None = Field(default=None, ge=0)
    comments: int | None = Field(default=None, ge=0)
    engagement_count: int | None = Field(default=None, ge=0)
    free_signups: int | None = Field(default=None, ge=0)
    paid_conversions: int | None = Field(default=None, ge=0)
    subscriber_delta: int | None = None
    revenue_cents: int | None = Field(default=None, ge=0)
    retention_rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_signal(self) -> PerformanceMetrics:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("performance snapshot requires at least one metric")
        return self

    @property
    def effective_engagement_count(self) -> int | None:
        if self.engagement_count is not None:
            return self.engagement_count
        if self.likes is None and self.comments is None:
            return None
        return (self.likes or 0) + (self.comments or 0)


class PublicationLink(PerformanceModel):
    id: str
    platform: str
    external_post_id: str
    pack_id: str
    scene_id: str | None = None
    publication_tier: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    source: str
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class PerformanceSnapshot(PerformanceModel):
    id: str
    publication_link_id: str
    observed_at: datetime
    metrics: PerformanceMetrics
    source: str
    provenance: dict[str, object] = Field(default_factory=dict)
    ingested_at: datetime


class PerformanceEvidence(PerformanceModel):
    publication_link_id: str
    pack_id: str
    concept_id: str | None = None
    scene_id: str | None = None
    series_id: str | None = None
    character_ids: tuple[str, ...] = ()
    format: str | None = None
    theme: str | None = None
    observed_at: datetime
    metrics: PerformanceMetrics
    raw_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    decay: float = Field(ge=0, le=1)
    component_scores: dict[str, float] = Field(default_factory=dict)


class PerformanceEffect(PerformanceModel):
    dimension: str
    key: str
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_count: int = Field(ge=0)
    effective_sample_size: float = Field(ge=0)
    metric_coverage: tuple[str, ...] = ()
    reason: str


class CandidatePerformance(PerformanceModel):
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    effects: tuple[PerformanceEffect, ...] = ()
    reason: str


class ManualPerformanceRecord(PerformanceModel):
    external_post_id: str
    pack_id: str
    scene_id: str | None = None
    publication_tier: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    observed_at: datetime
    metrics: PerformanceMetrics
    metadata: dict[str, object] = Field(default_factory=dict)


class ManualPerformanceImport(PerformanceModel):
    platform: str = "patreon"
    source: str = "manual"
    records: tuple[ManualPerformanceRecord, ...]
