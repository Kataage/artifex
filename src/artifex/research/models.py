from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchSource(StrEnum):
    WEB = "web"
    IMAGES = "images"
    NEWS = "news"
    VIDEOS = "videos"
    TAGS = "tags"


class ResearchIntent(StrEnum):
    CURRENT = "current"
    EVERGREEN = "evergreen"
    CHARACTER = "character"
    OUTFIT = "outfit"
    COMPOSITION = "composition"
    SEASONAL = "seasonal"
    POLICY = "policy"
    ADULT = "adult"


class SafeSearch(StrEnum):
    ON = "on"
    MODERATE = "moderate"
    OFF = "off"


class ProviderState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISABLED = "disabled"


class ResearchRunState(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


class ResearchSearchRequest(ResearchModel):
    query: str = Field(min_length=1, max_length=500)
    source: SearchSource = SearchSource.WEB
    intent: ResearchIntent = ResearchIntent.EVERGREEN
    max_results: int = Field(default=8, ge=1, le=50)
    region: str = Field(default="jp-jp", min_length=2, max_length=32)
    safesearch: SafeSearch = SafeSearch.MODERATE
    timelimit: str | None = Field(default=None, pattern="^(d|w|m|y)$")
    adult: bool = False
    backend: str | None = None


class ProviderResult(ResearchModel):
    provider: str = Field(min_length=1)
    source: SearchSource
    url: str = Field(min_length=1)
    title: str = ""
    snippet: str = ""
    image_url: str | None = None
    published_at: datetime | None = None
    provider_source: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(ResearchModel):
    provider: str
    state: ProviderState
    capabilities: tuple[SearchSource, ...]
    detail: str


class ResearchEvidence(ResearchModel):
    id: str
    run_id: str
    provider: str
    source: SearchSource
    rank: int = Field(ge=1)
    url: str
    canonical_url: str
    title: str
    snippet: str
    image_url: str | None = None
    published_at: datetime | None = None
    observed_at: datetime
    content_hash: str
    adult: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchSearchResponse(ResearchModel):
    run_id: str
    query: str
    source: SearchSource
    intent: ResearchIntent
    cached: bool
    degraded: bool
    evidence: tuple[ResearchEvidence, ...]
    errors: tuple[str, ...] = ()


class ResearchBriefItem(ResearchModel):
    evidence_id: str
    source: SearchSource
    provider: str
    title: str
    finding: str
    url: str
    published_at: datetime | None = None


class ResearchBrief(ResearchModel):
    id: str
    topic: str
    research_run_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    items: tuple[ResearchBriefItem, ...]
    key_findings: tuple[str, ...]
    generated_at: datetime
    expires_at: datetime
    freshness_confidence: float = Field(ge=0, le=1)
    degraded: bool
    adult: bool
    cache_key: str

    @model_validator(mode="after")
    def validate_evidence_links(self) -> ResearchBrief:
        item_ids = tuple(item.evidence_id for item in self.items)
        if not set(item_ids) <= set(self.evidence_ids):
            raise ValueError("ResearchBrief items must reference evidence_ids")
        return self

