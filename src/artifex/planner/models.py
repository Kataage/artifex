from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlannerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdeaSource(StrEnum):
    EVERGREEN = "evergreen"
    TREND = "trend"
    SEASONAL = "seasonal"
    EXPLORATION = "exploration"
    SERIES = "series"


class PackFormat(StrEnum):
    SINGLE_FEATURE = "single_feature"
    CONTINUATION = "continuation"
    MINI_STORY = "mini_story"
    VARIATION_PACK = "variation_pack"
    OUTFIT_FEATURE = "outfit_feature"
    SEASONAL = "seasonal"
    TREND = "trend"
    EVERGREEN = "evergreen"
    EXPERIMENTAL = "experimental"
    DUO = "duo"
    GROUP = "group"


class CharacterOption(PlannerModel):
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    readiness: float = Field(default=1.0, ge=0, le=1)
    recent_use_penalty: float = Field(default=0.0, ge=0, le=1)
    historical_performance: float = Field(default=0.5, ge=0, le=1)
    notes: tuple[str, ...] = ()


class TrendSignalSummary(PlannerModel):
    id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    strength: float = Field(ge=0, le=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    freshness: float = Field(ge=0, le=1)


class SeasonalEventSummary(PlannerModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    relevance: float = Field(default=1.0, ge=0, le=1)


class RecentConceptSummary(PlannerModel):
    concept_id: str = Field(min_length=1)
    character_ids: tuple[str, ...]
    theme: str
    setting: str
    visual_hook: str


class ContextProvenance(PlannerModel):
    version: str = Field(min_length=1)
    estimator: str = Field(min_length=1)
    section_estimated_tokens: dict[str, int]
    omitted_counts: dict[str, int]
    long_term_retrieval: str = Field(min_length=1)


class ResearchEvidenceSummary(PlannerModel):
    evidence_id: str = Field(min_length=1)
    source: str
    provider: str
    title: str
    finding: str


class ResearchBriefSummary(PlannerModel):
    id: str = Field(min_length=1)
    topic: str
    research_run_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    key_findings: tuple[str, ...]
    items: tuple[ResearchEvidenceSummary, ...] = ()
    generated_at: datetime
    expires_at: datetime
    freshness_confidence: float = Field(ge=0, le=1)
    degraded: bool
    adult: bool

    def fresh_at(self, when: datetime) -> bool:
        left = self.expires_at
        right = when
        if left.tzinfo is None and right.tzinfo is not None:
            left = left.replace(tzinfo=right.tzinfo)
        if right.tzinfo is None and left.tzinfo is not None:
            right = right.replace(tzinfo=left.tzinfo)
        return left > right


class PlanningContext(PlannerModel):
    as_of: datetime
    characters: tuple[CharacterOption, ...]
    trend_signals: tuple[TrendSignalSummary, ...] = ()
    seasonal_events: tuple[SeasonalEventSummary, ...] = ()
    recent_concepts: tuple[RecentConceptSummary, ...] = ()
    long_term_concepts: tuple[RecentConceptSummary, ...] = ()
    evergreen_prompts: tuple[str, ...] = ()
    operator_notes: tuple[str, ...] = ()
    research_brief: ResearchBriefSummary | None = None
    context_provenance: ContextProvenance | None = None

    @model_validator(mode="after")
    def require_characters(self) -> PlanningContext:
        if not self.characters:
            raise ValueError("planning context requires at least one character")
        ids = [character.id for character in self.characters]
        if len(ids) != len(set(ids)):
            raise ValueError("planning context character ids must be unique")
        return self


class CreativeAssessment(PlannerModel):
    character_fit: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    visual_strength: float = Field(ge=0, le=1)
    series_potential: float = Field(ge=0, le=1)


class ConceptCandidate(PlannerModel):
    candidate_key: str = Field(min_length=1, max_length=80)
    character_ids: tuple[str, ...] = Field(min_length=1)
    idea_source: IdeaSource
    format: PackFormat
    theme: str = Field(min_length=1, max_length=300)
    setting: str = Field(min_length=1, max_length=300)
    mood: str = Field(min_length=1, max_length=300)
    visual_hook: str = Field(min_length=1, max_length=500)
    progression: str = Field(min_length=1, max_length=1000)
    target_scene_count: int = Field(ge=1, le=12)
    continuity_requirements: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    research_run_ids: tuple[str, ...] = ()
    research_evidence_ids: tuple[str, ...] = ()
    research_notes: tuple[str, ...] = ()
    assessment: CreativeAssessment

    @model_validator(mode="after")
    def validate_character_cardinality(self) -> ConceptCandidate:
        count = len(self.character_ids)
        if self.format is PackFormat.DUO and count != 2:
            raise ValueError("duo format requires exactly two characters")
        if self.format is PackFormat.GROUP and count < 3:
            raise ValueError("group format requires at least three characters")
        if self.format not in {PackFormat.DUO, PackFormat.GROUP} and count != 1:
            raise ValueError("non-duo/group formats require exactly one character")
        return self


class ConceptCandidateBatch(PlannerModel):
    candidates: tuple[ConceptCandidate, ...]

    @model_validator(mode="after")
    def unique_candidate_keys(self) -> ConceptCandidateBatch:
        keys = [candidate.candidate_key for candidate in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate_key values must be unique")
        return self


class CandidateScore(PlannerModel):
    aggregate: float
    trend: float = Field(ge=0, le=1)
    evergreen: float = Field(ge=0, le=1)
    character_fit: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    visual_strength: float = Field(ge=0, le=1)
    historical_performance: float = Field(ge=0, le=1)
    seasonality: float = Field(ge=0, le=1)
    series_potential: float = Field(ge=0, le=1)
    readiness: float = Field(ge=0, le=1)
    similarity_penalty: float = Field(ge=0, le=1)
    recent_character_penalty: float = Field(ge=0, le=1)
    rejected_reason: str | None = None


class ScoredConcept(PlannerModel):
    candidate: ConceptCandidate
    score: CandidateScore


class SelectedConcept(PlannerModel):
    concept_id: str
    candidate: ConceptCandidate
    score: CandidateScore
