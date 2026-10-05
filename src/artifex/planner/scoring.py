from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import PlannerConfig
from artifex.planner.models import (
    CandidateScore,
    ConceptCandidate,
    IdeaSource,
    PlanningContext,
    ScoredConcept,
)


class CandidateSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trend: float = Field(default=0.0, ge=0, le=1)
    evergreen: float = Field(default=0.0, ge=0, le=1)
    character_fit: float = Field(default=0.0, ge=0, le=1)
    novelty: float = Field(default=0.0, ge=0, le=1)
    visual_strength: float = Field(default=0.0, ge=0, le=1)
    historical_performance: float = Field(default=0.0, ge=0, le=1)
    seasonality: float = Field(default=0.0, ge=0, le=1)
    series_potential: float = Field(default=0.0, ge=0, le=1)
    readiness: float = Field(default=0.0, ge=0, le=1)
    similarity_penalty: float = Field(default=0.0, ge=0, le=1)
    recent_character_penalty: float = Field(default=0.0, ge=0, le=1)


class SignalProvider(Protocol):
    async def evaluate(
        self,
        candidate: ConceptCandidate,
        context: PlanningContext,
    ) -> CandidateSignals: ...


class DefaultSignalProvider:
    async def evaluate(
        self,
        candidate: ConceptCandidate,
        context: PlanningContext,
    ) -> CandidateSignals:
        characters = {item.id: item for item in context.characters}
        selected = [characters[character_id] for character_id in candidate.character_ids]

        trend_by_id = {signal.id: signal for signal in context.trend_signals}
        seasonal_by_id = {event.id: event for event in context.seasonal_events}

        trend = max(
            (
                trend_by_id[ref].strength * trend_by_id[ref].freshness
                for ref in candidate.source_refs
                if ref in trend_by_id
            ),
            default=0.0,
        )
        seasonality = max(
            (
                seasonal_by_id[ref].relevance
                for ref in candidate.source_refs
                if ref in seasonal_by_id
            ),
            default=0.0,
        )

        return CandidateSignals(
            trend=trend,
            evergreen=1.0 if candidate.idea_source is IdeaSource.EVERGREEN else 0.0,
            character_fit=candidate.assessment.character_fit,
            novelty=candidate.assessment.novelty,
            visual_strength=candidate.assessment.visual_strength,
            historical_performance=sum(
                character.historical_performance for character in selected
            )
            / len(selected),
            seasonality=seasonality,
            series_potential=candidate.assessment.series_potential,
            readiness=min(character.readiness for character in selected),
            recent_character_penalty=max(
                character.recent_use_penalty for character in selected
            ),
        )


class ConceptScorer:
    def __init__(self, config: PlannerConfig) -> None:
        self._config = config

    def score(
        self,
        candidate: ConceptCandidate,
        signals: CandidateSignals,
    ) -> ScoredConcept:
        weights = self._config.score_weights
        positive = (
            signals.trend * weights.trend
            + signals.evergreen * weights.evergreen
            + signals.character_fit * weights.character_fit
            + signals.novelty * weights.novelty
            + signals.visual_strength * weights.visual_strength
            + signals.historical_performance * weights.historical_performance
            + signals.seasonality * weights.seasonality
            + signals.series_potential * weights.series_potential
            + signals.readiness * weights.readiness
        )
        aggregate = (
            positive
            - signals.similarity_penalty * self._config.similarity_penalty_weight
            - signals.recent_character_penalty
            * self._config.recent_character_penalty_weight
        )

        rejected_reason = None
        if signals.similarity_penalty >= self._config.hard_similarity_threshold:
            rejected_reason = "hard similarity threshold exceeded"

        score = CandidateScore(
            aggregate=aggregate,
            trend=signals.trend,
            evergreen=signals.evergreen,
            character_fit=signals.character_fit,
            novelty=signals.novelty,
            visual_strength=signals.visual_strength,
            historical_performance=signals.historical_performance,
            seasonality=signals.seasonality,
            series_potential=signals.series_potential,
            readiness=signals.readiness,
            similarity_penalty=signals.similarity_penalty,
            recent_character_penalty=signals.recent_character_penalty,
            rejected_reason=rejected_reason,
        )
        return ScoredConcept(candidate=candidate, score=score)
