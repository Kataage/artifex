from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from statistics import fmean

from artifex.config.models import PatreonPerformanceConfig
from artifex.db import Database
from artifex.db.models import ConceptRow, PackRow
from artifex.performance.models import (
    CandidatePerformance,
    PerformanceEffect,
    PerformanceEvidence,
    PerformanceMetrics,
)
from artifex.performance.repository import PerformanceRepository
from artifex.planner.models import ConceptCandidate


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _theme_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


class PerformanceLearningService:
    def __init__(
        self,
        database: Database,
        repository: PerformanceRepository,
        config: PatreonPerformanceConfig,
    ) -> None:
        self._database = database
        self._repository = repository
        self._config = config

    def character_effect(
        self,
        character_id: str,
        *,
        as_of: datetime | None = None,
    ) -> PerformanceEffect:
        return self._effect(
            "character",
            character_id,
            [
                item
                for item in self._evidence(as_of=as_of)
                if character_id in item.character_ids
            ],
        )

    def format_effect(
        self,
        format_key: str,
        *,
        as_of: datetime | None = None,
    ) -> PerformanceEffect:
        return self._effect(
            "format",
            format_key,
            [
                item
                for item in self._evidence(as_of=as_of)
                if item.format == format_key
            ],
        )

    def theme_effect(
        self,
        theme: str,
        *,
        as_of: datetime | None = None,
    ) -> PerformanceEffect:
        key = _theme_key(theme)
        return self._effect(
            "theme",
            key,
            [
                item
                for item in self._evidence(as_of=as_of)
                if item.theme is not None and _theme_key(item.theme) == key
            ],
        )

    def series_effect(
        self,
        series_id: str,
        *,
        as_of: datetime | None = None,
    ) -> PerformanceEffect:
        return self._effect(
            "series",
            series_id,
            [
                item
                for item in self._evidence(as_of=as_of)
                if item.series_id == series_id
            ],
        )

    def candidate_performance(
        self,
        candidate: ConceptCandidate,
        *,
        as_of: datetime | None = None,
    ) -> CandidatePerformance:
        character_effects = tuple(
            self.character_effect(character_id, as_of=as_of)
            for character_id in candidate.character_ids
        )
        effects = (
            *character_effects,
            self.format_effect(candidate.format.value, as_of=as_of),
            self.theme_effect(candidate.theme, as_of=as_of),
        )
        weighted: list[tuple[float, float]] = []
        if character_effects:
            char_score = fmean(effect.score for effect in character_effects)
            char_conf = fmean(effect.confidence for effect in character_effects)
            weighted.append((char_score, 0.50 * char_conf))
        format_effect = effects[-2]
        theme_effect = effects[-1]
        weighted.append((format_effect.score, 0.30 * format_effect.confidence))
        weighted.append((theme_effect.score, 0.20 * theme_effect.confidence))

        total_weight = sum(weight for _, weight in weighted)
        if total_weight <= 0:
            score = self._config.prior_score
            confidence = 0.0
        else:
            score = sum(score * weight for score, weight in weighted) / total_weight
            confidence = min(1.0, total_weight)

        reason = (
            f"performance score={score:.3f} confidence={confidence:.3f}; "
            + ", ".join(
                f"{effect.dimension}:{effect.key}="
                f"{effect.score:.3f}@{effect.confidence:.3f}"
                for effect in effects
            )
        )
        return CandidatePerformance(
            score=score,
            confidence=confidence,
            effects=tuple(effects),
            reason=reason,
        )

    def editorial_adjustment(
        self,
        candidate: ConceptCandidate,
        *,
        as_of: datetime | None = None,
    ) -> float:
        prediction = self.candidate_performance(candidate, as_of=as_of)
        return self._bounded_adjustment(
            prediction.score,
            prediction.confidence,
        )

    def series_editorial_adjustment(
        self,
        series_id: str,
        *,
        as_of: datetime | None = None,
    ) -> float:
        effect = self.series_effect(series_id, as_of=as_of)
        return self._bounded_adjustment(effect.score, effect.confidence)

    def strongest_effects(
        self,
        *,
        as_of: datetime | None = None,
        limit: int = 10,
    ) -> tuple[PerformanceEffect, ...]:
        evidence = self._evidence(as_of=as_of)
        keys: set[tuple[str, str]] = set()
        for item in evidence:
            keys.update(("character", key) for key in item.character_ids)
            if item.format:
                keys.add(("format", item.format))
            if item.theme:
                keys.add(("theme", _theme_key(item.theme)))
            if item.series_id:
                keys.add(("series", item.series_id))

        effects: list[PerformanceEffect] = []
        for dimension, key in sorted(keys):
            if dimension == "character":
                effects.append(self.character_effect(key, as_of=as_of))
            elif dimension == "format":
                effects.append(self.format_effect(key, as_of=as_of))
            elif dimension == "theme":
                effects.append(self.theme_effect(key, as_of=as_of))
            elif dimension == "series":
                effects.append(self.series_effect(key, as_of=as_of))
        effects.sort(
            key=lambda item: (
                -item.confidence,
                -abs(item.score - self._config.prior_score),
                item.dimension,
                item.key,
            )
        )
        return tuple(effects[: max(0, limit)])

    def _bounded_adjustment(self, score: float, confidence: float) -> float:
        if confidence < self._config.minimum_confidence_for_editorial:
            return 0.0
        centered = (score - self._config.prior_score) / 0.5
        centered = max(-1.0, min(1.0, centered))
        return (
            centered
            * self._config.editorial_effect_cap
            * confidence
        )

    def _effect(
        self,
        dimension: str,
        key: str,
        evidence: list[PerformanceEvidence],
    ) -> PerformanceEffect:
        prior = self._config.prior_score
        if not evidence:
            return PerformanceEffect(
                dimension=dimension,
                key=key,
                score=prior,
                confidence=0.0,
                evidence_count=0,
                effective_sample_size=0.0,
                reason="cold start: no persisted performance evidence",
            )

        weighted_raw = 0.0
        total_weight = 0.0
        effective_sample = 0.0
        coverage: set[str] = set()
        for item in evidence:
            weight = item.confidence * item.decay
            if weight <= 0:
                continue
            weighted_raw += item.raw_score * weight
            total_weight += weight
            effective_sample += self._sample_size(item.metrics) * item.decay
            coverage.update(item.component_scores)
        if total_weight <= 0:
            raw = prior
        else:
            raw = weighted_raw / total_weight

        confidence = 1.0 - math.exp(
            -effective_sample / self._config.confidence_sample_scale
        )
        score = prior + confidence * (raw - prior)
        score = max(0.0, min(1.0, score))
        return PerformanceEffect(
            dimension=dimension,
            key=key,
            score=score,
            confidence=confidence,
            evidence_count=len(evidence),
            effective_sample_size=effective_sample,
            metric_coverage=tuple(sorted(coverage)),
            reason=(
                f"{len(evidence)} latest publication snapshot(s); "
                f"effective_sample={effective_sample:.1f}; "
                f"raw={raw:.3f}; shrunk={score:.3f}"
            ),
        )

    def _evidence(
        self,
        *,
        as_of: datetime | None = None,
    ) -> list[PerformanceEvidence]:
        now = _utc(as_of or datetime.now(UTC))
        latest = self._repository.latest_snapshots(platform="patreon")
        result: list[PerformanceEvidence] = []
        with self._database.session() as session:
            for link, snapshot in latest:
                pack = session.get(PackRow, link.pack_id)
                if pack is None:
                    continue
                concept = (
                    session.get(ConceptRow, pack.concept_id)
                    if pack.concept_id is not None
                    else None
                )
                candidate: dict[str, object] = {}
                if concept is not None:
                    raw_candidate = concept.payload_json.get("candidate")
                    if isinstance(raw_candidate, dict):
                        candidate = raw_candidate

                raw_plan = pack.payload_json.get("plan")
                plan = raw_plan if isinstance(raw_plan, dict) else {}
                raw_ids = candidate.get(
                    "character_ids",
                    plan.get("character_ids", ()),
                )
                character_ids = (
                    tuple(str(value) for value in raw_ids)
                    if isinstance(raw_ids, list | tuple)
                    else ()
                )
                raw_format = candidate.get("format", plan.get("format"))
                format_key = (
                    str(raw_format)
                    if isinstance(raw_format, str)
                    else pack.format_type
                )
                raw_theme = candidate.get("theme")
                if not isinstance(raw_theme, str):
                    planning = pack.payload_json.get("planning_provenance")
                    if isinstance(planning, dict):
                        raw_theme = planning.get("theme")
                theme = str(raw_theme) if isinstance(raw_theme, str) else None
                raw_score, components = self._score_metrics(snapshot.metrics)
                sample = self._sample_size(snapshot.metrics)
                confidence = 1.0 - math.exp(
                    -sample / self._config.confidence_sample_scale
                )
                age_days = max(
                    0.0,
                    (now - _utc(snapshot.observed_at)).total_seconds() / 86400.0,
                )
                decay = math.exp(
                    -math.log(2.0) * age_days / self._config.half_life_days
                )
                result.append(
                    PerformanceEvidence(
                        publication_link_id=link.id,
                        pack_id=pack.id,
                        concept_id=pack.concept_id,
                        scene_id=link.scene_id,
                        series_id=pack.series_id,
                        character_ids=character_ids,
                        format=format_key,
                        theme=theme,
                        observed_at=snapshot.observed_at,
                        metrics=snapshot.metrics,
                        raw_score=raw_score,
                        confidence=confidence,
                        decay=decay,
                        component_scores=components,
                    )
                )
        return result

    def _score_metrics(
        self,
        metrics: PerformanceMetrics,
    ) -> tuple[float, dict[str, float]]:
        weights = self._config.metric_weights
        components: dict[str, float] = {}
        weighted: list[tuple[float, float]] = []

        if metrics.views is not None:
            score = 1.0 - math.exp(-metrics.views / self._config.view_scale)
            components["views"] = score
            weighted.append((score, weights.views))

        engagement = metrics.effective_engagement_count
        if engagement is not None and metrics.views and metrics.views > 0:
            rate = engagement / metrics.views
            score = min(1.0, rate / self._config.engagement_rate_scale)
            components["engagement"] = score
            weighted.append((score, weights.engagement))

        if metrics.paid_conversions is not None:
            denominator = None
            if metrics.free_signups is not None and metrics.free_signups > 0:
                denominator = metrics.free_signups
            elif metrics.views is not None and metrics.views > 0:
                denominator = metrics.views
            if denominator:
                rate = metrics.paid_conversions / denominator
                score = min(1.0, rate / self._config.conversion_rate_scale)
                components["conversion"] = score
                weighted.append((score, weights.conversion))

        if metrics.subscriber_delta is not None:
            magnitude = 1.0 - math.exp(
                -abs(metrics.subscriber_delta) / self._config.subscriber_scale
            )
            score = 0.5 + (0.5 * magnitude if metrics.subscriber_delta >= 0 else -0.5 * magnitude)
            components["subscribers"] = score
            weighted.append((score, weights.subscribers))

        if metrics.revenue_cents is not None:
            score = 1.0 - math.exp(
                -metrics.revenue_cents / self._config.revenue_scale_cents
            )
            components["revenue"] = score
            weighted.append((score, weights.revenue))

        if metrics.retention_rate is not None:
            components["retention"] = metrics.retention_rate
            weighted.append((metrics.retention_rate, weights.retention))

        active_weight = sum(weight for _, weight in weighted if weight > 0)
        if active_weight <= 0:
            return self._config.prior_score, components
        score = sum(value * weight for value, weight in weighted) / active_weight
        return max(0.0, min(1.0, score)), components

    @staticmethod
    def _sample_size(metrics: PerformanceMetrics) -> float:
        if metrics.views is not None:
            return float(metrics.views)
        signals = (
            metrics.effective_engagement_count or 0,
            metrics.free_signups or 0,
            metrics.paid_conversions or 0,
            abs(metrics.subscriber_delta or 0),
        )
        return float(max(1, sum(signals)))
