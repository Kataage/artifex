from __future__ import annotations

from datetime import UTC

from artifex.config.models import ResearchConfig
from artifex.planner.models import (
    PlanningContext,
    ResearchBriefSummary,
    ResearchEvidenceSummary,
)
from artifex.research.models import (
    ResearchIntent,
    ResearchSearchRequest,
    SafeSearch,
    SearchSource,
)
from artifex.research.service import ResearchService


class ResearchDirector:
    def __init__(
        self,
        service: ResearchService,
        config: ResearchConfig,
    ) -> None:
        self._service = service
        self._config = config

    async def prepare(
        self,
        context: PlanningContext,
        *,
        adult: bool = False,
        topic_hint: str | None = None,
    ) -> PlanningContext:
        if not self._config.enabled:
            if self._config.required_for_ideation:
                raise RuntimeError("research is required for ideation but disabled")
            return context

        now = context.as_of
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        existing = context.research_brief
        if (
            existing is not None
            and existing.adult == adult
            and existing.fresh_at(now)
        ):
            return context

        requests = self._query_plan(
            context,
            adult=adult,
            topic_hint=topic_hint,
        )
        names = ", ".join(
            character.display_name
            for character in sorted(
                context.characters,
                key=lambda item: (
                    item.recent_use_penalty,
                    -item.readiness,
                    item.id,
                ),
            )[:3]
        )
        topic = (
            f"illustration ideation for {topic_hint.strip()}"
            if topic_hint is not None and topic_hint.strip()
            else f"illustration ideation for {names or 'configured characters'}"
        )
        brief = await self._service.brief(
            topic,
            requests,
            adult=adult,
            now=now,
        )
        return context.model_copy(
            update={
                "research_brief": ResearchBriefSummary(
                    id=brief.id,
                    topic=brief.topic,
                    research_run_ids=brief.research_run_ids,
                    evidence_ids=brief.evidence_ids,
                    key_findings=brief.key_findings,
                    items=tuple(
                        ResearchEvidenceSummary(
                            evidence_id=item.evidence_id,
                            source=item.source.value,
                            provider=item.provider,
                            title=item.title,
                            finding=item.finding,
                        )
                        for item in brief.items
                    ),
                    generated_at=brief.generated_at,
                    expires_at=brief.expires_at,
                    freshness_confidence=brief.freshness_confidence,
                    degraded=brief.degraded,
                    adult=brief.adult,
                )
            }
        )

    def _query_plan(
        self,
        context: PlanningContext,
        *,
        adult: bool,
        topic_hint: str | None = None,
    ) -> tuple[ResearchSearchRequest, ...]:
        ranked = sorted(
            context.characters,
            key=lambda item: (
                item.recent_use_penalty,
                -item.readiness,
                item.id,
            ),
        )
        names = " ".join(item.display_name for item in ranked[:3])
        focus = (
            topic_hint.strip()
            if topic_hint is not None and topic_hint.strip()
            else names
        )
        month = context.as_of.strftime("%B")
        default_safe = SafeSearch(self._config.default_safesearch)
        base: list[ResearchSearchRequest] = [
            ResearchSearchRequest(
                query=f"{focus} fanart illustration ideas trends".strip(),
                source=SearchSource.WEB,
                intent=ResearchIntent.CURRENT,
                max_results=self._config.max_results_per_query,
                region=self._config.default_region,
                safesearch=default_safe,
                timelimit="m",
            ),
            ResearchSearchRequest(
                query=f"{focus} anime illustration composition".strip(),
                source=SearchSource.IMAGES,
                intent=ResearchIntent.COMPOSITION,
                max_results=self._config.max_results_per_query,
                region=self._config.default_region,
                safesearch=default_safe,
                timelimit="m",
            ),
            ResearchSearchRequest(
                query=f"{month} anime illustration seasonal themes",
                source=SearchSource.WEB,
                intent=ResearchIntent.SEASONAL,
                max_results=self._config.max_results_per_query,
                region=self._config.default_region,
                safesearch=default_safe,
                timelimit="m",
            ),
            ResearchSearchRequest(
                query=f"{focus} illustration outfit setting ideas".strip(),
                source=SearchSource.WEB,
                intent=ResearchIntent.EVERGREEN,
                max_results=self._config.max_results_per_query,
                region=self._config.default_region,
                safesearch=default_safe,
            ),
        ]
        if adult:
            base[-1] = ResearchSearchRequest(
                query=f"{focus} adult illustration tags".strip(),
                source=SearchSource.TAGS,
                intent=ResearchIntent.ADULT,
                max_results=self._config.max_results_per_query,
                region=self._config.default_region,
                safesearch=SafeSearch(self._config.adult_safesearch),
                adult=True,
            )
        return tuple(base[: self._config.max_queries_per_cycle])
