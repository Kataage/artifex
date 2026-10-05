from __future__ import annotations

import json

from artifex.config.models import EditorialConfig, PlannerConfig
from artifex.editorial.models import SeriesPlanKind
from artifex.llm import ChatMessage, StructuredGenerator
from artifex.llm.prompts import SERIES_IDEA_PROMPT
from artifex.planner.models import (
    ConceptCandidateBatch,
    IdeaSource,
    PackFormat,
    PlanningContext,
    ScoredConcept,
    SelectedConcept,
)
from artifex.planner.repository import ConceptRepository
from artifex.planner.scoring import ConceptScorer, DefaultSignalProvider, SignalProvider
from artifex.series import SeriesProfile


class SeriesIdeaDirector:
    """Create a research-grounded next-episode concept for one active Series."""

    def __init__(
        self,
        generator: StructuredGenerator,
        repository: ConceptRepository,
        planner_config: PlannerConfig,
        editorial_config: EditorialConfig,
        *,
        signal_provider: SignalProvider | None = None,
        require_research: bool = False,
    ) -> None:
        self._generator = generator
        self._repository = repository
        self._scorer = ConceptScorer(planner_config)
        self._signals = signal_provider or DefaultSignalProvider()
        self._candidate_count = editorial_config.series_candidate_count
        self._require_research = require_research

    async def create_concept(
        self,
        context: PlanningContext,
        series: SeriesProfile,
        *,
        plan_kind: SeriesPlanKind,
    ) -> SelectedConcept:
        self._validate_research(context)
        required_format = _series_format(series)

        def validate(batch: ConceptCandidateBatch) -> None:
            if len(batch.candidates) != self._candidate_count:
                raise ValueError(
                    f"expected {self._candidate_count} Series candidates, "
                    f"got {len(batch.candidates)}"
                )
            allowed_runs = (
                set(context.research_brief.research_run_ids)
                if context.research_brief is not None
                else set()
            )
            allowed_evidence = (
                set(context.research_brief.evidence_ids)
                if context.research_brief is not None
                else set()
            )
            for candidate in batch.candidates:
                if candidate.idea_source is not IdeaSource.SERIES:
                    raise ValueError("Series candidate must use idea_source=series")
                if candidate.character_ids != series.character_ids:
                    raise ValueError(
                        "Series candidate character_ids must exactly match Series"
                    )
                if candidate.format is not required_format:
                    raise ValueError(
                        "Series candidate format must match the required Series format"
                    )
                if not set(candidate.research_run_ids) <= allowed_runs:
                    raise ValueError("Series candidate references unknown research runs")
                if not set(candidate.research_evidence_ids) <= allowed_evidence:
                    raise ValueError(
                        "Series candidate references unknown research evidence"
                    )
                if self._require_research:
                    if allowed_runs and not candidate.research_run_ids:
                        raise ValueError("Series candidate must cite research run ids")
                    if allowed_evidence and not candidate.research_evidence_ids:
                        raise ValueError("Series candidate must cite research evidence")

        batch = await self._generator.generate(
            ConceptCandidateBatch,
            self._messages(
                context,
                series,
                plan_kind=plan_kind,
                required_format=required_format,
            ),
            schema_name=SERIES_IDEA_PROMPT.schema_name(
                "artifex_series_concept_candidates"
            ),
            post_validator=validate,
        )

        scored: list[ScoredConcept] = []
        for candidate in batch.candidates:
            signals = await self._signals.evaluate(candidate, context)
            scored.append(self._scorer.score(candidate, signals))

        viable = [item for item in scored if item.score.rejected_reason is None]
        if not viable:
            raise RuntimeError("all Series concept candidates were rejected")
        selected = max(
            viable,
            key=lambda item: (
                item.score.aggregate,
                item.candidate.candidate_key,
            ),
        )
        return self._repository.persist_selection(
            scored,
            selected_key=selected.candidate.candidate_key,
        )

    def _validate_research(self, context: PlanningContext) -> None:
        if not self._require_research:
            return
        brief = context.research_brief
        if brief is None:
            raise RuntimeError(
                "Series ideation requires a ResearchBrief before planning"
            )
        if not brief.fresh_at(context.as_of):
            raise RuntimeError("Series ResearchBrief is stale and must be refreshed")

    def _messages(
        self,
        context: PlanningContext,
        series: SeriesProfile,
        *,
        plan_kind: SeriesPlanKind,
        required_format: PackFormat,
    ) -> tuple[ChatMessage, ...]:
        return (
            ChatMessage(role="system", content=SERIES_IDEA_PROMPT.text),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "candidate_count": self._candidate_count,
                        "plan_kind": plan_kind.value,
                        "required_idea_source": IdeaSource.SERIES.value,
                        "required_format": required_format.value,
                        "required_character_ids": list(series.character_ids),
                        "series": {
                            "id": series.id,
                            "title": series.title,
                            "current_episode": series.current_episode,
                            "bible": list(series.bible),
                            "rolling_summary": series.rolling_summary,
                            "recent_episode_summaries": list(
                                series.recent_episode_summaries[-5:]
                            ),
                            "continuity_state": series.continuity_state,
                            "unresolved_hooks": list(series.unresolved_hooks),
                        },
                        "planning_context": context.model_dump(mode="json"),
                        "requirements": {
                            "episode": (
                                "continue existing continuity but make the visual "
                                "hook distinct from recent episodes"
                            ),
                            "research_grounding": (
                                "cite exact research_run_ids and research_evidence_ids "
                                "when supplied"
                            ),
                            "bonus": (
                                "when plan_kind=bonus, make the episode feel like a "
                                "special visual variation without breaking continuity"
                            ),
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        )


def _series_format(series: SeriesProfile) -> PackFormat:
    if series.preferred_format is not None:
        required = series.preferred_format
    elif len(series.character_ids) == 1:
        required = PackFormat.CONTINUATION
    elif len(series.character_ids) == 2:
        required = PackFormat.DUO
    else:
        required = PackFormat.GROUP

    count = len(series.character_ids)
    if required is PackFormat.DUO and count != 2:
        raise ValueError("Series DUO format requires exactly two characters")
    if required is PackFormat.GROUP and count < 3:
        raise ValueError("Series GROUP format requires at least three characters")
    if required not in {PackFormat.DUO, PackFormat.GROUP} and count != 1:
        raise ValueError(
            "multi-character Series requires DUO or GROUP preferred_format"
        )
    return required
