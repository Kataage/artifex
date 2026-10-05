from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from artifex.config.models import PlannerConfig, PlannerMixConfig
from artifex.db import Database
from artifex.db.models import ConceptRow
from artifex.llm import ChatMessage, StructuredGenerator
from artifex.planner import (
    CharacterOption,
    ConceptRepository,
    IdeaDirector,
    PlanningContext,
    SeasonalEventSummary,
    TrendSignalSummary,
)
from artifex.planner.allocation import source_quotas
from artifex.planner.models import IdeaSource


class ScriptedClient:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        del messages, response_schema, schema_name
        output = self.outputs[self.calls]
        self.calls += 1
        return output


def _context() -> PlanningContext:
    return PlanningContext(
        as_of=datetime(2026, 10, 5, tzinfo=UTC),
        characters=(
            CharacterOption(
                id="char-a",
                display_name="Character A",
                readiness=0.95,
                historical_performance=0.7,
            ),
        ),
        trend_signals=(
            TrendSignalSummary(
                id="trend-1",
                topic="night city",
                strength=0.9,
                freshness=1.0,
            ),
        ),
        seasonal_events=(
            SeasonalEventSummary(id="season-1", title="autumn", relevance=0.8),
        ),
        evergreen_prompts=("portrait", "casual day"),
    )


def _candidate(
    key: str,
    source: str,
    *,
    refs: list[str] | None = None,
    novelty: float = 0.5,
    visual: float = 0.5,
) -> dict[str, Any]:
    return {
        "candidate_key": key,
        "character_ids": ["char-a"],
        "idea_source": source,
        "format": "single_feature",
        "theme": f"theme {key}",
        "setting": f"setting {key}",
        "mood": "calm",
        "visual_hook": f"visual hook {key}",
        "progression": "A coherent multi-scene visual progression.",
        "target_scene_count": 4,
        "continuity_requirements": ["same outfit"],
        "source_refs": refs or [],
        "assessment": {
            "character_fit": 0.9,
            "novelty": novelty,
            "visual_strength": visual,
            "series_potential": 0.5,
        },
    }


def _valid_batch() -> str:
    return json.dumps(
        {
            "candidates": [
                _candidate("evergreen", "evergreen", visual=0.6),
                _candidate("trend", "trend", refs=["trend-1"], novelty=0.8, visual=0.9),
                _candidate("seasonal", "seasonal", refs=["season-1"], visual=0.7),
                _candidate("explore", "exploration", novelty=1.0, visual=0.8),
            ]
        }
    )


def test_source_quota_redistributes_unavailable_sources() -> None:
    context = PlanningContext(
        as_of=datetime(2026, 10, 5, tzinfo=UTC),
        characters=(CharacterOption(id="char-a", display_name="A"),),
    )
    quotas = source_quotas(8, PlannerMixConfig(), context)

    assert quotas == {
        IdeaSource.EVERGREEN: 6,
        IdeaSource.EXPLORATION: 2,
    }


@pytest.mark.asyncio
async def test_director_repairs_quota_error_scores_and_persists(tmp_path: Path) -> None:
    invalid = json.dumps(
        {
            "candidates": [
                _candidate("a", "evergreen"),
                _candidate("b", "evergreen"),
                _candidate("c", "evergreen"),
            ]
        }
    )
    client = ScriptedClient([invalid, _valid_batch()])
    database = Database(f"sqlite:///{(tmp_path / 'planner.sqlite3').as_posix()}")
    database.migrate()

    ids = iter(["id-1", "id-2", "id-3", "id-4"])
    repository = ConceptRepository(database, id_factory=lambda: next(ids))
    director = IdeaDirector(
        StructuredGenerator(client, repair_attempts=1),
        repository,
        PlannerConfig(
            candidate_count=4,
            mix=PlannerMixConfig(
                evergreen=0.25,
                trend=0.25,
                seasonal=0.25,
                exploration=0.25,
            ),
        ),
    )

    selected = await director.create_concept(_context())

    assert client.calls == 2
    assert selected.candidate.candidate_key == "trend"

    with database.session() as session:
        rows = session.scalars(select(ConceptRow).order_by(ConceptRow.id)).all()
    assert len(rows) == 4
    assert sum(row.status == "idea" for row in rows) == 1
    assert next(row for row in rows if row.status == "idea").id == selected.concept_id
    database.dispose()


@pytest.mark.asyncio
async def test_director_rejects_invented_trend_reference(tmp_path: Path) -> None:
    bad_batch = json.dumps(
        {
            "candidates": [
                _candidate("evergreen", "evergreen"),
                _candidate("trend", "trend", refs=["invented-trend"]),
                _candidate("seasonal", "seasonal", refs=["season-1"]),
                _candidate("explore", "exploration"),
            ]
        }
    )
    client = ScriptedClient([bad_batch])
    database = Database(f"sqlite:///{(tmp_path / 'invalid.sqlite3').as_posix()}")
    database.migrate()
    director = IdeaDirector(
        StructuredGenerator(client, repair_attempts=0),
        ConceptRepository(database),
        PlannerConfig(
            candidate_count=4,
            mix=PlannerMixConfig(
                evergreen=0.25,
                trend=0.25,
                seasonal=0.25,
                exploration=0.25,
            ),
        ),
    )

    with pytest.raises(RuntimeError):
        await director.create_concept(_context())

    database.dispose()



@pytest.mark.asyncio
async def test_director_rejects_invented_seasonal_reference(
    tmp_path: Path,
) -> None:
    bad_batch = json.dumps(
        {
            "candidates": [
                _candidate("evergreen", "evergreen"),
                _candidate("trend", "trend", refs=["trend-1"]),
                _candidate("seasonal", "seasonal", refs=["invented-season"]),
                _candidate("explore", "exploration"),
            ]
        }
    )
    client = ScriptedClient([bad_batch])
    database = Database(
        f"sqlite:///{(tmp_path / 'invalid-seasonal.sqlite3').as_posix()}"
    )
    database.migrate()
    director = IdeaDirector(
        StructuredGenerator(client, repair_attempts=0),
        ConceptRepository(database),
        PlannerConfig(
            candidate_count=4,
            mix=PlannerMixConfig(
                evergreen=0.25,
                trend=0.25,
                seasonal=0.25,
                exploration=0.25,
            ),
        ),
    )

    with pytest.raises(RuntimeError):
        await director.create_concept(_context())

    database.dispose()
