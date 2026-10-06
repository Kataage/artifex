from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.characters import CharacterRegistry
from artifex.config.models import (
    CharacterRegistryConfig,
    EditorialConfig,
    PatreonPerformanceConfig,
    PlannerConfig,
)
from artifex.db import Database
from artifex.db.models import ConceptRow, PackRow, SceneRow
from artifex.domain import CharacterProfile, LoRAPolicy, PackState, SceneState
from artifex.editorial import (
    EditorialRepository,
    EditorialService,
    PackInventoryRepository,
)
from artifex.loras import LoRARegistry
from artifex.performance import (
    PerformanceLearningService,
    PerformanceMetrics,
    PerformanceRepository,
    ingest_manual_performance,
    load_manual_performance,
)
from artifex.performance.provider import PerformanceAwareSignalProvider
from artifex.planner import ConceptRepository
from artifex.planner.models import (
    ConceptCandidate,
    CreativeAssessment,
    IdeaSource,
    PackFormat,
    PlanningContext,
)
from artifex.planner.scoring import (
    CandidateSignals,
    ConceptScorer,
    DefaultSignalProvider,
)
from artifex.production.context import PlanningContextBuilder
from artifex.series import SeriesRepository


def _candidate(
    *,
    character_id: str = "char-a",
    format: PackFormat = PackFormat.EVERGREEN,
    theme: str = "angel cafe",
) -> ConceptCandidate:
    return ConceptCandidate(
        candidate_key=f"{character_id}-{format.value}-{theme}",
        character_ids=(character_id,),
        idea_source=IdeaSource.EVERGREEN,
        format=format,
        theme=theme,
        setting="cafe",
        mood="warm",
        visual_hook="window light",
        progression="single scene",
        target_scene_count=1,
        assessment=CreativeAssessment(
            character_fit=0.9,
            novelty=0.8,
            visual_strength=0.8,
            series_potential=0.4,
        ),
    )


def _seed_pack(
    database: Database,
    candidate: ConceptCandidate,
    *,
    suffix: str = "a",
    series_id: str | None = None,
) -> tuple[str, str]:
    now = datetime.now(UTC)
    concept_id = f"concept-{suffix}"
    pack_id = f"pack-{suffix}"
    scene_id = f"scene-{suffix}"
    with database.session() as session:
        session.add(
            ConceptRow(
                id=concept_id,
                status="used",
                payload_json={
                    "candidate": candidate.model_dump(mode="json"),
                    "score": {},
                    "selected": True,
                },
                score=0.7,
                similarity_score=0.0,
                created_at=now,
            )
        )
        session.add(
            PackRow(
                id=pack_id,
                concept_id=concept_id,
                series_id=series_id,
                state=PackState.FINALIZED.value,
                format_type=candidate.format.value,
                payload_json={
                    "plan": {
                        "character_ids": list(candidate.character_ids),
                        "format": candidate.format.value,
                    },
                    "planning_provenance": {"theme": candidate.theme},
                },
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            SceneRow(
                id=scene_id,
                pack_id=pack_id,
                ordinal=1,
                state=SceneState.ACCEPTED.value,
                publication_tier="member",
                payload_json={},
            )
        )
    return pack_id, scene_id


def _service(
    database: Database,
    *,
    config: PatreonPerformanceConfig | None = None,
) -> tuple[PerformanceRepository, PerformanceLearningService]:
    repository = PerformanceRepository(database)
    return repository, PerformanceLearningService(
        database,
        repository,
        config
        or PatreonPerformanceConfig(
            confidence_sample_scale=50,
            view_scale=500,
            revenue_scale_cents=1000,
        ),
    )


def _add_evidence(
    repository: PerformanceRepository,
    *,
    pack_id: str,
    scene_id: str,
    post_id: str,
    observed_at: datetime,
    metrics: PerformanceMetrics,
) -> None:
    link = repository.upsert_publication(
        platform="patreon",
        external_post_id=post_id,
        pack_id=pack_id,
        scene_id=scene_id,
        publication_tier="member",
        url=f"https://patreon.example/posts/{post_id}",
        published_at=observed_at - timedelta(days=1),
        source="test",
    )
    repository.add_snapshot(
        link.id,
        observed_at=observed_at,
        metrics=metrics,
        source="test",
    )


def test_cold_start_stays_at_prior_with_zero_confidence(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'cold.sqlite3').as_posix()}")
    database.migrate()
    _, learning = _service(database)

    effect = learning.character_effect("char-a")

    assert effect.score == pytest.approx(0.5)
    assert effect.confidence == 0.0
    assert effect.evidence_count == 0
    assert "cold start" in effect.reason
    database.dispose()


def test_learning_uses_latest_snapshot_per_publication(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'latest.sqlite3').as_posix()}")
    database.migrate()
    candidate = _candidate()
    pack_id, scene_id = _seed_pack(database, candidate)
    repository, learning = _service(database)
    now = datetime.now(UTC)
    link = repository.upsert_publication(
        platform="patreon",
        external_post_id="post-1",
        pack_id=pack_id,
        scene_id=scene_id,
        publication_tier="member",
        source="test",
    )
    repository.add_snapshot(
        link.id,
        observed_at=now - timedelta(days=1),
        metrics=PerformanceMetrics(
            views=1000,
            engagement_count=150,
            paid_conversions=50,
            revenue_cents=5000,
            retention_rate=0.95,
        ),
        source="test",
    )
    repository.add_snapshot(
        link.id,
        observed_at=now,
        metrics=PerformanceMetrics(
            views=1000,
            engagement_count=0,
            paid_conversions=0,
            subscriber_delta=-20,
            revenue_cents=0,
            retention_rate=0.10,
        ),
        source="test",
    )

    effect = learning.character_effect("char-a", as_of=now)

    assert len(repository.snapshots_for_publication(link.id)) == 2
    assert effect.evidence_count == 1
    assert effect.confidence > 0.9
    assert effect.score < 0.5
    database.dispose()


def test_candidate_prediction_combines_character_format_and_theme(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'candidate.sqlite3').as_posix()}")
    database.migrate()
    candidate = _candidate()
    pack_id, scene_id = _seed_pack(database, candidate)
    repository, learning = _service(database)
    now = datetime.now(UTC)
    _add_evidence(
        repository,
        pack_id=pack_id,
        scene_id=scene_id,
        post_id="post-good",
        observed_at=now,
        metrics=PerformanceMetrics(
            views=2000,
            engagement_count=300,
            free_signups=100,
            paid_conversions=20,
            subscriber_delta=30,
            revenue_cents=8000,
            retention_rate=0.9,
        ),
    )

    prediction = learning.candidate_performance(candidate, as_of=now)

    assert prediction.score > 0.8
    assert prediction.confidence > 0.9
    assert {effect.dimension for effect in prediction.effects} == {
        "character",
        "format",
        "theme",
    }
    assert "confidence=" in prediction.reason
    assert abs(learning.editorial_adjustment(candidate, as_of=now)) <= 0.15
    database.dispose()


def test_time_decay_reduces_effective_confidence(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'decay.sqlite3').as_posix()}")
    database.migrate()
    fresh_candidate = _candidate(character_id="fresh")
    old_candidate = _candidate(character_id="old")
    fresh_pack, fresh_scene = _seed_pack(
        database,
        fresh_candidate,
        suffix="fresh",
    )
    old_pack, old_scene = _seed_pack(database, old_candidate, suffix="old")
    repository, learning = _service(
        database,
        config=PatreonPerformanceConfig(
            half_life_days=10,
            confidence_sample_scale=100,
            view_scale=500,
        ),
    )
    now = datetime.now(UTC)
    metrics = PerformanceMetrics(
        views=500,
        engagement_count=50,
        revenue_cents=1000,
        retention_rate=0.8,
    )
    _add_evidence(
        repository,
        pack_id=fresh_pack,
        scene_id=fresh_scene,
        post_id="fresh-post",
        observed_at=now,
        metrics=metrics,
    )
    _add_evidence(
        repository,
        pack_id=old_pack,
        scene_id=old_scene,
        post_id="old-post",
        observed_at=now - timedelta(days=40),
        metrics=metrics,
    )

    fresh = learning.character_effect("fresh", as_of=now)
    old = learning.character_effect("old", as_of=now)

    assert fresh.confidence > old.confidence
    assert fresh.effective_sample_size > old.effective_sample_size
    assert abs(fresh.score - 0.5) > abs(old.score - 0.5)
    database.dispose()


def test_manual_import_maps_post_back_to_pack_and_scene(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'import.sqlite3').as_posix()}")
    database.migrate()
    pack_id, scene_id = _seed_pack(database, _candidate())
    path = tmp_path / "performance.json"
    path.write_text(
        json.dumps(
            {
                "platform": "patreon",
                "source": "manual-export",
                "records": [
                    {
                        "external_post_id": "post-import",
                        "pack_id": pack_id,
                        "scene_id": scene_id,
                        "publication_tier": "member",
                        "observed_at": datetime.now(UTC).isoformat(),
                        "views": 100,
                        "engagement_count": 20,
                        "paid_conversions": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    repository = PerformanceRepository(database)

    publications, snapshots = ingest_manual_performance(
        repository,
        load_manual_performance(path),
        import_path=path,
    )
    link = repository.get_publication("patreon", "post-import")

    assert publications == 1
    assert snapshots == 1
    assert link is not None
    assert link.pack_id == pack_id
    assert link.scene_id == scene_id
    assert len(repository.snapshots_for_publication(link.id)) == 1
    database.dispose()


def test_planning_context_uses_real_persisted_character_performance(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'context.sqlite3').as_posix()}")
    database.migrate()
    candidate = _candidate()
    pack_id, scene_id = _seed_pack(database, candidate)
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="Character A",
            namespace="test",
            lora_policy=LoRAPolicy.NONE,
            readiness=1.0,
        )
    )
    repository, learning = _service(database)
    now = datetime.now(UTC)
    _add_evidence(
        repository,
        pack_id=pack_id,
        scene_id=scene_id,
        post_id="post-context",
        observed_at=now,
        metrics=PerformanceMetrics(
            views=1500,
            engagement_count=200,
            revenue_cents=5000,
            retention_rate=0.9,
        ),
    )

    context = PlanningContextBuilder(
        characters,
        LoRARegistry(database),
        ConceptRepository(database),
        CharacterRegistryConfig(),
        performance=learning,
    ).build(as_of=now)

    option = context.characters[0]
    assert option.historical_performance > 0.5
    assert option.historical_performance_confidence > 0.9
    assert "latest publication snapshot" in option.historical_performance_reason
    database.dispose()


@pytest.mark.asyncio
async def test_performance_provider_overrides_character_only_baseline(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'provider.sqlite3').as_posix()}")
    database.migrate()
    candidate = _candidate()
    pack_id, scene_id = _seed_pack(database, candidate)
    repository, learning = _service(database)
    now = datetime.now(UTC)
    _add_evidence(
        repository,
        pack_id=pack_id,
        scene_id=scene_id,
        post_id="post-provider",
        observed_at=now,
        metrics=PerformanceMetrics(
            views=2000,
            engagement_count=300,
            paid_conversions=50,
            revenue_cents=10000,
            retention_rate=0.95,
        ),
    )
    context = PlanningContext(
        as_of=now,
        characters=(
            {
                "id": "char-a",
                "display_name": "Character A",
                "historical_performance": 0.5,
            },
        ),
    )

    signals = await PerformanceAwareSignalProvider(
        DefaultSignalProvider(),
        learning,
    ).evaluate(candidate, context)

    assert signals.historical_performance > 0.8
    assert signals.historical_performance_confidence > 0.9
    assert "format:evergreen" in signals.historical_performance_reason
    database.dispose()


def test_high_performance_cannot_override_hard_similarity_rejection() -> None:
    candidate = _candidate()
    signals = CandidateSignals(
        character_fit=1.0,
        novelty=1.0,
        visual_strength=1.0,
        historical_performance=1.0,
        historical_performance_confidence=1.0,
        historical_performance_reason="strong evidence",
        readiness=1.0,
        similarity_penalty=1.0,
    )

    scored = ConceptScorer(PlannerConfig()).score(candidate, signals)

    assert scored.score.historical_performance == 1.0
    assert scored.score.historical_performance_confidence == 1.0
    assert scored.score.rejected_reason == "hard similarity threshold exceeded"



def test_views_only_metric_stays_near_neutral_prior(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'views-only.sqlite3').as_posix()}")
    database.migrate()
    candidate = _candidate()
    pack_id, scene_id = _seed_pack(database, candidate)
    repository, learning = _service(
        database,
        config=PatreonPerformanceConfig(
            confidence_sample_scale=10,
            view_scale=100,
        ),
    )
    now = datetime.now(UTC)
    _add_evidence(
        repository,
        pack_id=pack_id,
        scene_id=scene_id,
        post_id="views-only",
        observed_at=now,
        metrics=PerformanceMetrics(views=100000),
    )

    effect = learning.character_effect("char-a", as_of=now)

    assert effect.confidence > 0.99
    assert effect.metric_coverage == ("views",)
    assert 0.5 < effect.score < 0.55
    database.dispose()


def test_manual_import_rejects_non_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "invalid-performance.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "external_post_id": "valid",
                        "pack_id": "pack-1",
                        "observed_at": datetime.now(UTC).isoformat(),
                        "views": 10,
                    },
                    "not-an-object",
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="item 2 must be an object"):
        load_manual_performance(path)



def test_high_performance_cannot_override_editorial_character_cooldown(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'editorial-learning.sqlite3').as_posix()}")
    database.migrate()
    now = datetime.now(UTC)
    hot_candidate = _candidate(character_id="char-a", theme="hot repeat")
    fresh_candidate = _candidate(character_id="char-b", theme="fresh choice")
    hot_pack, hot_scene = _seed_pack(
        database,
        hot_candidate,
        suffix="hot-history",
    )
    repository, learning = _service(
        database,
        config=PatreonPerformanceConfig(
            confidence_sample_scale=10,
            view_scale=100,
            revenue_scale_cents=100,
        ),
    )
    _add_evidence(
        repository,
        pack_id=hot_pack,
        scene_id=hot_scene,
        post_id="hot-post",
        observed_at=now,
        metrics=PerformanceMetrics(
            views=10000,
            engagement_count=2000,
            free_signups=500,
            paid_conversions=100,
            subscriber_delta=100,
            revenue_cents=10000,
            retention_rate=1.0,
        ),
    )
    with database.session() as session:
        session.add_all(
            (
                ConceptRow(
                    id="idea-hot",
                    status="idea",
                    payload_json={
                        "candidate": hot_candidate.model_dump(mode="json"),
                    },
                    score=0.99,
                    similarity_score=0.0,
                    created_at=now + timedelta(seconds=1),
                ),
                ConceptRow(
                    id="idea-fresh",
                    status="idea",
                    payload_json={
                        "candidate": fresh_candidate.model_dump(mode="json"),
                    },
                    score=0.60,
                    similarity_score=0.0,
                    created_at=now + timedelta(seconds=2),
                ),
            )
        )

    config = EditorialConfig(
        mode="continuous",
        character_cooldown_packs=2,
        diversity_window_packs=10,
        series_target_share=0.0,
    )
    editorial = EditorialService(
        database,
        config,
        SeriesRepository(database),
        PackInventoryRepository(
            database,
            expiry_hours=config.inventory_expiry_hours,
        ),
        EditorialRepository(database),
        performance=learning,
    )

    decision = editorial.decide_plan(has_ideas=True)

    assert decision.concept_id == "idea-fresh"
    assert learning.candidate_performance(
        hot_candidate,
        as_of=now,
    ).score > 0.8
    database.dispose()
