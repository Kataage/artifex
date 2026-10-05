from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from artifex.config.models import EditorialConfig, ProductionConfig
from artifex.db import Database
from artifex.db.models import CharacterRow, ConceptRow, PackRow
from artifex.domain import AgentState, PackState
from artifex.editorial import (
    EditorialRepository,
    EditorialService,
    PackInventoryRepository,
    PackInventoryState,
)
from artifex.runtime import RuntimeStore
from artifex.scheduler import Scheduler, SchedulerAction
from artifex.series import SeriesEditorialPolicy, SeriesRepository


def _database(tmp_path: Path) -> Database:
    db = Database(f"sqlite:///{(tmp_path / 'editorial.sqlite3').as_posix()}")
    db.migrate()
    return db


def _running_runtime(db: Database) -> RuntimeStore:
    runtime = RuntimeStore(db)
    runtime.set_agent_state(AgentState.STARTING, expected=AgentState.STOPPED)
    runtime.set_agent_state(AgentState.RUNNING, expected=AgentState.STARTING)
    return runtime


def _pack(
    pack_id: str,
    *,
    when: datetime,
    state: PackState = PackState.FINALIZED,
    series_id: str | None = None,
    character_ids: tuple[str, ...] = ("char-a",),
    format_type: str = "single_feature",
    theme: str = "theme-a",
) -> PackRow:
    return PackRow(
        id=pack_id,
        series_id=series_id,
        state=state.value,
        format_type=format_type,
        payload_json={
            "plan": {
                "format": format_type,
                "character_ids": list(character_ids),
            },
            "planning_provenance": {"theme": theme},
        },
        checkpoint_json={},
        created_at=when,
        updated_at=when,
    )


def _editorial(
    db: Database,
    config: EditorialConfig,
    *,
    series: SeriesRepository | None = None,
) -> EditorialService:
    series_repo = series or SeriesRepository(db)
    return EditorialService(
        db,
        config,
        series_repo,
        PackInventoryRepository(
            db,
            expiry_hours=config.inventory_expiry_hours,
        ),
        EditorialRepository(db),
    )


def test_inventory_watermark_resumes_after_consumption(tmp_path: Path) -> None:
    db = _database(tmp_path)
    now = datetime.now(UTC)
    with db.session() as session:
        session.add_all(
            (
                _pack("pack-1", when=now),
                _pack("pack-2", when=now),
            )
        )

    editorial = _editorial(
        db,
        EditorialConfig(
            mode="watermark",
            inventory_low_watermark=1,
            inventory_high_watermark=2,
            inventory_expiry_hours=168,
        ),
    )

    allowed, _ = editorial.production_allowed()
    assert allowed is False
    assert editorial.inventory_counts().available == 2

    editorial.consume_pack("pack-1")
    allowed, reason = editorial.production_allowed()

    assert allowed is True
    assert "refill window" in reason
    counts = editorial.inventory_counts()
    assert counts.available == 1
    assert counts.consumed == 1
    db.dispose()


def test_old_unconsumed_inventory_expires_instead_of_stopping_forever(
    tmp_path: Path,
) -> None:
    db = _database(tmp_path)
    old = datetime.now(UTC) - timedelta(hours=4)
    with db.session() as session:
        session.add(_pack("stale-pack", when=old))

    editorial = _editorial(
        db,
        EditorialConfig(
            inventory_low_watermark=0,
            inventory_high_watermark=1,
            inventory_expiry_hours=1,
        ),
    )
    counts = editorial.inventory_counts()
    allowed, _ = editorial.production_allowed()

    assert counts.available == 0
    assert counts.expired == 1
    assert allowed is True
    db.dispose()


def test_inventory_reserve_release_consume_is_strict(tmp_path: Path) -> None:
    db = _database(tmp_path)
    now = datetime.now(UTC)
    with db.session() as session:
        session.add(_pack("pack-1", when=now))

    inventory = PackInventoryRepository(db, expiry_hours=24)
    inventory.sync_finalized(now=now)

    assert inventory.reserve("pack-1").state is PackInventoryState.RESERVED
    assert inventory.release("pack-1").state is PackInventoryState.AVAILABLE
    assert inventory.consume("pack-1").state is PackInventoryState.CONSUMED
    db.dispose()


def test_scheduler_persists_and_reuses_series_decision_across_restart(
    tmp_path: Path,
) -> None:
    db = _database(tmp_path)
    runtime = _running_runtime(db)
    series = SeriesRepository(db)
    series.create(
        title="Autonomous Arc",
        character_ids=("char-a",),
        editorial_policy=SeriesEditorialPolicy(min_gap_packs=0),
        series_id="series-1",
    )
    config = EditorialConfig(mode="continuous")
    editorial = _editorial(db, config, series=series)
    production = ProductionConfig(
        idea_inventory_target=0,
        planned_inventory_target=1,
        completed_inventory_target=None,
    )

    first = Scheduler(
        db,
        runtime,
        production,
        editorial=editorial,
    ).decide()
    second = Scheduler(
        db,
        runtime,
        production,
        editorial=_editorial(db, config, series=series),
    ).decide()

    assert first.action is SchedulerAction.PLAN_SERIES_PACK
    assert first.series_id == "series-1"
    assert first.series_plan_kind == "start"
    assert first.editorial_decision_id
    assert second.editorial_decision_id == first.editorial_decision_id
    db.dispose()


def test_full_roster_diversity_prefers_unused_branch_and_group(
    tmp_path: Path,
) -> None:
    db = _database(tmp_path)
    runtime = _running_runtime(db)
    now = datetime.now(UTC)
    with db.session() as session:
        session.add_all(
            (
                CharacterRow(
                    id="char-a",
                    display_name="A",
                    namespace="hololive",
                    enabled=True,
                    lora_policy="optional",
                    readiness=1.0,
                    profile_json={"branch": "JP", "group": "gen0"},
                    last_used_at=now,
                    total_generated=10,
                ),
                CharacterRow(
                    id="char-b",
                    display_name="B",
                    namespace="hololive",
                    enabled=True,
                    lora_policy="optional",
                    readiness=1.0,
                    profile_json={"branch": "EN", "group": "myth"},
                    last_used_at=None,
                    total_generated=0,
                ),
                _pack(
                    "recent-a",
                    when=now,
                    character_ids=("char-a",),
                    theme="repeat",
                ),
                ConceptRow(
                    id="idea-a",
                    status="idea",
                    payload_json={
                        "candidate": {
                            "character_ids": ["char-a"],
                            "format": "single_feature",
                            "theme": "repeat",
                        }
                    },
                    score=0.99,
                    created_at=now,
                ),
                ConceptRow(
                    id="idea-b",
                    status="idea",
                    payload_json={
                        "candidate": {
                            "character_ids": ["char-b"],
                            "format": "outfit_feature",
                            "theme": "fresh",
                        }
                    },
                    score=0.60,
                    created_at=now + timedelta(seconds=1),
                ),
            )
        )

    config = EditorialConfig(
        mode="continuous",
        character_cooldown_packs=2,
        diversity_window_packs=10,
        series_target_share=0.0,
    )
    editorial = _editorial(db, config)
    decision = Scheduler(
        db,
        runtime,
        ProductionConfig(
            idea_inventory_target=1,
            planned_inventory_target=1,
            completed_inventory_target=None,
        ),
        editorial=editorial,
    ).decide()

    assert decision.action is SchedulerAction.PLAN_PACK
    assert decision.concept_id == "idea-b"
    assert "diversity-ranked" in decision.reason
    db.dispose()



def test_series_advances_again_after_autonomous_standalone_gap(
    tmp_path: Path,
) -> None:
    db = _database(tmp_path)
    now = datetime.now(UTC)
    series = SeriesRepository(db)
    series.create(
        title="Long Arc",
        character_ids=("char-a",),
        editorial_policy=SeriesEditorialPolicy(
            min_gap_packs=1,
            priority=100,
        ),
        series_id="series-1",
    )
    config = EditorialConfig(
        mode="continuous",
        series_target_share=0.75,
        max_consecutive_series=2,
        max_consecutive_standalone=4,
    )
    editorial = _editorial(db, config, series=series)

    first = editorial.decide_plan(has_ideas=False)
    assert first.series_id == "series-1"
    assert first.series_plan_kind is not None
    assert first.series_plan_kind.value == "start"

    with db.session() as session:
        session.add(
            _pack(
                "series-pack-1",
                when=now,
                series_id="series-1",
                character_ids=("char-a",),
                format_type="continuation",
                theme="episode one",
            )
        )
        session.add(
            ConceptRow(
                id="standalone-idea",
                status="idea",
                payload_json={
                    "candidate": {
                        "character_ids": ["char-b"],
                        "format": "single_feature",
                        "theme": "standalone",
                    }
                },
                score=0.8,
                created_at=now + timedelta(seconds=1),
            )
        )
    series.record_completed_pack(
        "series-1",
        pack_id="series-pack-1",
        episode_number=1,
    )
    editorial.complete_decision(
        first.decision_id,
        pack_id="series-pack-1",
        concept_id=None,
    )

    middle = editorial.decide_plan(has_ideas=True)
    assert middle.series_id is None
    assert middle.concept_id == "standalone-idea"

    with db.session() as session:
        session.add(
            _pack(
                "standalone-pack",
                when=now + timedelta(seconds=2),
                character_ids=("char-b",),
                theme="standalone",
            )
        )
    editorial.complete_decision(
        middle.decision_id,
        pack_id="standalone-pack",
        concept_id="standalone-idea",
    )
    with db.session() as session:
        concept = session.get(ConceptRow, "standalone-idea")
        assert concept is not None
        concept.status = "planned"

    second = editorial.decide_plan(has_ideas=False)
    assert second.series_id == "series-1"
    assert second.series_plan_kind is not None
    assert second.series_plan_kind.value == "continue"
    db.dispose()
