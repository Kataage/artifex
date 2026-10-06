from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from artifex.characters import CharacterRegistry
from artifex.config.models import ProductionConfig
from artifex.db import Database
from artifex.db.models import ConceptRow, PackRow, SceneRow
from artifex.discord import ArtifexRemoteOperations, CommandName, CommandRequest
from artifex.domain import (
    AgentState,
    CharacterProfile,
    PackState,
    PublicationTier,
    SceneState,
)
from artifex.policy import (
    PolicyDecision,
    PolicyDecisionRepository,
    PolicyOutcome,
)
from artifex.review import ReviewQueueRepository, ReviewState
from artifex.runtime import RuntimeStore
from artifex.scheduler import Scheduler
from artifex.series import SeriesRepository


def _setup(tmp_path: Path) -> tuple[
    Database,
    RuntimeStore,
    ReviewQueueRepository,
    ArtifexRemoteOperations,
]:
    database = Database(f"sqlite:///{(tmp_path / 'discord.sqlite3').as_posix()}")
    database.migrate()
    runtime = RuntimeStore(database)
    runtime.set_agent_state(AgentState.STARTING, expected=AgentState.STOPPED)
    runtime.set_agent_state(AgentState.RUNNING, expected=AgentState.STARTING)
    reviews = ReviewQueueRepository(database)
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="Character A",
            namespace="test",
            readiness=0.9,
        )
    )
    operations = ArtifexRemoteOperations(
        database,
        runtime,
        Scheduler(
            database,
            runtime,
            ProductionConfig(
                idea_inventory_target=0,
                planned_inventory_target=0,
                completed_inventory_target=None,
            ),
        ),
        reviews,
        characters,
        SeriesRepository(database),
        PolicyDecisionRepository(database),
    )
    return database, runtime, reviews, operations


def _scene_review(database: Database) -> None:
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            ConceptRow(
                id="concept-1",
                status="idea",
                payload_json={},
                created_at=now,
            )
        )
        session.add(
            PackRow(
                id="pack-1",
                concept_id="concept-1",
                state=PackState.REVIEW.value,
                format_type="single_feature",
                payload_json={},
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            SceneRow(
                id="scene-1",
                pack_id="pack-1",
                ordinal=1,
                state=SceneState.REVIEW.value,
                publication_tier="private_review",
                payload_json={},
            )
        )


def test_status_pause_resume_and_character_inspection(tmp_path: Path) -> None:
    database, runtime, _, operations = _setup(tmp_path)

    status = operations.execute(CommandRequest(name=CommandName.STATUS))
    paused = operations.execute(CommandRequest(name=CommandName.PAUSE))
    resumed = operations.execute(CommandRequest(name=CommandName.RESUME))
    character = operations.execute(
        CommandRequest(name=CommandName.CHARACTER, args=("char-a",))
    )

    assert status.ok is True
    assert status.data["agent_state"] == "running"
    assert paused.ok is True
    assert runtime.get_agent_state() is AgentState.RUNNING
    assert resumed.ok is True
    assert character.data["id"] == "char-a"
    database.dispose()


def test_review_approve_and_retry_are_persisted(tmp_path: Path) -> None:
    database, _, reviews, operations = _setup(tmp_path)
    _scene_review(database)
    review = reviews.enqueue(
        subject_type="scene",
        subject_id="scene-1",
        reason="manual quality review",
    )

    approved = operations.execute(
        CommandRequest(name=CommandName.APPROVE, args=(review.id,))
    )

    assert approved.ok is True
    assert reviews.require(review.id).state is ReviewState.APPROVED
    with database.session() as session:
        scene = session.get(SceneRow, "scene-1")
        assert scene is not None
        assert scene.state == SceneState.ACCEPTED.value
    database.dispose()


def test_alternate_idea_retires_pack_and_concept(tmp_path: Path) -> None:
    database, _, reviews, operations = _setup(tmp_path)
    _scene_review(database)
    review = reviews.enqueue(
        subject_type="pack",
        subject_id="pack-1",
        reason="idea review",
    )

    response = operations.execute(
        CommandRequest(name=CommandName.ALTERNATE, args=(review.id,))
    )

    assert response.ok is True
    assert reviews.require(review.id).state is ReviewState.SKIPPED
    with database.session() as session:
        pack = session.get(PackRow, "pack-1")
        concept = session.get(ConceptRow, "concept-1")
        assert pack is not None
        assert concept is not None
        assert pack.state == PackState.FAILED.value
        assert concept.status == "rejected_by_operator"
    database.dispose()


def test_retry_moves_review_scene_back_to_ready(tmp_path: Path) -> None:
    database, _, reviews, operations = _setup(tmp_path)
    _scene_review(database)
    review = reviews.enqueue(
        subject_type="scene",
        subject_id="scene-1",
        reason="regenerate",
    )

    response = operations.execute(
        CommandRequest(name=CommandName.RETRY, args=(review.id,))
    )

    assert response.ok is True
    assert reviews.require(review.id).state is ReviewState.RETRY
    with database.session() as session:
        scene = session.get(SceneRow, "scene-1")
        assert scene is not None
        assert scene.state == SceneState.READY.value
        assert scene.payload_json["operator_retry"] is True
    database.dispose()



def test_policy_review_approval_restores_effective_publication_tier(
    tmp_path: Path,
) -> None:
    database, _, reviews, operations = _setup(tmp_path)
    _scene_review(database)
    decisions = PolicyDecisionRepository(database)
    decision = decisions.persist(
        PolicyDecision(
            decision_id="policy-review-1",
            subject_type="scene",
            subject_id="scene-1",
            outcome=PolicyOutcome.REVIEW,
            requested_tier=PublicationTier.PUBLIC,
            effective_tier=PublicationTier.PRIVATE_REVIEW,
            profile_versions=("test@1",),
            reasons=("manual policy review",),
            content_labels=(),
            created_at=datetime.now(UTC),
        )
    )
    review = reviews.enqueue(
        subject_type="scene",
        subject_id="scene-1",
        reason="policy review",
        payload={"policy_decision_id": decision.decision_id},
    )

    response = operations.execute(
        CommandRequest(name=CommandName.APPROVE, args=(review.id,))
    )

    assert response.ok is True
    with database.session() as session:
        scene = session.get(SceneRow, "scene-1")
        assert scene is not None
        assert scene.publication_tier == PublicationTier.PUBLIC.value
        assert scene.state == SceneState.ACCEPTED.value
        assert scene.payload_json["latest_policy_decision_id"] != decision.decision_id
    database.dispose()


def test_hard_blocked_generated_scene_can_be_retried(
    tmp_path: Path,
) -> None:
    database, _, reviews, operations = _setup(tmp_path)
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            PackRow(
                id="pack-blocked",
                state=PackState.BLOCKED.value,
                format_type="single_feature",
                payload_json={},
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            SceneRow(
                id="scene-blocked",
                pack_id="pack-blocked",
                ordinal=1,
                state=SceneState.BLOCKED.value,
                publication_tier=PublicationTier.BLOCKED.value,
                payload_json={},
            )
        )
    review = reviews.enqueue(
        subject_type="scene",
        subject_id="scene-blocked",
        reason="post-generation policy hard block",
    )

    response = operations.execute(
        CommandRequest(name=CommandName.RETRY, args=(review.id,))
    )

    assert response.ok is True
    assert reviews.require(review.id).state is ReviewState.RETRY
    with database.session() as session:
        scene = session.get(SceneRow, "scene-blocked")
        pack = session.get(PackRow, "pack-blocked")
        assert scene is not None
        assert pack is not None
        assert scene.state == SceneState.PLANNED.value
        assert scene.payload_json["operator_retry"] is True
        assert pack.state == PackState.PLANNED.value
    database.dispose()
