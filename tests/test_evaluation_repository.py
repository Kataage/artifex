from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from artifex.db import Database
from artifex.db.models import PackRow, SceneRow
from artifex.domain import ResultState, SceneState
from artifex.evaluation import (
    AttemptSelector,
    EvaluationRepository,
    EvaluationResult,
    EvaluationScores,
    GenerationAttemptRepository,
)
from artifex.runtime import RuntimeStore


def _scores(aggregate: float) -> EvaluationScores:
    return EvaluationScores(
        identity=0.9,
        alignment=0.9,
        face_quality=0.9,
        technical_quality=0.9,
        aesthetic=0.9,
        image_similarity=0.1,
        novelty=0.9,
        continuity=0.9,
        integrity=1.0,
        aggregate=aggregate,
    )


def _database(tmp_path: Path) -> Database:
    database = Database(f"sqlite:///{(tmp_path / 'evaluation.sqlite3').as_posix()}")
    database.migrate()
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            PackRow(
                id="pack-1",
                state="evaluating",
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
                state=SceneState.EVALUATING.value,
                publication_tier="public",
                payload_json={},
            )
        )
    return database


def test_attempt_parent_chain_and_selection_persist(tmp_path: Path) -> None:
    database = _database(tmp_path)
    ids = iter(("attempt-1", "attempt-2"))
    attempts = GenerationAttemptRepository(database, id_factory=lambda: next(ids))

    first = attempts.create(
        scene_id="scene-1",
        prompt="p1",
        negative_prompt="n",
        seed=1,
        provenance={"workflow": "v1"},
    )
    second = attempts.create(
        scene_id="scene-1",
        prompt="p2",
        negative_prompt="n",
        seed=2,
        provenance={"retry_reason": "identity"},
        parent_attempt_id=first.id,
    )

    assert first.ordinal == 1
    assert second.ordinal == 2
    assert second.parent_attempt_id == "attempt-1"

    eval_ids = iter(("eval-1", "eval-2"))
    evaluations = EvaluationRepository(
        database,
        id_factory=lambda: next(eval_ids),
    )
    evaluations.record(
        EvaluationResult(
            attempt_id="attempt-1",
            state=ResultState.REVIEW,
            scores=_scores(0.7),
            reasons=("identity_needs_review",),
        )
    )
    evaluations.record(
        EvaluationResult(
            attempt_id="attempt-2",
            state=ResultState.ACCEPTED,
            scores=_scores(0.9),
        )
    )

    selection = AttemptSelector(
        database,
        RuntimeStore(database),
        evaluations,
    ).select("scene-1")

    assert selection.state is ResultState.ACCEPTED
    assert selection.attempt_id == "attempt-2"

    with database.session() as session:
        scene = session.get(SceneRow, "scene-1")
        assert scene is not None
        assert scene.state == SceneState.ACCEPTED.value
        assert scene.selected_attempt_id == "attempt-2"
    database.dispose()
