from __future__ import annotations

from pathlib import Path

import pytest

from artifex.db import Database
from artifex.review import ReviewQueueRepository, ReviewState


def test_review_queue_persists_and_orders_open_items(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'review.sqlite3').as_posix()}")
    database.migrate()
    ids = iter(("review-1", "review-2"))
    repository = ReviewQueueRepository(database, id_factory=lambda: next(ids))

    first = repository.enqueue(
        subject_type="scene",
        subject_id="scene-1",
        reason="quality review",
        payload={"attempt_id": "attempt-1"},
    )
    second = repository.enqueue(
        subject_type="pack",
        subject_id="pack-1",
        reason="policy review",
    )

    assert [item.id for item in repository.list_open()] == ["review-1", "review-2"]
    assert repository.next_open() == first

    resolved = repository.resolve(first.id, ReviewState.APPROVED)
    assert resolved.state is ReviewState.APPROVED
    assert resolved.resolved_at is not None
    assert repository.next_open() == second

    with pytest.raises(ValueError, match="already resolved"):
        repository.resolve(first.id, ReviewState.REJECTED)
    database.dispose()
