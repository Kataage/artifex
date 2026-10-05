from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import ReviewQueueRow
from artifex.review.models import ReviewItem, ReviewState


def _new_id() -> str:
    return uuid4().hex


class ReviewQueueRepository:
    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def enqueue(
        self,
        *,
        subject_type: str,
        subject_id: str,
        reason: str,
        payload: Mapping[str, object] | None = None,
    ) -> ReviewItem:
        now = datetime.now(UTC)
        row = ReviewQueueRow(
            id=self._id_factory(),
            subject_type=subject_type,
            subject_id=subject_id,
            state=ReviewState.OPEN.value,
            reason=reason,
            payload_json=dict(payload or {}),
            created_at=now,
            resolved_at=None,
        )
        with self._database.session() as session:
            session.add(row)
            item_id = row.id
        return self.require(item_id)

    def get(self, item_id: str) -> ReviewItem | None:
        with self._database.session() as session:
            row = session.get(ReviewQueueRow, item_id)
            if row is None:
                return None
            return self._to_item(row)

    def require(self, item_id: str) -> ReviewItem:
        item = self.get(item_id)
        if item is None:
            raise KeyError(f"unknown review item: {item_id}")
        return item

    def list_open(self, *, limit: int = 20) -> tuple[ReviewItem, ...]:
        with self._database.session() as session:
            rows = session.scalars(
                select(ReviewQueueRow)
                .where(ReviewQueueRow.state == ReviewState.OPEN.value)
                .order_by(ReviewQueueRow.created_at.asc(), ReviewQueueRow.id.asc())
                .limit(limit)
            ).all()
            return tuple(self._to_item(row) for row in rows)

    def next_open(self) -> ReviewItem | None:
        items = self.list_open(limit=1)
        return items[0] if items else None

    def resolve(
        self,
        item_id: str,
        state: ReviewState,
        *,
        payload_patch: Mapping[str, object] | None = None,
    ) -> ReviewItem:
        if state is ReviewState.OPEN:
            raise ValueError("resolved review state cannot be open")
        now = datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(ReviewQueueRow, item_id)
            if row is None:
                raise KeyError(f"unknown review item: {item_id}")
            if row.state != ReviewState.OPEN.value:
                raise ValueError(f"review item is already resolved: {item_id}")
            row.state = state.value
            row.resolved_at = now
            if payload_patch:
                row.payload_json = {**row.payload_json, **dict(payload_patch)}
        return self.require(item_id)

    @staticmethod
    def _to_item(row: ReviewQueueRow) -> ReviewItem:
        return ReviewItem(
            id=row.id,
            subject_type=row.subject_type,
            subject_id=row.subject_id,
            state=ReviewState(row.state),
            reason=row.reason,
            payload=dict(row.payload_json),
            created_at=row.created_at,
            resolved_at=row.resolved_at,
        )
