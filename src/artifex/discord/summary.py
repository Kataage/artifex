from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from artifex.db import Database
from artifex.db.models import AgentEventRow, PackRow, ReviewQueueRow
from artifex.domain import PackState
from artifex.review import ReviewState


class DailySummaryBuilder:
    def __init__(self, database: Database) -> None:
        self._database = database

    def build(
        self,
        *,
        now: datetime | None = None,
        hours: int = 24,
    ) -> str:
        current = now or datetime.now(UTC)
        since = current - timedelta(hours=hours)

        with self._database.session() as session:
            finalized = session.scalar(
                select(func.count())
                .select_from(PackRow)
                .where(
                    PackRow.state == PackState.FINALIZED.value,
                    PackRow.updated_at >= since,
                )
            ) or 0
            failed = session.scalar(
                select(func.count())
                .select_from(PackRow)
                .where(
                    PackRow.state == PackState.FAILED.value,
                    PackRow.updated_at >= since,
                )
            ) or 0
            blocked = session.scalar(
                select(func.count())
                .select_from(PackRow)
                .where(
                    PackRow.state == PackState.BLOCKED.value,
                    PackRow.updated_at >= since,
                )
            ) or 0
            open_reviews = session.scalar(
                select(func.count())
                .select_from(ReviewQueueRow)
                .where(ReviewQueueRow.state == ReviewState.OPEN.value)
            ) or 0
            errors = session.scalar(
                select(func.count())
                .select_from(AgentEventRow)
                .where(
                    AgentEventRow.created_at >= since,
                    AgentEventRow.severity.in_(("error", "critical")),
                )
            ) or 0

        return (
            f"Artifex daily: finalized={finalized}, failed={failed}, "
            f"blocked={blocked}, open_reviews={open_reviews}, "
            f"errors={errors} (last {hours}h)"
        )
