from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import SettingRow
from artifex.trends.models import SignalSourceHealth


class SignalHealthRepository:
    _PREFIX = "signals.source."

    def __init__(self, database: Database) -> None:
        self._database = database

    def success(
        self,
        provider: str,
        *,
        kind: str,
        count: int,
        at: datetime,
    ) -> SignalSourceHealth:
        current = _utc(at)
        previous = self.get(provider, kind=kind)
        health = SignalSourceHealth(
            provider=provider,
            kind=kind,
            state="healthy",
            last_attempt_at=current,
            last_success_at=current,
            last_error=None,
            consecutive_failures=0,
            last_count=max(0, count),
        )
        self._write(health, previous=previous)
        return health

    def failure(
        self,
        provider: str,
        *,
        kind: str,
        error: str,
        at: datetime,
    ) -> SignalSourceHealth:
        current = _utc(at)
        previous = self.get(provider, kind=kind)
        failures = (
            previous.consecutive_failures + 1
            if previous is not None
            else 1
        )
        health = SignalSourceHealth(
            provider=provider,
            kind=kind,
            state="degraded",
            last_attempt_at=current,
            last_success_at=(
                previous.last_success_at if previous is not None else None
            ),
            last_error=error[:1000],
            consecutive_failures=failures,
            last_count=previous.last_count if previous is not None else 0,
        )
        self._write(health, previous=previous)
        return health

    def get(self, provider: str, *, kind: str) -> SignalSourceHealth | None:
        key = self._key(provider, kind)
        with self._database.session() as session:
            row = session.get(SettingRow, key)
            if row is None or not isinstance(row.value_json, dict):
                return None
            return SignalSourceHealth.model_validate(row.value_json)

    def list(self) -> tuple[SignalSourceHealth, ...]:
        with self._database.session() as session:
            rows = session.scalars(
                select(SettingRow)
                .where(SettingRow.key.like(f"{self._PREFIX}%"))
                .order_by(SettingRow.key.asc())
            ).all()
        result: list[SignalSourceHealth] = []
        for row in rows:
            if isinstance(row.value_json, dict):
                result.append(SignalSourceHealth.model_validate(row.value_json))
        return tuple(result)

    def _write(
        self,
        health: SignalSourceHealth,
        *,
        previous: SignalSourceHealth | None,
    ) -> None:
        del previous
        now = _utc(health.last_attempt_at or datetime.now(UTC))
        key = self._key(health.provider, health.kind)
        payload = health.model_dump(mode="json")
        with self._database.session() as session:
            row = session.get(SettingRow, key)
            if row is None:
                session.add(
                    SettingRow(
                        key=key,
                        value_json=payload,
                        updated_at=now,
                    )
                )
            else:
                row.value_json = payload
                row.updated_at = now

    @classmethod
    def _key(cls, provider: str, kind: str) -> str:
        safe_provider = provider.replace(" ", "_")
        safe_kind = kind.replace(" ", "_")
        return f"{cls._PREFIX}{safe_kind}.{safe_provider}"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
