from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import ClassVar
from uuid import uuid4

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import (
    EditorialDecisionRow,
    PackInventoryRow,
    PackRow,
    SettingRow,
)
from artifex.domain import PackState
from artifex.editorial.models import (
    EditorialLane,
    EditorialPlanDecision,
    InventoryCounts,
    InventoryItem,
    PackInventoryState,
    SeriesPlanKind,
)


def _new_id() -> str:
    return uuid4().hex


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class PackInventoryRepository:
    _ALLOWED: ClassVar[
        dict[PackInventoryState, frozenset[PackInventoryState]]
    ] = {
        PackInventoryState.AVAILABLE: frozenset(
            {
                PackInventoryState.RESERVED,
                PackInventoryState.CONSUMED,
                PackInventoryState.EXPIRED,
            }
        ),
        PackInventoryState.RESERVED: frozenset(
            {
                PackInventoryState.AVAILABLE,
                PackInventoryState.CONSUMED,
                PackInventoryState.EXPIRED,
            }
        ),
        PackInventoryState.CONSUMED: frozenset(),
        PackInventoryState.EXPIRED: frozenset(),
    }

    def __init__(
        self,
        database: Database,
        *,
        expiry_hours: float | None,
    ) -> None:
        self._database = database
        self._expiry_hours = expiry_hours

    def sync_finalized(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(UTC)
        created = 0
        with self._database.session() as session:
            packs = session.scalars(
                select(PackRow).where(PackRow.state == PackState.FINALIZED.value)
            ).all()
            for pack in packs:
                if session.get(PackInventoryRow, pack.id) is not None:
                    continue
                legacy_consumed = bool(
                    pack.payload_json.get("inventory_consumed", False)
                )
                state = (
                    PackInventoryState.CONSUMED
                    if legacy_consumed
                    else PackInventoryState.AVAILABLE
                )
                expires_at = (
                    None
                    if legacy_consumed or self._expiry_hours is None
                    else _utc(pack.updated_at)
                    + timedelta(hours=self._expiry_hours)
                )
                session.add(
                    PackInventoryRow(
                        pack_id=pack.id,
                        state=state.value,
                        reserved_at=None,
                        consumed_at=current if legacy_consumed else None,
                        expires_at=expires_at,
                        metadata_json={"migrated_legacy": True},
                        updated_at=current,
                    )
                )
                created += 1
        return created

    def mark_available(
        self,
        pack_id: str,
        *,
        metadata: dict[str, object] | None = None,
        now: datetime | None = None,
    ) -> InventoryItem:
        current = now or datetime.now(UTC)
        with self._database.session() as session:
            pack = session.get(PackRow, pack_id)
            if pack is None:
                raise KeyError(f"unknown pack: {pack_id}")
            if pack.state != PackState.FINALIZED.value:
                raise ValueError("only finalized Packs may enter completed inventory")
            row = session.get(PackInventoryRow, pack_id)
            if row is None:
                expires_at = (
                    current + timedelta(hours=self._expiry_hours)
                    if self._expiry_hours is not None
                    else None
                )
                session.add(
                    PackInventoryRow(
                        pack_id=pack_id,
                        state=PackInventoryState.AVAILABLE.value,
                        reserved_at=None,
                        consumed_at=None,
                        expires_at=expires_at,
                        metadata_json=dict(metadata or {}),
                        updated_at=current,
                    )
                )
            else:
                row.metadata_json = {
                    **dict(row.metadata_json),
                    **dict(metadata or {}),
                }
                row.updated_at = current
        return self.require(pack_id)

    def expire_due(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(UTC)
        with self._database.session() as session:
            rows = session.scalars(
                select(PackInventoryRow).where(
                    PackInventoryRow.state.in_(
                        (
                            PackInventoryState.AVAILABLE.value,
                            PackInventoryState.RESERVED.value,
                        )
                    ),
                    PackInventoryRow.expires_at.is_not(None),
                    PackInventoryRow.expires_at <= current,
                )
            ).all()
            for row in rows:
                row.state = PackInventoryState.EXPIRED.value
                row.updated_at = current
            return len(rows)

    def transition(
        self,
        pack_id: str,
        target: PackInventoryState,
        *,
        metadata: dict[str, object] | None = None,
        now: datetime | None = None,
    ) -> InventoryItem:
        current = now or datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(PackInventoryRow, pack_id)
            if row is None:
                raise KeyError(f"pack is not in completed inventory: {pack_id}")
            source = PackInventoryState(row.state)
            if target is source:
                return self._item(row)
            if target not in self._ALLOWED[source]:
                raise ValueError(
                    f"invalid inventory transition: {source.value} -> {target.value}"
                )
            row.state = target.value
            if target is PackInventoryState.RESERVED:
                row.reserved_at = current
            elif target is PackInventoryState.AVAILABLE:
                row.reserved_at = None
            elif target is PackInventoryState.CONSUMED:
                row.consumed_at = current
            row.metadata_json = {
                **dict(row.metadata_json),
                **dict(metadata or {}),
            }
            row.updated_at = current
        return self.require(pack_id)

    def reserve(self, pack_id: str) -> InventoryItem:
        return self.transition(pack_id, PackInventoryState.RESERVED)

    def release(self, pack_id: str) -> InventoryItem:
        return self.transition(pack_id, PackInventoryState.AVAILABLE)

    def consume(self, pack_id: str) -> InventoryItem:
        return self.transition(pack_id, PackInventoryState.CONSUMED)

    def expire(self, pack_id: str) -> InventoryItem:
        return self.transition(pack_id, PackInventoryState.EXPIRED)

    def get(self, pack_id: str) -> InventoryItem | None:
        with self._database.session() as session:
            row = session.get(PackInventoryRow, pack_id)
            return None if row is None else self._item(row)

    def require(self, pack_id: str) -> InventoryItem:
        item = self.get(pack_id)
        if item is None:
            raise KeyError(f"pack is not in completed inventory: {pack_id}")
        return item

    def list(self) -> tuple[InventoryItem, ...]:
        with self._database.session() as session:
            rows = session.scalars(
                select(PackInventoryRow).order_by(
                    PackInventoryRow.updated_at.desc(),
                    PackInventoryRow.pack_id.asc(),
                )
            ).all()
            return tuple(self._item(row) for row in rows)

    def counts(self) -> InventoryCounts:
        values = {state: 0 for state in PackInventoryState}
        with self._database.session() as session:
            rows = session.scalars(select(PackInventoryRow)).all()
            for row in rows:
                values[PackInventoryState(row.state)] += 1
        return InventoryCounts(
            available=values[PackInventoryState.AVAILABLE],
            reserved=values[PackInventoryState.RESERVED],
            consumed=values[PackInventoryState.CONSUMED],
            expired=values[PackInventoryState.EXPIRED],
        )

    @staticmethod
    def _item(row: PackInventoryRow) -> InventoryItem:
        return InventoryItem(
            pack_id=row.pack_id,
            state=PackInventoryState(row.state),
            reserved_at=row.reserved_at,
            consumed_at=row.consumed_at,
            expires_at=row.expires_at,
            metadata=dict(row.metadata_json),
            updated_at=row.updated_at,
        )


class EditorialRepository:
    _REFILL_KEY = "editorial.refill_active"

    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def refill_active(self) -> bool | None:
        with self._database.session() as session:
            row = session.get(SettingRow, self._REFILL_KEY)
            if row is None:
                return None
            value = row.value_json
            if isinstance(value, bool):
                return value
            if isinstance(value, dict) and isinstance(value.get("active"), bool):
                return bool(value["active"])
            return None

    def set_refill_active(self, active: bool) -> None:
        now = datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(SettingRow, self._REFILL_KEY)
            if row is None:
                session.add(
                    SettingRow(
                        key=self._REFILL_KEY,
                        value_json={"active": active},
                        updated_at=now,
                    )
                )
            else:
                row.value_json = {"active": active}
                row.updated_at = now

    def pending(self) -> EditorialPlanDecision | None:
        with self._database.session() as session:
            row = session.scalar(
                select(EditorialDecisionRow)
                .where(EditorialDecisionRow.status == "pending")
                .order_by(
                    EditorialDecisionRow.created_at.asc(),
                    EditorialDecisionRow.id.asc(),
                )
                .limit(1)
            )
            return None if row is None else self._decision(row)

    def create(
        self,
        *,
        lane: EditorialLane,
        reason: str,
        series_id: str | None,
        concept_id: str | None,
        series_plan_kind: SeriesPlanKind | None,
        payload: dict[str, object],
    ) -> EditorialPlanDecision:
        now = datetime.now(UTC)
        decision_id = self._id_factory()
        row = EditorialDecisionRow(
            id=decision_id,
            action=lane.value,
            status="pending",
            series_id=series_id,
            concept_id=concept_id,
            pack_id=None,
            reason=reason,
            payload_json={
                **payload,
                "series_plan_kind": (
                    series_plan_kind.value
                    if series_plan_kind is not None
                    else None
                ),
            },
            created_at=now,
            resolved_at=None,
        )
        with self._database.session() as session:
            session.add(row)
        return self.require(decision_id)

    def require(self, decision_id: str) -> EditorialPlanDecision:
        with self._database.session() as session:
            row = session.get(EditorialDecisionRow, decision_id)
            if row is None:
                raise KeyError(f"unknown editorial decision: {decision_id}")
            return self._decision(row)

    def resolve(
        self,
        decision_id: str,
        *,
        status: str,
        pack_id: str | None = None,
        concept_id: str | None = None,
        payload_patch: dict[str, object] | None = None,
    ) -> None:
        if status not in {"completed", "cancelled", "failed"}:
            raise ValueError("invalid editorial decision resolution status")
        now = datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(EditorialDecisionRow, decision_id)
            if row is None:
                raise KeyError(f"unknown editorial decision: {decision_id}")
            row.status = status
            if pack_id is not None:
                row.pack_id = pack_id
            if concept_id is not None:
                row.concept_id = concept_id
            row.payload_json = {
                **dict(row.payload_json),
                **dict(payload_patch or {}),
            }
            row.resolved_at = now

    @staticmethod
    def _decision(row: EditorialDecisionRow) -> EditorialPlanDecision:
        raw_kind = row.payload_json.get("series_plan_kind")
        return EditorialPlanDecision(
            decision_id=row.id,
            lane=EditorialLane(row.action),
            reason=row.reason,
            series_id=row.series_id,
            concept_id=row.concept_id,
            series_plan_kind=SeriesPlanKind(raw_kind) if raw_kind else None,
            payload=dict(row.payload_json),
        )
