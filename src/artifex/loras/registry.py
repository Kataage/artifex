from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import LoRARow
from artifex.domain import LoRAProfile, LoRAState


class InvalidLoRAStateTransition(ValueError):
    pass


_TRANSITIONS: dict[LoRAState, frozenset[LoRAState]] = {
    LoRAState.DISCOVERED: frozenset(
        {LoRAState.PENDING, LoRAState.DISABLED, LoRAState.FAILED}
    ),
    LoRAState.PENDING: frozenset(
        {LoRAState.VALIDATED, LoRAState.DISABLED, LoRAState.FAILED}
    ),
    LoRAState.VALIDATED: frozenset(
        {
            LoRAState.PRODUCTION,
            LoRAState.PENDING,
            LoRAState.DISABLED,
            LoRAState.FAILED,
        }
    ),
    LoRAState.PRODUCTION: frozenset(
        {LoRAState.VALIDATED, LoRAState.DISABLED, LoRAState.FAILED}
    ),
    LoRAState.DISABLED: frozenset({LoRAState.PENDING}),
    LoRAState.FAILED: frozenset({LoRAState.PENDING}),
}


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


class LoRARegistry:
    def __init__(self, database: Database) -> None:
        self._database = database

    def upsert(self, profile: LoRAProfile) -> LoRAProfile:
        normalized = profile.model_copy(update={"path": _normalize_path(profile.path)})
        payload = normalized.model_dump(mode="json")
        now = datetime.now(UTC)

        with self._database.session() as session:
            row = session.get(LoRARow, normalized.id)
            path_row = session.scalar(
                select(LoRARow).where(LoRARow.path == str(normalized.path))
            )
            if row is None and path_row is not None and path_row.id != normalized.id:
                raise ValueError(
                    f"LoRA path already belongs to {path_row.id}: {normalized.path}"
                )
            if row is None:
                session.add(
                    LoRARow(
                        id=normalized.id,
                        path=str(normalized.path),
                        checksum=normalized.checksum,
                        state=normalized.state.value,
                        lora_type=normalized.lora_type,
                        readiness=normalized.readiness,
                        metadata_json=payload,
                        updated_at=now,
                    )
                )
            else:
                row.path = str(normalized.path)
                row.checksum = normalized.checksum
                row.state = normalized.state.value
                row.lora_type = normalized.lora_type
                row.readiness = normalized.readiness
                row.metadata_json = payload
                row.updated_at = now
        return normalized

    def get(self, lora_id: str) -> LoRAProfile | None:
        with self._database.session() as session:
            row = session.get(LoRARow, lora_id)
            return None if row is None else self._profile_from_row(row)

    def require(self, lora_id: str) -> LoRAProfile:
        profile = self.get(lora_id)
        if profile is None:
            raise KeyError(f"unknown LoRA: {lora_id}")
        return profile

    def find_by_path(self, path: Path) -> LoRAProfile | None:
        normalized = str(_normalize_path(path))
        with self._database.session() as session:
            row = session.scalar(select(LoRARow).where(LoRARow.path == normalized))
            return None if row is None else self._profile_from_row(row)

    def list(
        self,
        *,
        states: Iterable[LoRAState] | None = None,
    ) -> tuple[LoRAProfile, ...]:
        query = select(LoRARow).order_by(LoRARow.id.asc())
        if states is not None:
            state_values = tuple(state.value for state in states)
            if not state_values:
                return ()
            query = query.where(LoRARow.state.in_(state_values))
        with self._database.session() as session:
            rows = session.scalars(query).all()
            return tuple(self._profile_from_row(row) for row in rows)

    def for_character(
        self,
        character_id: str,
        *,
        model_family: str | None = None,
        production_only: bool = False,
    ) -> tuple[LoRAProfile, ...]:
        states = (LoRAState.PRODUCTION,) if production_only else None
        result = []
        for profile in self.list(states=states):
            if character_id not in profile.target_character_ids:
                continue
            if (
                model_family is not None
                and profile.model_families
                and model_family not in profile.model_families
            ):
                continue
            result.append(profile)
        return tuple(result)

    def transition_state(self, lora_id: str, target: LoRAState) -> LoRAProfile:
        current = self.require(lora_id)
        if current.state == target:
            return current
        if target not in _TRANSITIONS[current.state]:
            raise InvalidLoRAStateTransition(
                f"invalid LoRA transition for {lora_id}: "
                f"{current.state.value} -> {target.value}"
            )
        return self.upsert(current.model_copy(update={"state": target}))

    @staticmethod
    def _profile_from_row(row: LoRARow) -> LoRAProfile:
        payload = dict(row.metadata_json)
        payload.update(
            {
                "id": row.id,
                "path": row.path,
                "checksum": row.checksum,
                "state": row.state,
                "lora_type": row.lora_type,
                "readiness": row.readiness,
            }
        )
        return LoRAProfile.model_validate(payload)
