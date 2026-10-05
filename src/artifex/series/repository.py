from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import SeriesRow
from artifex.planner.models import PackFormat
from artifex.series.models import SeriesProfile, SeriesStatus


def _new_id() -> str:
    return uuid4().hex


class SeriesRepository:
    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def create(
        self,
        *,
        title: str,
        character_ids: Sequence[str],
        preferred_format: PackFormat | None = None,
        continuity_state: Mapping[str, Any] | None = None,
        bible: Sequence[str] = (),
        unresolved_hooks: Sequence[str] = (),
        series_id: str | None = None,
    ) -> SeriesProfile:
        now = datetime.now(UTC)
        profile = SeriesProfile(
            id=series_id or self._id_factory(),
            title=title,
            character_ids=tuple(character_ids),
            preferred_format=preferred_format,
            continuity_state=dict(continuity_state or {}),
            bible=tuple(bible),
            unresolved_hooks=tuple(unresolved_hooks),
            created_at=now,
            updated_at=now,
        )
        with self._database.session() as session:
            if session.get(SeriesRow, profile.id) is not None:
                raise ValueError(f"series already exists: {profile.id}")
            session.add(self._row_from_profile(profile))
        return profile

    def get(self, series_id: str) -> SeriesProfile | None:
        with self._database.session() as session:
            row = session.get(SeriesRow, series_id)
            return None if row is None else self._profile_from_row(row)

    def require(self, series_id: str) -> SeriesProfile:
        profile = self.get(series_id)
        if profile is None:
            raise KeyError(f"unknown series: {series_id}")
        return profile

    def list(
        self,
        *,
        status: SeriesStatus | None = None,
    ) -> tuple[SeriesProfile, ...]:
        query = select(SeriesRow).order_by(SeriesRow.updated_at.asc(), SeriesRow.id.asc())
        if status is not None:
            query = query.where(SeriesRow.status == status.value)
        with self._database.session() as session:
            rows = session.scalars(query).all()
            return tuple(self._profile_from_row(row) for row in rows)

    def set_status(self, series_id: str, status: SeriesStatus) -> SeriesProfile:
        now = datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(SeriesRow, series_id)
            if row is None:
                raise KeyError(f"unknown series: {series_id}")
            row.status = status.value
            row.updated_at = now
        return self.require(series_id)

    def record_completed_pack(
        self,
        series_id: str,
        *,
        pack_id: str,
        episode_number: int,
        continuity_updates: Mapping[str, Any] | None = None,
        hooks_added: Sequence[str] = (),
        hooks_resolved: Sequence[str] = (),
        episode_summary: str | None = None,
    ) -> SeriesProfile:
        now = datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(SeriesRow, series_id)
            if row is None:
                raise KeyError(f"unknown series: {series_id}")
            current = self._profile_from_row(row)
            expected_episode = current.current_episode + 1
            if episode_number != expected_episode:
                raise ValueError(
                    f"series {series_id} expected episode {expected_episode}, "
                    f"got {episode_number}"
                )
            if pack_id in current.prior_pack_ids:
                raise ValueError(f"pack already recorded in series: {pack_id}")

            continuity = dict(current.continuity_state)
            continuity.update(dict(continuity_updates or {}))

            resolved = set(hooks_resolved)
            hooks = [
                hook
                for hook in current.unresolved_hooks
                if hook not in resolved
            ]
            for hook in hooks_added:
                if hook not in hooks:
                    hooks.append(hook)

            summary = (
                episode_summary.strip()
                if episode_summary is not None and episode_summary.strip()
                else (
                    f"Episode {episode_number} completed. "
                    f"Continuity keys: {', '.join(sorted(continuity)) or 'none'}. "
                    f"Unresolved hooks: {', '.join(hooks) or 'none'}."
                )
            )
            recent_summaries = (
                *current.recent_episode_summaries,
                summary[:1200],
            )[-20:]
            rolling_summary = (
                f"Through episode {episode_number}: "
                f"{'; '.join(f'{key}={continuity[key]}' for key in sorted(continuity))}. "
                f"Unresolved hooks: {', '.join(hooks) or 'none'}."
            )[:3000]

            updated = current.model_copy(
                update={
                    "current_episode": episode_number,
                    "prior_pack_ids": (*current.prior_pack_ids, pack_id),
                    "continuity_state": continuity,
                    "rolling_summary": rolling_summary,
                    "recent_episode_summaries": recent_summaries,
                    "unresolved_hooks": tuple(hooks),
                    "updated_at": now,
                }
            )
            row.current_episode = updated.current_episode
            row.status = updated.status.value
            row.state_json = self._state_json(updated)
            row.updated_at = now
        return self.require(series_id)

    @staticmethod
    def _row_from_profile(profile: SeriesProfile) -> SeriesRow:
        return SeriesRow(
            id=profile.id,
            status=profile.status.value,
            current_episode=profile.current_episode,
            state_json=SeriesRepository._state_json(profile),
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )

    @staticmethod
    def _state_json(profile: SeriesProfile) -> dict[str, Any]:
        return {
            "title": profile.title,
            "prior_pack_ids": list(profile.prior_pack_ids),
            "character_ids": list(profile.character_ids),
            "continuity_state": profile.continuity_state,
            "bible": list(profile.bible),
            "rolling_summary": profile.rolling_summary,
            "recent_episode_summaries": list(profile.recent_episode_summaries),
            "unresolved_hooks": list(profile.unresolved_hooks),
            "preferred_format": (
                profile.preferred_format.value
                if profile.preferred_format is not None
                else None
            ),
        }

    @staticmethod
    def _profile_from_row(row: SeriesRow) -> SeriesProfile:
        state = dict(row.state_json)
        raw_format = state.get("preferred_format")
        return SeriesProfile(
            id=row.id,
            title=str(state["title"]),
            status=SeriesStatus(row.status),
            current_episode=row.current_episode,
            prior_pack_ids=tuple(state.get("prior_pack_ids", ())),
            character_ids=tuple(state.get("character_ids", ())),
            continuity_state=dict(state.get("continuity_state", {})),
            bible=tuple(state.get("bible", ())),
            rolling_summary=str(state.get("rolling_summary", "")),
            recent_episode_summaries=tuple(
                state.get("recent_episode_summaries", ())
            ),
            unresolved_hooks=tuple(state.get("unresolved_hooks", ())),
            preferred_format=PackFormat(raw_format) if raw_format else None,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
