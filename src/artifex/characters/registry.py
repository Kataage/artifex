from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from sqlalchemy import select

from artifex.characters.matching import contains_alias
from artifex.db import Database
from artifex.db.models import CharacterRow
from artifex.domain import CharacterProfile


class CharacterRegistry:
    def __init__(self, database: Database) -> None:
        self._database = database

    def upsert(self, profile: CharacterProfile) -> None:
        payload = profile.model_dump(mode="json")
        with self._database.session() as session:
            row = session.get(CharacterRow, profile.id)
            if row is None:
                session.add(
                    CharacterRow(
                        id=profile.id,
                        display_name=profile.display_name,
                        namespace=profile.namespace,
                        enabled=profile.enabled,
                        lora_policy=profile.lora_policy.value,
                        readiness=profile.readiness,
                        profile_json=payload,
                        last_used_at=profile.last_used_at,
                        total_generated=profile.total_generated,
                    )
                )
                return
            row.display_name = profile.display_name
            row.namespace = profile.namespace
            row.enabled = profile.enabled
            row.lora_policy = profile.lora_policy.value
            row.readiness = profile.readiness
            row.profile_json = payload
            row.last_used_at = profile.last_used_at
            row.total_generated = profile.total_generated

    def get(self, character_id: str) -> CharacterProfile | None:
        with self._database.session() as session:
            row = session.get(CharacterRow, character_id)
            if row is None:
                return None
            return self._profile_from_row(row)

    def require(self, character_id: str) -> CharacterProfile:
        profile = self.get(character_id)
        if profile is None:
            raise KeyError(f"unknown character: {character_id}")
        return profile

    def list(
        self,
        *,
        enabled_only: bool = False,
        namespace: str | None = None,
    ) -> tuple[CharacterProfile, ...]:
        query = select(CharacterRow).order_by(CharacterRow.id.asc())
        if enabled_only:
            query = query.where(CharacterRow.enabled.is_(True))
        if namespace is not None:
            query = query.where(CharacterRow.namespace == namespace)

        with self._database.session() as session:
            rows = session.scalars(query).all()
            return tuple(self._profile_from_row(row) for row in rows)

    def load_directories(self, directories: Iterable[Path]) -> int:
        count = 0
        for directory in directories:
            if not directory.exists():
                continue
            paths = sorted(
                (
                    *directory.rglob("*.yaml"),
                    *directory.rglob("*.yml"),
                ),
                key=lambda path: path.as_posix().casefold(),
            )
            for path in paths:
                for profile in self._profiles_from_yaml(path):
                    self.upsert(profile)
                    count += 1
        return count

    def match_text(self, text: str) -> tuple[str, ...]:
        matches: list[tuple[int, str]] = []
        for profile in self.list(enabled_only=True):
            candidates = {
                profile.id,
                profile.display_name,
                *profile.aliases,
                *profile.canonical_tags,
            }
            best_length = max(
                (
                    len(alias)
                    for alias in candidates
                    if contains_alias(text, alias)
                ),
                default=0,
            )
            if best_length:
                matches.append((best_length, profile.id))
        matches.sort(key=lambda item: (-item[0], item[1]))
        return tuple(character_id for _, character_id in matches)

    def record_use(self, character_ids: Iterable[str]) -> None:
        now = datetime.now(UTC)
        ids = tuple(dict.fromkeys(character_ids))
        if not ids:
            return
        with self._database.session() as session:
            rows = session.scalars(
                select(CharacterRow).where(CharacterRow.id.in_(ids))
            ).all()
            found = {row.id for row in rows}
            missing = set(ids) - found
            if missing:
                raise KeyError("unknown characters: " + ", ".join(sorted(missing)))
            for row in rows:
                row.last_used_at = now
                row.total_generated += 1
                payload = dict(row.profile_json)
                payload["last_used_at"] = now.isoformat()
                payload["total_generated"] = row.total_generated
                row.profile_json = payload

    @staticmethod
    def _profile_from_row(row: CharacterRow) -> CharacterProfile:
        payload = dict(row.profile_json)
        payload.update(
            {
                "id": row.id,
                "display_name": row.display_name,
                "namespace": row.namespace,
                "enabled": row.enabled,
                "lora_policy": row.lora_policy,
                "readiness": row.readiness,
                "last_used_at": row.last_used_at,
                "total_generated": row.total_generated,
            }
        )
        return CharacterProfile.model_validate(payload)

    @staticmethod
    def _profiles_from_yaml(path: Path) -> tuple[CharacterProfile, ...]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            return ()
        items: list[dict[str, Any]]
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and isinstance(raw.get("characters"), list):
            items = raw["characters"]
        elif isinstance(raw, dict):
            items = [raw]
        else:
            raise TypeError(f"invalid character profile YAML root: {path}")

        try:
            return tuple(CharacterProfile.model_validate(item) for item in items)
        except ValidationError as exc:
            raise ValueError(f"invalid character profile in {path}: {exc}") from exc
