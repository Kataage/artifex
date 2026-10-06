from __future__ import annotations

import json
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from artifex.policy.models import PolicyProfile


class PolicyRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, PolicyProfile] = {}

    def register(self, profile: PolicyProfile) -> None:
        existing = self._profiles.get(profile.id)
        if existing is not None and existing.version == profile.version:
            raise ValueError(
                f"duplicate policy profile/version: {profile.id}@{profile.version}"
            )
        self._profiles[profile.id] = profile

    def get(self, profile_id: str) -> PolicyProfile | None:
        return self._profiles.get(profile_id)

    def require(self, profile_id: str) -> PolicyProfile:
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(f"unknown policy profile: {profile_id}")
        return profile

    def load_directories(self, directories: Iterable[Path]) -> int:
        count = 0
        for directory in directories:
            if not directory.exists():
                continue
            paths = sorted(
                (*directory.rglob("*.yaml"), *directory.rglob("*.yml")),
                key=lambda path: path.as_posix().casefold(),
            )
            for path in paths:
                for profile in self._profiles_from_yaml(path):
                    self.register(profile)
                    count += 1
        return count

    @classmethod
    def with_packaged_defaults(cls) -> PolicyRegistry:
        registry = cls()
        root = files("artifex.policy").joinpath("profiles")
        for filename in ("unconfigured.json", "patreon_2026_10.json"):
            raw = json.loads(root.joinpath(filename).read_text(encoding="utf-8"))
            registry.register(PolicyProfile.model_validate(raw))
        return registry

    @staticmethod
    def _profiles_from_yaml(path: Path) -> tuple[PolicyProfile, ...]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            return ()
        items: list[dict[str, Any]]
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and isinstance(raw.get("policies"), list):
            items = raw["policies"]
        elif isinstance(raw, dict):
            items = [raw]
        else:
            raise TypeError(f"invalid policy YAML root: {path}")

        try:
            return tuple(PolicyProfile.model_validate(item) for item in items)
        except ValidationError as exc:
            raise ValueError(f"invalid policy profile in {path}: {exc}") from exc
