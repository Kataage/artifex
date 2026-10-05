from __future__ import annotations

from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from artifex.policy.models import RightsPolicyProfile


class RightsPolicyRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, RightsPolicyProfile] = {}

    def register(self, profile: RightsPolicyProfile) -> None:
        existing = self._profiles.get(profile.id)
        if existing is not None and existing.version != profile.version:
            raise ValueError(
                f"rights policy id already registered with another version: {profile.id}"
            )
        if existing is not None:
            raise ValueError(f"duplicate rights policy id: {profile.id}")
        self._profiles[profile.id] = profile

    def require(self, policy_id: str) -> RightsPolicyProfile:
        try:
            return self._profiles[policy_id]
        except KeyError as exc:
            raise KeyError(f"unknown rights policy: {policy_id}") from exc

    def get(self, policy_id: str) -> RightsPolicyProfile | None:
        return self._profiles.get(policy_id)

    def list(self) -> tuple[RightsPolicyProfile, ...]:
        return tuple(self._profiles[key] for key in sorted(self._profiles))

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
                for profile in self._profiles_from_text(
                    path.read_text(encoding="utf-8"),
                    source=str(path),
                ):
                    self.register(profile)
                    count += 1
        return count

    @classmethod
    def with_packaged_defaults(
        cls,
        directories: Iterable[Path] = (),
    ) -> RightsPolicyRegistry:
        registry = cls()
        root = files("artifex.policy").joinpath("profiles")
        text = root.joinpath("default.yaml").read_text(encoding="utf-8")
        for profile in cls._profiles_from_text(text, source="packaged:default.yaml"):
            registry.register(profile)
        registry.load_directories(directories)
        return registry

    @staticmethod
    def _profiles_from_text(
        text: str,
        *,
        source: str,
    ) -> tuple[RightsPolicyProfile, ...]:
        raw = yaml.safe_load(text)
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
            raise TypeError(f"invalid rights policy YAML root: {source}")
        try:
            return tuple(RightsPolicyProfile.model_validate(item) for item in items)
        except ValidationError as exc:
            raise ValueError(f"invalid rights policy in {source}: {exc}") from exc
