from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from artifex.characters.registry import CharacterRegistry
from artifex.domain import CharacterProfile


_CATALOG_FILENAME = "hololive_2026_10_06.json"


class RosterAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_id: str
    catalog_version: str
    expected_count: int
    loaded_count: int
    status_counts: dict[str, int]
    missing_ids: tuple[str, ...] = ()
    unexpected_ids: tuple[str, ...] = ()
    duplicate_profile_ids: tuple[str, ...] = ()
    excluded_namespace_ids: tuple[str, ...] = ()
    disabled_ids: tuple[str, ...] = ()
    unready_ids: tuple[str, ...] = ()
    missing_canonical_tag_ids: tuple[str, ...] = ()
    missing_policy_profile_ids: tuple[str, ...] = ()
    missing_provenance_ids: tuple[str, ...] = ()
    missing_reference_slot_ids: tuple[str, ...] = ()
    catalog_complete: bool
    production_ready: bool


class HololiveCatalog:
    def __init__(self, manifest: dict[str, Any]) -> None:
        if manifest.get("catalog_id") != "hololive":
            raise ValueError("unsupported character catalog")
        self.catalog_id = str(manifest["catalog_id"])
        self.catalog_version = str(manifest["catalog_version"])
        self.checked_at = str(manifest["checked_at"])
        self.scope = dict(manifest.get("scope", {}))

        defaults_raw = manifest.get("defaults")
        sources_raw = manifest.get("sources")
        characters_raw = manifest.get("characters")
        if not isinstance(defaults_raw, dict):
            raise ValueError("catalog defaults must be an object")
        if not isinstance(sources_raw, dict):
            raise ValueError("catalog sources must be an object")
        if not isinstance(characters_raw, list):
            raise ValueError("catalog characters must be a list")

        defaults = deepcopy(defaults_raw)
        reference_template = str(
            defaults.pop(
                "reference_image_dir_template",
                "references/characters/{id}",
            )
        )
        base_notes = tuple(str(note) for note in defaults.get("generation_notes", ()))
        profiles: list[CharacterProfile] = []

        for item_raw in characters_raw:
            if not isinstance(item_raw, dict):
                raise ValueError("catalog character entry must be an object")
            item = deepcopy(item_raw)
            source_refs = tuple(str(ref) for ref in item.pop("source_refs", ()))
            extra_notes = tuple(
                str(note) for note in item.pop("generation_notes_extra", ())
            )

            payload = deepcopy(defaults)
            payload.update(item)
            character_id = str(payload["id"])
            display_name = str(payload["display_name"])
            aliases = tuple(
                dict.fromkeys(
                    (
                        *(str(alias) for alias in payload.get("aliases", ())),
                        display_name,
                    )
                )
            )
            payload["aliases"] = aliases
            payload["generation_notes"] = (*base_notes, *extra_notes)
            payload["reference_image_dirs"] = (
                reference_template.format(id=character_id),
            )

            provenance: list[dict[str, Any]] = []
            for source_ref in source_refs:
                source = sources_raw.get(source_ref)
                if not isinstance(source, dict):
                    raise ValueError(
                        f"unknown catalog source {source_ref!r} for {character_id}"
                    )
                provenance.append(deepcopy(source))
            payload["provenance"] = provenance
            profiles.append(CharacterProfile.model_validate(payload))

        ids = [profile.id for profile in profiles]
        duplicates = sorted(
            character_id
            for character_id, count in Counter(ids).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(
                "duplicate character ids in catalog: " + ", ".join(duplicates)
            )
        self._profiles = tuple(profiles)

    @classmethod
    def packaged(cls) -> HololiveCatalog:
        resource = (
            files("artifex.characters")
            .joinpath("catalogs")
            .joinpath(_CATALOG_FILENAME)
        )
        raw = json.loads(resource.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("packaged Hololive catalog root must be an object")
        return cls(raw)

    @property
    def profiles(self) -> tuple[CharacterProfile, ...]:
        return self._profiles

    def materialize(self, destination: Path, *, overwrite: bool = False) -> Path:
        destination = destination.expanduser()
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --force to replace it"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "checked_at": self.checked_at,
            "characters": [
                profile.model_dump(mode="json", exclude_none=True)
                for profile in self._profiles
            ],
        }
        destination.write_text(
            yaml.safe_dump(
                payload,
                allow_unicode=True,
                sort_keys=False,
                width=100,
            ),
            encoding="utf-8",
        )
        return destination

    def audit(
        self,
        registry: CharacterRegistry,
        profile_dirs: tuple[Path, ...],
        *,
        minimum_readiness: float = 0.0,
    ) -> RosterAuditReport:
        expected = {profile.id: profile for profile in self._profiles}
        loaded = registry.list()
        loaded_by_id = {profile.id: profile for profile in loaded}
        hololive_loaded = {
            profile.id: profile
            for profile in loaded
            if profile.namespace.casefold() == "hololive"
        }

        missing_ids = tuple(sorted(set(expected) - set(loaded_by_id)))
        unexpected_ids = tuple(sorted(set(hololive_loaded) - set(expected)))
        duplicate_profile_ids = _duplicate_profile_ids(profile_dirs)

        excluded_namespace_ids = tuple(
            sorted(
                profile.id
                for profile in loaded
                if "holostars"
                in " ".join(
                    (
                        profile.namespace,
                        profile.branch or "",
                        profile.group or "",
                    )
                ).casefold()
            )
        )

        expected_loaded = tuple(
            loaded_by_id[character_id]
            for character_id in sorted(expected)
            if character_id in loaded_by_id
        )
        disabled_ids = tuple(
            profile.id for profile in expected_loaded if not profile.enabled
        )
        unready_ids = tuple(
            profile.id
            for profile in expected_loaded
            if profile.readiness < minimum_readiness
        )
        missing_canonical_tag_ids = tuple(
            profile.id
            for profile in expected_loaded
            if not profile.canonical_tags
        )
        missing_policy_profile_ids = tuple(
            profile.id
            for profile in expected_loaded
            if not profile.policy_profile
        )
        missing_provenance_ids = tuple(
            profile.id
            for profile in expected_loaded
            if not profile.provenance
        )
        missing_reference_slot_ids = tuple(
            profile.id
            for profile in expected_loaded
            if not profile.reference_image_dirs
        )
        status_counts = dict(
            sorted(Counter(profile.status.value for profile in self._profiles).items())
        )

        structural_failures = (
            missing_ids,
            unexpected_ids,
            duplicate_profile_ids,
            excluded_namespace_ids,
            missing_canonical_tag_ids,
            missing_policy_profile_ids,
            missing_provenance_ids,
            missing_reference_slot_ids,
        )
        catalog_complete = not any(structural_failures)
        production_ready = (
            catalog_complete
            and not disabled_ids
            and not unready_ids
        )

        return RosterAuditReport(
            catalog_id=self.catalog_id,
            catalog_version=self.catalog_version,
            expected_count=len(expected),
            loaded_count=len(hololive_loaded),
            status_counts=status_counts,
            missing_ids=missing_ids,
            unexpected_ids=unexpected_ids,
            duplicate_profile_ids=duplicate_profile_ids,
            excluded_namespace_ids=excluded_namespace_ids,
            disabled_ids=disabled_ids,
            unready_ids=unready_ids,
            missing_canonical_tag_ids=missing_canonical_tag_ids,
            missing_policy_profile_ids=missing_policy_profile_ids,
            missing_provenance_ids=missing_provenance_ids,
            missing_reference_slot_ids=missing_reference_slot_ids,
            catalog_complete=catalog_complete,
            production_ready=production_ready,
        )


def _duplicate_profile_ids(profile_dirs: tuple[Path, ...]) -> tuple[str, ...]:
    ids: list[str] = []
    for directory in profile_dirs:
        if not directory.exists():
            continue
        paths = sorted(
            (*directory.rglob("*.yaml"), *directory.rglob("*.yml")),
            key=lambda path: path.as_posix().casefold(),
        )
        for path in paths:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if raw is None:
                continue
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, dict) and isinstance(raw.get("characters"), list):
                items = raw["characters"]
            elif isinstance(raw, dict):
                items = [raw]
            else:
                continue
            for item in items:
                if isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))
    return tuple(
        sorted(
            character_id
            for character_id, count in Counter(ids).items()
            if count > 1
        )
    )
