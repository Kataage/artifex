from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from artifex.characters import CharacterRegistry
from artifex.config.models import CharacterRegistryConfig, LoRARegistryConfig
from artifex.domain import CharacterOutfit, CharacterProfile, LoRAPolicy, LoRAProfile
from artifex.loras.registry import LoRARegistry


class LoRAResolutionError(RuntimeError):
    def __init__(self, issues: tuple[str, ...]) -> None:
        super().__init__("; ".join(issues))
        self.issues = issues


class LoRAPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lora_id: str
    path: str
    asset_name: str | None = None
    weight: float
    character_ids: tuple[str, ...] = ()
    trigger_tags: tuple[str, ...] = ()
    layer: str = "character"


class LoRAPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_family: str
    character_ids: tuple[str, ...]
    entries: tuple[LoRAPlanEntry, ...]
    warnings: tuple[str, ...] = ()


def _normalized(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", " ", value.casefold()).strip()


def _matches_outfit(clothing: str, outfit: CharacterOutfit) -> int:
    haystack = f" {_normalized(clothing)} "
    terms = (
        outfit.id,
        outfit.display_name,
        *outfit.aliases,
        *outfit.canonical_tags,
    )
    return max(
        (
            len(normalized)
            for term in terms
            if (normalized := _normalized(term))
            and f" {normalized} " in haystack
        ),
        default=0,
    )


class LoRAResolver:
    LAYER_ORDER: ClassVar[dict[str, int]] = {
        "character": 0,
        "outfit": 1,
        "style": 2,
        "utility": 3,
        "other": 4,
    }

    def __init__(
        self,
        characters: CharacterRegistry,
        loras: LoRARegistry,
        character_config: CharacterRegistryConfig,
        lora_config: LoRARegistryConfig,
    ) -> None:
        self._characters = characters
        self._loras = loras
        self._character_config = character_config
        self._lora_config = lora_config

    def resolve(
        self,
        character_ids: tuple[str, ...],
        *,
        model_family: str,
        exclude_lora_ids: tuple[str, ...] = (),
        clothing: str | None = None,
        style_lora_ids: tuple[str, ...] = (),
        utility_lora_ids: tuple[str, ...] = (),
    ) -> LoRAPlan:
        if not character_ids:
            raise LoRAResolutionError(("scene contains no characters",))
        if len(character_ids) != len(set(character_ids)):
            raise LoRAResolutionError(("scene contains duplicate character ids",))

        issues: list[str] = []
        warnings: list[str] = []
        characters: list[CharacterProfile] = []
        excluded = frozenset(exclude_lora_ids)

        for character_id in character_ids:
            character_profile = self._characters.get(character_id)
            if character_profile is None:
                issues.append(f"unknown character: {character_id}")
                continue
            characters.append(character_profile)
            if not character_profile.enabled:
                issues.append(f"character is disabled: {character_id}")
            if character_profile.readiness < self._character_config.minimum_readiness:
                issues.append(
                    f"character readiness below threshold: {character_id} "
                    f"({character_profile.readiness:.3f})"
                )
            if (
                character_profile.model_families
                and model_family not in character_profile.model_families
            ):
                issues.append(
                    f"character {character_id} is not configured for {model_family}"
                )

        if issues:
            raise LoRAResolutionError(tuple(issues))

        selected: dict[str, tuple[LoRAProfile, list[str], str, int]] = {}

        def add(
            profile: LoRAProfile,
            *,
            layer: str,
            related_character_ids: Iterable[str],
            order: int = 0,
        ) -> None:
            related = list(dict.fromkeys(related_character_ids))
            existing = selected.get(profile.id)
            if existing is None:
                selected[profile.id] = (profile, related, layer, order)
                return
            for character_id in related:
                if character_id not in existing[1]:
                    existing[1].append(character_id)

        for character in characters:
            profile = self._select_for_character(
                character,
                model_family,
                excluded=excluded,
            )
            if profile is None:
                if character.lora_policy is LoRAPolicy.REQUIRED:
                    issues.append(
                        f"required production LoRA unavailable for {character.id}"
                    )
                elif character.lora_policy is LoRAPolicy.PREFERRED:
                    warnings.append(
                        f"preferred LoRA unavailable for {character.id}; using base model"
                    )
            else:
                add(
                    profile,
                    layer="character",
                    related_character_ids=(character.id,),
                )

            if clothing:
                outfit = self._select_outfit(character, clothing)
                if outfit is not None and outfit.preferred_lora_ids:
                    outfit_profile = self._first_available_by_id(
                        outfit.preferred_lora_ids,
                        model_family=model_family,
                        allowed_types={"outfit", "other"},
                        scene_character_ids=character_ids,
                        excluded=excluded,
                    )
                    if outfit_profile is None:
                        warnings.append(
                            f"outfit LoRA unavailable for {character.id}/{outfit.id}"
                        )
                    else:
                        add(
                            outfit_profile,
                            layer="outfit",
                            related_character_ids=(character.id,),
                        )

        self._add_explicit_layer(
            selected,
            issues,
            warnings,
            ids=(
                *self._lora_config.default_style_lora_ids,
                *style_lora_ids,
            ),
            layer="style",
            allowed_types={"style", "other"},
            model_family=model_family,
            character_ids=character_ids,
            excluded=excluded,
        )
        self._add_explicit_layer(
            selected,
            issues,
            warnings,
            ids=(
                *self._lora_config.default_utility_lora_ids,
                *utility_lora_ids,
            ),
            layer="utility",
            allowed_types={"utility", "other"},
            model_family=model_family,
            character_ids=character_ids,
            excluded=excluded,
        )

        if issues:
            raise LoRAResolutionError(tuple(issues))

        selected_profiles = [item[0] for item in selected.values()]
        selected_ids = {profile.id for profile in selected_profiles}
        for profile in selected_profiles:
            conflicts = selected_ids.intersection(profile.incompatible_lora_ids)
            conflicts.discard(profile.id)
            if conflicts:
                issues.append(
                    f"LoRA {profile.id} conflicts with "
                    + ", ".join(sorted(conflicts))
                )

        character_count = sum(
            1 for _, _, layer, _ in selected.values() if layer == "character"
        )
        if character_count > self._lora_config.maximum_character_loras_per_scene:
            issues.append(
                "selected character LoRA count exceeds configured scene maximum: "
                f"{character_count} > "
                f"{self._lora_config.maximum_character_loras_per_scene}"
            )
        if len(selected) > self._lora_config.maximum_total_loras_per_scene:
            issues.append(
                "selected total LoRA count exceeds configured scene maximum: "
                f"{len(selected)} > "
                f"{self._lora_config.maximum_total_loras_per_scene}"
            )

        if issues:
            raise LoRAResolutionError(tuple(issues))

        ordered = sorted(
            selected.values(),
            key=lambda item: (
                self.LAYER_ORDER.get(item[2], 99),
                item[3],
                item[0].id,
            ),
        )
        entries = tuple(
            LoRAPlanEntry(
                lora_id=profile.id,
                path=str(profile.path),
                asset_name=(
                    str(profile.metadata.get("asset_name"))
                    if profile.metadata.get("asset_name")
                    else Path(profile.path).name
                ),
                weight=self._effective_weight(profile),
                character_ids=tuple(character_list),
                trigger_tags=profile.trigger_tags,
                layer=layer,
            )
            for profile, character_list, layer, _ in ordered
        )
        return LoRAPlan(
            model_family=model_family,
            character_ids=character_ids,
            entries=entries,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def adjust_weights(
        self,
        plan: LoRAPlan,
        *,
        delta: float,
    ) -> LoRAPlan:
        entries: list[LoRAPlanEntry] = []
        for entry in plan.entries:
            profile = self._loras.require(entry.lora_id)
            lower = (
                profile.validated_min_weight
                if profile.validated_min_weight is not None
                else 0.0
            )
            upper = (
                profile.validated_max_weight
                if profile.validated_max_weight is not None
                else 2.0
            )
            weight = min(upper, max(lower, entry.weight + delta))
            entries.append(entry.model_copy(update={"weight": weight}))
        return plan.model_copy(update={"entries": tuple(entries)})

    def _select_for_character(
        self,
        character: CharacterProfile,
        model_family: str,
        *,
        excluded: frozenset[str] = frozenset(),
    ) -> LoRAProfile | None:
        if character.lora_policy is LoRAPolicy.NONE:
            return None

        candidates = [
            candidate
            for candidate in self._loras.for_character(
                character.id,
                model_family=model_family,
                production_only=True,
            )
            if candidate.id not in excluded
            and candidate.lora_type in {"character", "other"}
        ]
        preferred_order = {
            lora_id: index for index, lora_id in enumerate(character.preferred_lora_ids)
        }

        if character.lora_policy is LoRAPolicy.OPTIONAL:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.id in preferred_order
            ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda candidate: (
                preferred_order.get(candidate.id, len(preferred_order) + 1),
                -candidate.readiness,
                -(candidate.identity_score or 0.0),
                candidate.id,
            )
        )
        return candidates[0]

    @staticmethod
    def _select_outfit(
        character: CharacterProfile,
        clothing: str,
    ) -> CharacterOutfit | None:
        matches = [
            (_matches_outfit(clothing, outfit), outfit.id, outfit)
            for outfit in character.outfits
        ]
        matches = [item for item in matches if item[0] > 0]
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], item[1]))
        return matches[0][2]

    def _first_available_by_id(
        self,
        ids: Iterable[str],
        *,
        model_family: str,
        allowed_types: set[str],
        scene_character_ids: tuple[str, ...],
        excluded: frozenset[str],
    ) -> LoRAProfile | None:
        for lora_id in ids:
            if lora_id in excluded:
                continue
            profile = self._loras.get(lora_id)
            if profile is None or profile.state.value != "production":
                continue
            if profile.lora_type not in allowed_types:
                continue
            if profile.model_families and model_family not in profile.model_families:
                continue
            if (
                profile.target_character_ids
                and not set(profile.target_character_ids).intersection(
                    scene_character_ids
                )
            ):
                continue
            return profile
        return None

    def _add_explicit_layer(
        self,
        selected: dict[str, tuple[LoRAProfile, list[str], str, int]],
        issues: list[str],
        warnings: list[str],
        *,
        ids: Iterable[str],
        layer: str,
        allowed_types: set[str],
        model_family: str,
        character_ids: tuple[str, ...],
        excluded: frozenset[str],
    ) -> None:
        for order, lora_id in enumerate(dict.fromkeys(ids)):
            if lora_id in excluded:
                continue
            profile = self._loras.get(lora_id)
            if profile is None:
                warnings.append(f"configured {layer} LoRA is unknown: {lora_id}")
                continue
            if profile.state.value != "production":
                warnings.append(
                    f"configured {layer} LoRA is not production-ready: {lora_id}"
                )
                continue
            if profile.lora_type not in allowed_types:
                issues.append(
                    f"configured {layer} LoRA has incompatible type: "
                    f"{lora_id} ({profile.lora_type})"
                )
                continue
            if profile.model_families and model_family not in profile.model_families:
                issues.append(
                    f"configured {layer} LoRA {lora_id} does not support "
                    f"{model_family}"
                )
                continue
            if (
                profile.target_character_ids
                and not set(profile.target_character_ids).intersection(character_ids)
            ):
                issues.append(
                    f"configured {layer} LoRA {lora_id} targets unrelated characters"
                )
                continue
            existing = selected.get(profile.id)
            if existing is None:
                selected[profile.id] = (
                    profile,
                    list(character_ids),
                    layer,
                    order,
                )

    @staticmethod
    def _effective_weight(profile: LoRAProfile) -> float:
        weight = profile.recommended_weight
        if profile.validated_min_weight is not None:
            weight = max(weight, profile.validated_min_weight)
        if profile.validated_max_weight is not None:
            weight = min(weight, profile.validated_max_weight)
        return weight
