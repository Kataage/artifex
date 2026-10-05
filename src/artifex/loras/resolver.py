from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from artifex.characters import CharacterRegistry
from artifex.config.models import CharacterRegistryConfig, LoRARegistryConfig
from artifex.domain import CharacterProfile, LoRAPolicy, LoRAProfile
from artifex.loras.registry import LoRARegistry


class LoRAResolutionError(RuntimeError):
    def __init__(self, issues: tuple[str, ...]) -> None:
        super().__init__("; ".join(issues))
        self.issues = issues


class LoRAPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lora_id: str
    path: str
    weight: float
    character_ids: tuple[str, ...] = Field(min_length=1)
    trigger_tags: tuple[str, ...] = ()


class LoRAPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_family: str
    character_ids: tuple[str, ...]
    entries: tuple[LoRAPlanEntry, ...]
    warnings: tuple[str, ...] = ()


class LoRAResolver:
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
    ) -> LoRAPlan:
        if not character_ids:
            raise LoRAResolutionError(("scene contains no characters",))
        if len(character_ids) != len(set(character_ids)):
            raise LoRAResolutionError(("scene contains duplicate character ids",))

        issues: list[str] = []
        warnings: list[str] = []
        characters: list[CharacterProfile] = []

        for character_id in character_ids:
            profile = self._characters.get(character_id)
            if profile is None:
                issues.append(f"unknown character: {character_id}")
                continue
            characters.append(profile)
            if not profile.enabled:
                issues.append(f"character is disabled: {character_id}")
            if profile.readiness < self._character_config.minimum_readiness:
                issues.append(
                    f"character readiness below threshold: {character_id} "
                    f"({profile.readiness:.3f})"
                )
            if profile.model_families and model_family not in profile.model_families:
                issues.append(
                    f"character {character_id} is not configured for {model_family}"
                )

        if issues:
            raise LoRAResolutionError(tuple(issues))

        selected: dict[str, tuple[LoRAProfile, list[str]]] = {}
        for character in characters:
            profile = self._select_for_character(character, model_family)
            if profile is None:
                if character.lora_policy is LoRAPolicy.REQUIRED:
                    issues.append(
                        f"required production LoRA unavailable for {character.id}"
                    )
                elif character.lora_policy is LoRAPolicy.PREFERRED:
                    warnings.append(
                        f"preferred LoRA unavailable for {character.id}; using base model"
                    )
                continue
            if profile.id in selected:
                selected[profile.id][1].append(character.id)
            else:
                selected[profile.id] = (profile, [character.id])

        if issues:
            raise LoRAResolutionError(tuple(issues))

        selected_profiles = [item[0] for item in selected.values()]
        selected_ids = {profile.id for profile in selected_profiles}
        for profile in selected_profiles:
            conflicts = selected_ids.intersection(profile.incompatible_lora_ids)
            if conflicts:
                issues.append(
                    f"LoRA {profile.id} conflicts with "
                    + ", ".join(sorted(conflicts))
                )

        if len(selected_profiles) > self._lora_config.maximum_character_loras_per_scene:
            issues.append(
                "selected character LoRA count exceeds configured scene maximum: "
                f"{len(selected_profiles)} > "
                f"{self._lora_config.maximum_character_loras_per_scene}"
            )

        if issues:
            raise LoRAResolutionError(tuple(issues))

        entries = tuple(
            LoRAPlanEntry(
                lora_id=profile.id,
                path=str(profile.path),
                weight=self._effective_weight(profile),
                character_ids=tuple(character_list),
                trigger_tags=profile.trigger_tags,
            )
            for profile, character_list in sorted(
                selected.values(),
                key=lambda item: item[0].id,
            )
        )
        return LoRAPlan(
            model_family=model_family,
            character_ids=character_ids,
            entries=entries,
            warnings=tuple(warnings),
        )

    def _select_for_character(
        self,
        character: CharacterProfile,
        model_family: str,
    ) -> LoRAProfile | None:
        if character.lora_policy is LoRAPolicy.NONE:
            return None

        candidates = list(
            self._loras.for_character(
                character.id,
                model_family=model_family,
                production_only=True,
            )
        )
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
    def _effective_weight(profile: LoRAProfile) -> float:
        weight = profile.recommended_weight
        if profile.validated_min_weight is not None:
            weight = max(weight, profile.validated_min_weight)
        if profile.validated_max_weight is not None:
            weight = min(weight, profile.validated_max_weight)
        return weight
