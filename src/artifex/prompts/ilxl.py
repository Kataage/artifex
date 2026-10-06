from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from artifex.domain import CharacterOutfit, CharacterProfile
from artifex.loras import LoRAPlan
from artifex.packs import ScenePlan
from artifex.prompts.lexicon import PromptProfile, ValidatedTagLexicon
from artifex.prompts.models import CompiledPrompt, PromptProvenance

_SPLIT = re.compile(r"[,;\n]+")
_SPACES = re.compile(r"\s+")
_LEADING_NEGATION = re.compile(r"^(?:no|not|avoid)\s+", re.IGNORECASE)


def _tag_key(tag: str) -> str:
    return _SPACES.sub(" ", tag.replace("_", " ").strip().casefold())


def _dedupe(tags: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = raw.strip()
        if not tag:
            continue
        key = _tag_key(tag)
        if key in seen:
            continue
        seen.add(key)
        result.append(tag)
    return tuple(result)


def _segments(value: str, *, negative: bool = False) -> tuple[str, ...]:
    result: list[str] = []
    for part in _SPLIT.split(value):
        cleaned = part.strip()
        if negative:
            cleaned = _LEADING_NEGATION.sub("", cleaned).strip()
        if cleaned:
            result.append(cleaned)
    return tuple(result)


def _outfit_match(
    character: CharacterProfile,
    clothing: str,
) -> CharacterOutfit | None:
    haystack = f" {_tag_key(clothing)} "
    matches: list[tuple[int, str, CharacterOutfit]] = []
    for outfit in character.outfits:
        terms = (
            outfit.id,
            outfit.display_name,
            *outfit.aliases,
            *outfit.canonical_tags,
        )
        best = 0
        for term in terms:
            needle = _tag_key(term)
            if needle and f" {needle} " in haystack:
                best = max(best, len(needle))
        if best:
            matches.append((best, outfit.id, outfit))
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1]))
    return matches[0][2]


class ILXLDanbooruAdapter:
    model_families = frozenset({"ilxl"})
    adapter_id = "ilxl_danbooru"
    adapter_version = "2"

    def __init__(
        self,
        *,
        lexicon: ValidatedTagLexicon | None = None,
        profile_id: str = "ilxl-danbooru-v2",
        checkpoint: str | None = None,
        checkpoint_profile_overrides: Mapping[str, str] | None = None,
        unresolved_policy: str = "report",
        extra_lexicon_paths: Sequence[Path] = (),
    ) -> None:
        current = lexicon or ValidatedTagLexicon.packaged()
        for path in extra_lexicon_paths:
            current = current.merge_file(path)
        if unresolved_policy not in {"report", "error"}:
            raise ValueError("unresolved_policy must be 'report' or 'error'")
        self._lexicon = current
        self._profile_id = profile_id
        self._checkpoint = checkpoint
        self._checkpoint_profile_overrides = dict(
            checkpoint_profile_overrides or {}
        )
        self._unresolved_policy = unresolved_policy

    @property
    def lexicon(self) -> ValidatedTagLexicon:
        return self._lexicon

    def _profile(self, model_family: str) -> PromptProfile:
        return self._lexicon.profile(
            self._profile_id,
            model_family=model_family,
            checkpoint=self._checkpoint,
            checkpoint_overrides=self._checkpoint_profile_overrides,
        )

    def compile(
        self,
        scene: ScenePlan,
        characters: Sequence[CharacterProfile],
        lora_plan: LoRAPlan,
    ) -> CompiledPrompt:
        profile = self._profile(lora_plan.model_family)
        source_categories: dict[str, str] = {}
        trusted_concepts: set[str] = set()
        positive: list[str] = []
        negative: list[str] = []
        generated_positive: list[str] = list(profile.required_tags)
        generated_negative: list[str] = []
        unresolved: list[str] = []
        forbidden: list[str] = [*profile.forbidden_tags]

        def add_trusted(
            tags: Iterable[str],
            *,
            destination: list[str],
            category: str,
        ) -> None:
            for tag in tags:
                cleaned = tag.strip()
                if not cleaned:
                    continue
                destination.append(cleaned)
                source_categories.setdefault(cleaned, category)

        for character in characters:
            trusted_concepts.update(
                _tag_key(value)
                for value in (
                    character.id,
                    character.display_name,
                    *character.aliases,
                    *character.canonical_tags,
                    *character.required_tags,
                )
                if value.strip()
            )
            add_trusted(
                (*character.canonical_tags, *character.required_tags),
                destination=positive,
                category="character",
            )
            forbidden.extend(character.forbidden_tags)
            outfit = _outfit_match(character, scene.visual.clothing)
            if outfit is not None:
                add_trusted(
                    (*outfit.canonical_tags, *outfit.required_tags),
                    destination=positive,
                    category="clothing",
                )
                forbidden.extend(outfit.forbidden_tags)

        trigger_tags = tuple(
            tag
            for entry in lora_plan.entries
            for tag in entry.trigger_tags
            if tag.strip()
        )
        trusted_concepts.update(_tag_key(tag) for tag in trigger_tags)
        add_trusted(
            trigger_tags,
            destination=positive,
            category="trigger",
        )

        fields = (
            ("composition", scene.visual.composition),
            ("camera", scene.visual.camera),
            ("pose", scene.visual.pose),
            ("expression", scene.visual.expression),
            ("clothing", scene.visual.clothing),
            ("setting", scene.visual.setting),
            ("lighting", scene.visual.lighting),
            ("atmosphere", scene.visual.atmosphere),
        )
        for field, value in fields:
            field_tags: list[str] = []
            for segment in _segments(value):
                resolved = self._lexicon.resolve(segment)
                if resolved.matched:
                    field_tags.extend(resolved.tags)
                else:
                    unresolved.append(f"{field}:{segment}")
            if not field_tags and value.strip() and field in profile.field_required:
                unresolved.append(f"{field}:{value.strip()}")
            generated_positive.extend(field_tags)

        for constraint in scene.visual.positive_constraints:
            for segment in _segments(constraint):
                resolved = self._lexicon.resolve(segment)
                if resolved.matched:
                    generated_positive.extend(resolved.tags)
                elif _tag_key(segment) not in trusted_concepts:
                    unresolved.append(f"positive_constraint:{segment}")

        for constraint in scene.visual.negative_constraints:
            for segment in _segments(constraint, negative=True):
                resolved = self._lexicon.resolve(segment)
                if resolved.matched:
                    generated_negative.extend(resolved.tags)
                else:
                    unresolved.append(f"negative_constraint:{segment}")

        self._lexicon.validate_generated(generated_positive)
        self._lexicon.validate_generated(generated_negative)

        generated_positive = list(
            self._lexicon.resolve_conflicts(
                self._lexicon.expand_implications(generated_positive)
            )
        )
        generated_negative = list(
            self._lexicon.resolve_conflicts(
                self._lexicon.expand_implications(generated_negative)
            )
        )

        positive.extend(generated_positive)
        negative.extend(generated_negative)
        add_trusted(
            forbidden,
            destination=negative,
            category="trusted",
        )

        forbidden_keys = {_tag_key(tag) for tag in forbidden}
        positive_tags = tuple(
            tag
            for tag in _dedupe(positive)
            if _tag_key(tag) not in forbidden_keys
        )
        positive_keys = {_tag_key(tag) for tag in positive_tags}
        negative_tags = tuple(
            tag
            for tag in _dedupe(negative)
            if _tag_key(tag) not in positive_keys
        )

        positive_tags = self._lexicon.order(
            positive_tags,
            profile=profile,
            category_overrides=source_categories,
        )
        negative_tags = self._lexicon.order(
            negative_tags,
            profile=profile,
            negative=True,
            category_overrides=source_categories,
        )

        unresolved_concepts = tuple(dict.fromkeys(unresolved))
        if unresolved_concepts and self._unresolved_policy == "error":
            raise ValueError(
                "unresolved prompt concepts: " + "; ".join(unresolved_concepts)
            )

        return CompiledPrompt(
            positive_prompt=", ".join(positive_tags),
            negative_prompt=", ".join(negative_tags),
            positive_tags=positive_tags,
            negative_tags=negative_tags,
            unresolved_concepts=unresolved_concepts,
            provenance=PromptProvenance(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                model_family=lora_plan.model_family,
                character_ids=tuple(character.id for character in characters),
                lora_ids=tuple(entry.lora_id for entry in lora_plan.entries),
                lora_weights=tuple(entry.weight for entry in lora_plan.entries),
                lexicon_id=self._lexicon.lexicon_id,
                lexicon_version=self._lexicon.version,
                prompt_profile_id=profile.id,
                checkpoint=self._checkpoint,
            ),
        )
