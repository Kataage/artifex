from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from artifex.domain import CharacterProfile
from artifex.loras import LoRAPlan
from artifex.packs import ScenePlan
from artifex.prompts.models import CompiledPrompt, PromptProvenance

_SPLIT = re.compile(r"[,;\n]+")
_SPACES = re.compile(r"\s+")
_NON_TAG = re.compile(r"[^0-9a-zA-Z_()'\-:.]+")
_LEADING_NEGATION = re.compile(r"^(?:no|not|avoid)\s+", re.IGNORECASE)

_PHRASE_MAP = {
    "full body": "full_body",
    "upper body": "upper_body",
    "cowboy shot": "cowboy_shot",
    "close up": "close-up",
    "close-up": "close-up",
    "low angle": "from_below",
    "high angle": "from_above",
    "from below": "from_below",
    "from above": "from_above",
    "eye level": "straight-on",
    "looking at viewer": "looking_at_viewer",
    "soft smile": "smile",
    "city street": "street",
    "open window": "window",
    "sunset light": "sunset",
    "sunset rim light": "rim_light",
}

_CONFLICT_GROUPS = (
    frozenset({"from_above", "from_below", "straight-on"}),
    frozenset({"close-up", "upper_body", "cowboy_shot", "full_body"}),
    frozenset({"standing", "sitting", "lying"}),
    frozenset({"smile", "frown", "crying"}),
)

_CONFLICT_LOOKUP = {
    tag: group
    for group in _CONFLICT_GROUPS
    for tag in group
}


def _tag_key(tag: str) -> str:
    return _SPACES.sub(" ", tag.replace("_", " ").strip().casefold())


def _normalize_generated_tag(value: str) -> str:
    value = _SPACES.sub(" ", value.strip().casefold())
    mapped = _PHRASE_MAP.get(value)
    if mapped is not None:
        return mapped
    value = value.replace(" ", "_")
    value = _NON_TAG.sub("", value)
    return value.strip("_")


def _split_generated(value: str) -> tuple[str, ...]:
    result: list[str] = []
    for part in _SPLIT.split(value):
        tag = _normalize_generated_tag(part)
        if tag:
            result.append(tag)
    return tuple(result)


def _split_constraint(value: str, *, negative: bool) -> tuple[str, ...]:
    result: list[str] = []
    for part in _SPLIT.split(value):
        cleaned = part.strip()
        if negative:
            cleaned = _LEADING_NEGATION.sub("", cleaned)
        tag = _normalize_generated_tag(cleaned)
        if tag:
            result.append(tag)
    return tuple(result)


def _dedupe_and_resolve(tags: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    claimed_groups: set[frozenset[str]] = set()

    for raw in tags:
        tag = raw.strip()
        if not tag:
            continue
        key = _tag_key(tag)
        if key in seen:
            continue

        conflict_group = _CONFLICT_LOOKUP.get(tag.casefold())
        if conflict_group is not None:
            if conflict_group in claimed_groups:
                continue
            claimed_groups.add(conflict_group)

        seen.add(key)
        result.append(tag)
    return tuple(result)


class ILXLDanbooruAdapter:
    model_families = frozenset({"ilxl"})
    adapter_id = "ilxl_danbooru"
    adapter_version = "1"

    def compile(
        self,
        scene: ScenePlan,
        characters: Sequence[CharacterProfile],
        lora_plan: LoRAPlan,
    ) -> CompiledPrompt:
        canonical_tags = tuple(
            tag
            for character in characters
            for tag in character.canonical_tags
            if tag.strip()
        )
        trigger_tags = tuple(
            tag
            for entry in lora_plan.entries
            for tag in entry.trigger_tags
            if tag.strip()
        )

        visual = scene.visual
        ordered_generated = (
            *_split_generated(visual.composition),
            *_split_generated(visual.camera),
            *_split_generated(visual.pose),
            *_split_generated(visual.expression),
            *_split_generated(visual.clothing),
            *_split_generated(visual.setting),
            *_split_generated(visual.lighting),
            *_split_generated(visual.atmosphere),
            *(
                tag
                for constraint in visual.positive_constraints
                for tag in _split_constraint(constraint, negative=False)
            ),
        )

        positive_tags = _dedupe_and_resolve(
            (*canonical_tags, *trigger_tags, *ordered_generated)
        )
        positive_keys = {_tag_key(tag) for tag in positive_tags}

        negative_candidates = tuple(
            tag
            for constraint in visual.negative_constraints
            for tag in _split_constraint(constraint, negative=True)
        )
        negative_tags = tuple(
            tag
            for tag in _dedupe_and_resolve(negative_candidates)
            if _tag_key(tag) not in positive_keys
        )

        return CompiledPrompt(
            positive_prompt=", ".join(positive_tags),
            negative_prompt=", ".join(negative_tags),
            positive_tags=positive_tags,
            negative_tags=negative_tags,
            provenance=PromptProvenance(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                model_family=lora_plan.model_family,
                character_ids=tuple(character.id for character in characters),
                lora_ids=tuple(entry.lora_id for entry in lora_plan.entries),
                lora_weights=tuple(entry.weight for entry in lora_plan.entries),
            ),
        )
