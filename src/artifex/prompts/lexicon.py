from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

_WORDS = re.compile(r"[0-9a-zA-Z_+:'()\-]+")


def _phrase_key(value: str) -> str:
    return " ".join(
        token.casefold().replace("_", " ")
        for token in value.strip().split()
        if token.strip()
    )


class LexiconModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LexiconTag(LexiconModel):
    tag: str = Field(min_length=1)
    category: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    implies: tuple[str, ...] = ()


class PromptProfile(LexiconModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    model_families: tuple[str, ...] = ("ilxl",)
    checkpoint_globs: tuple[str, ...] = ()
    positive_category_order: tuple[str, ...]
    negative_category_order: tuple[str, ...]
    required_tags: tuple[str, ...] = ()
    forbidden_tags: tuple[str, ...] = ()
    field_required: tuple[str, ...] = ("composition", "camera", "pose", "expression")


class PromptLexiconDocument(LexiconModel):
    lexicon_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    tags: tuple[LexiconTag, ...]
    phrase_aliases: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    conflict_groups: tuple[tuple[str, ...], ...] = ()
    profiles: tuple[PromptProfile, ...]

    @model_validator(mode="after")
    def validate_references(self) -> PromptLexiconDocument:
        tags = {item.tag for item in self.tags}
        if len(tags) != len(self.tags):
            raise ValueError("prompt lexicon canonical tags must be unique")
        for item in self.tags:
            missing = set(item.implies) - tags
            if missing:
                raise ValueError(
                    f"tag {item.tag} implies unknown tags: {', '.join(sorted(missing))}"
                )
        for phrase, values in self.phrase_aliases.items():
            if not phrase.strip():
                raise ValueError("phrase alias key must be non-empty")
            missing = set(values) - tags
            if missing:
                raise ValueError(
                    f"phrase alias {phrase!r} references unknown tags: "
                    + ", ".join(sorted(missing))
                )
        for group in self.conflict_groups:
            missing = set(group) - tags
            if missing:
                raise ValueError(
                    "conflict group references unknown tags: "
                    + ", ".join(sorted(missing))
                )
        profile_ids = [profile.id for profile in self.profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("prompt profile ids must be unique")
        for profile in self.profiles:
            missing = (set(profile.required_tags) | set(profile.forbidden_tags)) - tags
            if missing:
                raise ValueError(
                    f"profile {profile.id} references unknown tags: "
                    + ", ".join(sorted(missing))
                )
        return self


@dataclass(frozen=True, slots=True)
class ResolvedConcept:
    tags: tuple[str, ...]
    matched: bool
    source_phrase: str


class ValidatedTagLexicon:
    def __init__(self, document: PromptLexiconDocument) -> None:
        self.document = document
        self._tags = {item.tag: item for item in document.tags}
        alias_map: dict[str, tuple[str, ...]] = {}
        for item in document.tags:
            alias_map[_phrase_key(item.tag)] = (item.tag,)
            for alias in item.aliases:
                alias_map[_phrase_key(alias)] = (item.tag,)
        for phrase, tags in document.phrase_aliases.items():
            alias_map[_phrase_key(phrase)] = tuple(tags)
        self._aliases = alias_map
        self._profiles = {profile.id: profile for profile in document.profiles}
        self._conflict_lookup: dict[str, frozenset[str]] = {}
        for group in document.conflict_groups:
            frozen = frozenset(group)
            for tag in group:
                self._conflict_lookup[tag] = frozen

    @classmethod
    def packaged(cls) -> ValidatedTagLexicon:
        package = resources.files("artifex.prompts").joinpath(
            "profiles/ilxl_danbooru_2026_10.json"
        )
        raw = json.loads(package.read_text(encoding="utf-8"))
        return cls(PromptLexiconDocument.model_validate(raw))

    def merge_file(self, path: Path) -> ValidatedTagLexicon:
        raw = json.loads(path.read_text(encoding="utf-8"))
        extra = PromptLexiconDocument.model_validate(raw)
        combined_tags = {item.tag: item for item in self.document.tags}
        combined_tags.update({item.tag: item for item in extra.tags})
        phrases = dict(self.document.phrase_aliases)
        phrases.update(extra.phrase_aliases)
        profiles = {item.id: item for item in self.document.profiles}
        profiles.update({item.id: item for item in extra.profiles})
        document = PromptLexiconDocument(
            lexicon_id=f"{self.document.lexicon_id}+{extra.lexicon_id}",
            version=f"{self.document.version}+{extra.version}",
            tags=tuple(combined_tags.values()),
            phrase_aliases=phrases,
            conflict_groups=(
                *self.document.conflict_groups,
                *extra.conflict_groups,
            ),
            profiles=tuple(profiles.values()),
        )
        return ValidatedTagLexicon(document)

    @property
    def lexicon_id(self) -> str:
        return self.document.lexicon_id

    @property
    def version(self) -> str:
        return self.document.version

    def is_valid(self, tag: str) -> bool:
        return tag in self._tags

    def category(self, tag: str) -> str:
        item = self._tags.get(tag)
        return item.category if item is not None else "trusted"

    def resolve(self, value: str) -> ResolvedConcept:
        key = _phrase_key(value)
        exact = self._aliases.get(key)
        if exact is not None:
            return ResolvedConcept(tags=exact, matched=True, source_phrase=value)

        # Conservative fallback: find only explicitly registered aliases that occur
        # as complete token sequences inside the prose. Unknown words are never
        # converted into guessed tags.
        normalized = f" {key} "
        matches: list[tuple[int, str, tuple[str, ...]]] = []
        for phrase, tags in self._aliases.items():
            if len(phrase) < 3:
                continue
            needle = f" {phrase} "
            if needle in normalized:
                matches.append((len(phrase.split()), phrase, tags))
        matches.sort(key=lambda item: (-item[0], item[1]))

        chosen: list[str] = []
        covered_phrases: list[str] = []
        occupied: set[str] = set()
        for _, phrase, tags in matches:
            phrase_tokens = set(phrase.split())
            if phrase_tokens <= occupied:
                continue
            occupied.update(phrase_tokens)
            covered_phrases.append(phrase)
            for tag in tags:
                if tag not in chosen:
                    chosen.append(tag)
        return ResolvedConcept(
            tags=tuple(chosen),
            matched=bool(chosen),
            source_phrase=value,
        )

    def resolve_many(self, values: Iterable[str]) -> tuple[ResolvedConcept, ...]:
        return tuple(self.resolve(value) for value in values if value.strip())

    def profile(
        self,
        requested_id: str,
        *,
        model_family: str,
        checkpoint: str | None,
        checkpoint_overrides: Mapping[str, str] | None = None,
    ) -> PromptProfile:
        profile_id = requested_id
        for pattern, override in (checkpoint_overrides or {}).items():
            if checkpoint and fnmatch.fnmatch(checkpoint.casefold(), pattern.casefold()):
                profile_id = override
                break

        requested = self._profiles.get(profile_id)
        if requested is None:
            raise KeyError(f"unknown prompt profile: {profile_id}")
        if model_family not in requested.model_families:
            raise ValueError(
                f"prompt profile {requested.id} does not support {model_family}"
            )
        if requested.checkpoint_globs and checkpoint is not None:
            if not any(
                fnmatch.fnmatch(checkpoint.casefold(), pattern.casefold())
                for pattern in requested.checkpoint_globs
            ):
                raise ValueError(
                    f"prompt profile {requested.id} is incompatible with "
                    f"checkpoint {checkpoint}"
                )
        return requested

    def expand_implications(self, tags: Iterable[str]) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()

        def add(tag: str) -> None:
            if tag in seen:
                return
            seen.add(tag)
            result.append(tag)
            item = self._tags.get(tag)
            if item is None:
                return
            for implied in item.implies:
                add(implied)

        for tag in tags:
            add(tag)
        return tuple(result)

    def resolve_conflicts(self, tags: Iterable[str]) -> tuple[str, ...]:
        result: list[str] = []
        claimed: set[frozenset[str]] = set()
        seen: set[str] = set()
        for tag in tags:
            if tag in seen:
                continue
            group = self._conflict_lookup.get(tag)
            if group is not None:
                if group in claimed:
                    continue
                claimed.add(group)
            seen.add(tag)
            result.append(tag)
        return tuple(result)

    def order(
        self,
        tags: Sequence[str],
        *,
        profile: PromptProfile,
        negative: bool = False,
        category_overrides: Mapping[str, str] | None = None,
    ) -> tuple[str, ...]:
        order = (
            profile.negative_category_order
            if negative
            else profile.positive_category_order
        )
        priority = {category: index for index, category in enumerate(order)}
        indexed = list(enumerate(tags))
        indexed.sort(
            key=lambda item: (
                priority.get(
                    (category_overrides or {}).get(
                        item[1],
                        self.category(item[1]),
                    ),
                    len(priority),
                ),
                item[0],
            )
        )
        return tuple(tag for _, tag in indexed)

    def validate_generated(self, tags: Iterable[str]) -> None:
        invalid = sorted({tag for tag in tags if tag not in self._tags})
        if invalid:
            raise ValueError(
                "generated prompt contains tags outside validated vocabulary: "
                + ", ".join(invalid)
            )

    @staticmethod
    def tokenize(value: str) -> tuple[str, ...]:
        return tuple(match.group(0) for match in _WORDS.finditer(value))
