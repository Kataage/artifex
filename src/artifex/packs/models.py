from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.domain import ContentRating, PackState, PublicationTier, SceneState
from artifex.planner.models import PackFormat


class PackModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SceneRole(StrEnum):
    PREVIEW = "preview"
    FEATURE = "feature"
    CONTINUATION = "continuation"
    OPENING = "opening"
    DEVELOPMENT = "development"
    RESOLUTION = "resolution"
    ALTERNATE = "alternate"
    VARIATION = "variation"
    OUTFIT_REVEAL = "outfit_reveal"
    DETAIL = "detail"
    SEASONAL_HERO = "seasonal_hero"
    TREND_HERO = "trend_hero"
    INTERACTION = "interaction"
    EXPERIMENT = "experiment"


class EditorialArchetype(StrEnum):
    PUBLIC_ONLY = "public_only"
    MEMBER_ONLY = "member_only"
    PUBLIC_PREVIEW_MEMBER_CONTINUATION = "public_preview_member_continuation"
    SFW_COMPLETE_MEMBER_ALTERNATE = "sfw_complete_member_alternate"
    MINI_STORY = "mini_story"
    VARIATION_PACK = "variation_pack"
    SEASONAL_TREND_PACK = "seasonal_trend_pack"


class VisualSpecification(PackModel):
    composition: str = Field(min_length=1, max_length=500)
    camera: str = Field(min_length=1, max_length=300)
    pose: str = Field(min_length=1, max_length=500)
    expression: str = Field(min_length=1, max_length=300)
    clothing: str = Field(min_length=1, max_length=500)
    setting: str = Field(min_length=1, max_length=500)
    lighting: str = Field(min_length=1, max_length=300)
    atmosphere: str = Field(min_length=1, max_length=300)
    positive_constraints: tuple[str, ...] = ()
    negative_constraints: tuple[str, ...] = ()


class ScenePlan(PackModel):
    ordinal: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=500)
    character_ids: tuple[str, ...] = Field(min_length=1)
    role: SceneRole = SceneRole.FEATURE
    continuity_constraints: tuple[str, ...] = ()
    visual: VisualSpecification
    publication_tier: PublicationTier = PublicationTier.PUBLIC
    planned_content_rating: ContentRating = ContentRating.GENERAL
    transition_from_previous: str | None = Field(default=None, max_length=500)
    series_state_updates: dict[str, Any] = Field(default_factory=dict)
    unresolved_hooks_added: tuple[str, ...] = ()
    unresolved_hooks_resolved: tuple[str, ...] = ()


class ContentPackPlan(PackModel):
    format: PackFormat
    editorial_archetype: EditorialArchetype = EditorialArchetype.PUBLIC_ONLY
    character_ids: tuple[str, ...] = Field(min_length=1)
    title: str = Field(min_length=1, max_length=300)
    logline: str = Field(min_length=1, max_length=1000)
    continuity_bible: tuple[str, ...] = ()
    scenes: tuple[ScenePlan, ...] = Field(min_length=1)
    series_id: str | None = None
    episode_number: int | None = Field(default=None, ge=1)
    prior_pack_ids: tuple[str, ...] = ()
    unresolved_hooks_carried: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> ContentPackPlan:
        if len(self.character_ids) != len(set(self.character_ids)):
            raise ValueError("pack character_ids must be unique")

        if self.format is PackFormat.DUO and len(self.character_ids) != 2:
            raise ValueError("duo pack requires exactly two characters")
        if self.format is PackFormat.GROUP and len(self.character_ids) < 3:
            raise ValueError("group pack requires at least three characters")
        if self.format not in {PackFormat.DUO, PackFormat.GROUP} and len(self.character_ids) != 1:
            raise ValueError("non-duo/group pack requires exactly one character")

        ordinals = [scene.ordinal for scene in self.scenes]
        if ordinals != list(range(1, len(self.scenes) + 1)):
            raise ValueError("scene ordinals must be contiguous and start at 1")

        pack_characters = set(self.character_ids)
        seen_characters: set[str] = set()
        for scene in self.scenes:
            scene_characters = set(scene.character_ids)
            if len(scene.character_ids) != len(scene_characters):
                raise ValueError(
                    f"scene {scene.ordinal} contains duplicate character ids"
                )
            if not scene_characters <= pack_characters:
                raise ValueError(
                    f"scene {scene.ordinal} references characters outside the pack"
                )
            seen_characters.update(scene_characters)

        if seen_characters != pack_characters:
            missing = ", ".join(sorted(pack_characters - seen_characters))
            raise ValueError(f"pack characters never appear in scenes: {missing}")

        if self.series_id is None and self.episode_number is not None:
            raise ValueError("episode_number requires series_id")
        if self.series_id is not None and self.episode_number is None:
            raise ValueError("series packs require episode_number")

        self._validate_format_roles()
        self._validate_editorial_archetype()
        return self

    def _validate_format_roles(self) -> None:
        roles = tuple(scene.role for scene in self.scenes)
        first = roles[0]

        if self.format is PackFormat.SINGLE_FEATURE:
            _require_role(first, {SceneRole.FEATURE, SceneRole.PREVIEW}, self.format)
            _require_roles(
                roles[1:],
                {SceneRole.DETAIL, SceneRole.VARIATION, SceneRole.ALTERNATE},
                self.format,
            )
        elif self.format is PackFormat.CONTINUATION:
            _require_role(first, {SceneRole.CONTINUATION, SceneRole.PREVIEW}, self.format)
            _require_roles(
                roles[1:],
                {SceneRole.DEVELOPMENT, SceneRole.RESOLUTION, SceneRole.ALTERNATE},
                self.format,
            )
            if len(roles) > 1 and roles[-1] not in {
                SceneRole.RESOLUTION,
                SceneRole.ALTERNATE,
            }:
                raise ValueError(
                    "continuation pack must end with resolution or alternate"
                )
        elif self.format is PackFormat.MINI_STORY:
            if len(roles) < 3:
                raise ValueError("mini_story pack requires at least three scenes")
            _require_role(first, {SceneRole.OPENING, SceneRole.PREVIEW}, self.format)
            _require_roles(roles[1:-1], {SceneRole.DEVELOPMENT}, self.format)
            _require_role(roles[-1], {SceneRole.RESOLUTION}, self.format)
        elif self.format is PackFormat.VARIATION_PACK:
            if len(roles) < 2:
                raise ValueError("variation_pack requires at least two scenes")
            _require_roles(
                roles,
                {SceneRole.VARIATION, SceneRole.PREVIEW, SceneRole.ALTERNATE},
                self.format,
            )
        elif self.format is PackFormat.OUTFIT_FEATURE:
            if len(roles) < 2:
                raise ValueError("outfit_feature requires at least two scenes")
            _require_role(
                first,
                {SceneRole.OUTFIT_REVEAL, SceneRole.PREVIEW},
                self.format,
            )
            _require_roles(
                roles[1:],
                {SceneRole.DETAIL, SceneRole.VARIATION, SceneRole.ALTERNATE},
                self.format,
            )
        elif self.format is PackFormat.SEASONAL:
            _require_role(
                first,
                {SceneRole.SEASONAL_HERO, SceneRole.PREVIEW},
                self.format,
            )
            _require_roles(
                roles[1:],
                {SceneRole.DETAIL, SceneRole.VARIATION, SceneRole.ALTERNATE},
                self.format,
            )
        elif self.format is PackFormat.TREND:
            _require_role(
                first,
                {SceneRole.TREND_HERO, SceneRole.PREVIEW},
                self.format,
            )
            _require_roles(
                roles[1:],
                {SceneRole.DETAIL, SceneRole.VARIATION, SceneRole.ALTERNATE},
                self.format,
            )
        elif self.format is PackFormat.EVERGREEN:
            _require_role(first, {SceneRole.FEATURE, SceneRole.PREVIEW}, self.format)
            _require_roles(
                roles[1:],
                {SceneRole.DETAIL, SceneRole.VARIATION, SceneRole.ALTERNATE},
                self.format,
            )
        elif self.format is PackFormat.EXPERIMENTAL:
            _require_roles(
                roles,
                {SceneRole.EXPERIMENT, SceneRole.PREVIEW},
                self.format,
            )
        elif self.format in {PackFormat.DUO, PackFormat.GROUP}:
            _require_role(
                first,
                {SceneRole.FEATURE, SceneRole.INTERACTION, SceneRole.PREVIEW},
                self.format,
            )
            _require_roles(
                roles[1:],
                {
                    SceneRole.INTERACTION,
                    SceneRole.DETAIL,
                    SceneRole.VARIATION,
                    SceneRole.ALTERNATE,
                },
                self.format,
            )

    def _validate_editorial_archetype(self) -> None:
        scenes = self.scenes
        tiers = tuple(scene.publication_tier for scene in scenes)
        public = PublicationTier.PUBLIC
        member = PublicationTier.MEMBER

        for scene in scenes:
            if (
                scene.publication_tier is public
                and scene.planned_content_rating
                not in {ContentRating.GENERAL, ContentRating.SUGGESTIVE}
            ):
                raise ValueError(
                    "public scenes cannot intentionally target adult/explicit content"
                )

        if self.editorial_archetype is EditorialArchetype.PUBLIC_ONLY:
            if any(tier is not public for tier in tiers):
                raise ValueError("public_only archetype requires every scene public")
            return

        if self.editorial_archetype is EditorialArchetype.MEMBER_ONLY:
            if any(tier is not member for tier in tiers):
                raise ValueError("member_only archetype requires every scene member")
            return

        if self.editorial_archetype is EditorialArchetype.PUBLIC_PREVIEW_MEMBER_CONTINUATION:
            if len(scenes) < 2:
                raise ValueError(
                    "public_preview_member_continuation requires at least two scenes"
                )
            if scenes[0].publication_tier is not public:
                raise ValueError("preview scene must be public")
            if scenes[0].role is not SceneRole.PREVIEW:
                raise ValueError("preview scene must use preview role")
            if any(scene.publication_tier is not member for scene in scenes[1:]):
                raise ValueError(
                    "all scenes after the public preview must be member-only"
                )
            return

        if self.editorial_archetype is EditorialArchetype.SFW_COMPLETE_MEMBER_ALTERNATE:
            _validate_public_then_member(scenes)
            first_member = next(
                scene for scene in scenes if scene.publication_tier is member
            )
            if first_member.role not in {SceneRole.ALTERNATE, SceneRole.VARIATION}:
                raise ValueError(
                    "member continuation must begin with alternate or variation role"
                )
            return

        if self.editorial_archetype is EditorialArchetype.MINI_STORY:
            if self.format is not PackFormat.MINI_STORY:
                raise ValueError("mini_story archetype requires mini_story format")
            _validate_public_then_member(scenes)
            return

        if self.editorial_archetype is EditorialArchetype.VARIATION_PACK:
            if self.format is not PackFormat.VARIATION_PACK:
                raise ValueError(
                    "variation_pack archetype requires variation_pack format"
                )
            _validate_public_then_member(scenes)
            return

        if self.editorial_archetype is EditorialArchetype.SEASONAL_TREND_PACK:
            if self.format not in {PackFormat.SEASONAL, PackFormat.TREND}:
                raise ValueError(
                    "seasonal_trend_pack archetype requires seasonal or trend format"
                )
            if scenes[0].publication_tier is not public:
                raise ValueError(
                    "seasonal/trend Pack must expose a public first scene"
                )
            _require_public_prefix(tiers)


class PackRecord(PackModel):
    pack_id: str
    concept_id: str | None
    series_id: str | None
    state: PackState
    plan: ContentPackPlan
    scene_states: tuple[SceneState, ...]
    created_at: datetime
    updated_at: datetime


def _require_role(
    role: SceneRole,
    allowed: set[SceneRole],
    pack_format: PackFormat,
) -> None:
    if role not in allowed:
        allowed_text = ", ".join(sorted(item.value for item in allowed))
        raise ValueError(
            f"{pack_format.value} scene role {role.value} is invalid; "
            f"expected one of: {allowed_text}"
        )


def _require_roles(
    roles: tuple[SceneRole, ...],
    allowed: set[SceneRole],
    pack_format: PackFormat,
) -> None:
    for role in roles:
        _require_role(role, allowed, pack_format)


def _validate_public_then_member(scenes: tuple[ScenePlan, ...]) -> None:
    tiers = tuple(scene.publication_tier for scene in scenes)
    if PublicationTier.PUBLIC not in tiers or PublicationTier.MEMBER not in tiers:
        raise ValueError(
            "archetype requires at least one public and one member scene"
        )
    _require_public_prefix(tiers)


def _require_public_prefix(tiers: tuple[PublicationTier, ...]) -> None:
    member_seen = False
    for tier in tiers:
        if tier is PublicationTier.MEMBER:
            member_seen = True
            continue
        if tier is PublicationTier.PUBLIC and member_seen:
            raise ValueError("public scenes cannot appear after member-only scenes")
        if tier not in {PublicationTier.PUBLIC, PublicationTier.MEMBER}:
            raise ValueError(
                "editorial archetypes may plan only public/member tiers"
            )
