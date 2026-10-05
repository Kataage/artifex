from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.domain import PackState, PublicationTier, SceneState
from artifex.planner.models import PackFormat


class PackModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    continuity_constraints: tuple[str, ...] = ()
    visual: VisualSpecification
    publication_tier: PublicationTier = PublicationTier.PUBLIC
    transition_from_previous: str | None = Field(default=None, max_length=500)
    series_state_updates: dict[str, Any] = Field(default_factory=dict)
    unresolved_hooks_added: tuple[str, ...] = ()
    unresolved_hooks_resolved: tuple[str, ...] = ()


class ContentPackPlan(PackModel):
    format: PackFormat
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
                raise ValueError(f"scene {scene.ordinal} contains duplicate character ids")
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
        return self


class PackRecord(PackModel):
    pack_id: str
    concept_id: str | None
    series_id: str | None
    state: PackState
    plan: ContentPackPlan
    scene_states: tuple[SceneState, ...]
    created_at: datetime
    updated_at: datetime
