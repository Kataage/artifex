from __future__ import annotations

import pytest

from artifex.packs import ContentPackPlan, ScenePlan, VisualSpecification


def _visual() -> VisualSpecification:
    return VisualSpecification(
        composition="full body",
        camera="eye level",
        pose="standing",
        expression="smile",
        clothing="casual outfit",
        setting="city street",
        lighting="soft evening light",
        atmosphere="warm",
    )


def _scene(ordinal: int, characters: tuple[str, ...]) -> ScenePlan:
    return ScenePlan(
        ordinal=ordinal,
        title=f"scene {ordinal}",
        purpose="advance the visual sequence",
        character_ids=characters,
        visual=_visual(),
    )


def test_duo_pack_requires_two_characters() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        ContentPackPlan(
            format="duo",
            character_ids=("a",),
            title="duo",
            logline="duo feature",
            scenes=(_scene(1, ("a",)),),
        )


def test_group_pack_requires_all_pack_characters_to_appear() -> None:
    with pytest.raises(ValueError, match="never appear"):
        ContentPackPlan(
            format="group",
            character_ids=("a", "b", "c"),
            title="group",
            logline="group feature",
            scenes=(
                _scene(1, ("a", "b")),
                _scene(2, ("a", "b")),
            ),
        )


def test_scene_ordinals_must_be_contiguous() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        ContentPackPlan(
            format="single_feature",
            character_ids=("a",),
            title="single",
            logline="single feature",
            scenes=(_scene(2, ("a",)),),
        )
