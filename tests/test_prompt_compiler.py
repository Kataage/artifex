from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifex.characters import CharacterRegistry
from artifex.db import Database
from artifex.domain import CharacterProfile, LoRAPolicy
from artifex.loras import LoRAPlan, LoRAPlanEntry
from artifex.packs import ScenePlan, VisualSpecification
from artifex.prompts import ILXLDanbooruAdapter, PromptCompiler, PromptCompilerError


def _scene() -> ScenePlan:
    return ScenePlan(
        ordinal=1,
        title="Window angel",
        purpose="hero image",
        character_ids=("amane_kanata",),
        visual=VisualSpecification(
            composition="full body, upper body",
            camera="low angle",
            pose="sitting on window frame",
            expression="soft smile",
            clothing="white dress",
            setting="open window",
            lighting="sunset rim light",
            atmosphere="airy",
            positive_constraints=(
                "looking at viewer",
                "amane kanata",
                "from above",
            ),
            negative_constraints=(
                "no extra fingers",
                "bad hands",
                "full body",
                "extra fingers",
            ),
        ),
    )


def _lora_plan() -> LoRAPlan:
    return LoRAPlan(
        model_family="ilxl",
        character_ids=("amane_kanata",),
        entries=(
            LoRAPlanEntry(
                lora_id="kanata-lora",
                path="kanata.safetensors",
                weight=0.8,
                character_ids=("amane_kanata",),
                trigger_tags=("kanata", "angel", "amane kanata"),
            ),
        ),
    )


def _registry(tmp_path: Path) -> tuple[Database, CharacterRegistry]:
    database = Database(f"sqlite:///{(tmp_path / 'prompts.sqlite3').as_posix()}")
    database.migrate()
    registry = CharacterRegistry(database)
    registry.upsert(
        CharacterProfile(
            id="amane_kanata",
            display_name="天音かなた",
            namespace="hololive",
            canonical_tags=("amane_kanata", "hololive"),
            model_families=("ilxl",),
            lora_policy=LoRAPolicy.REQUIRED,
            readiness=1.0,
        )
    )
    return database, registry


def test_ilxl_prompt_regression_fixture(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)
    compiled = PromptCompiler(
        registry,
        (ILXLDanbooruAdapter(),),
    ).compile(_scene(), _lora_plan())

    fixture_path = (
        Path(__file__).parent / "fixtures" / "prompt_ilxl_window_angel.json"
    )
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert compiled.positive_prompt == expected["positive_prompt"]
    assert compiled.negative_prompt == expected["negative_prompt"]
    assert list(compiled.positive_tags) == expected["positive_tags"]
    assert list(compiled.negative_tags) == expected["negative_tags"]
    assert compiled.provenance.adapter_id == "ilxl_danbooru"
    assert compiled.provenance.adapter_version == "1"
    assert compiled.provenance.lora_ids == ("kanata-lora",)
    assert compiled.provenance.lora_weights == (0.8,)
    database.dispose()


def test_conflicts_are_first_wins_by_compiler_order(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)
    compiled = PromptCompiler(
        registry,
        (ILXLDanbooruAdapter(),),
    ).compile(_scene(), _lora_plan())

    assert "full_body" in compiled.positive_tags
    assert "upper_body" not in compiled.positive_tags
    assert "from_below" in compiled.positive_tags
    assert "from_above" not in compiled.positive_tags
    assert "full_body" not in compiled.negative_tags
    assert compiled.negative_tags.count("extra_fingers") == 1
    database.dispose()


def test_scene_and_lora_plan_character_sets_must_match(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)
    bad_plan = _lora_plan().model_copy(update={"character_ids": ("other",)})
    compiler = PromptCompiler(registry, (ILXLDanbooruAdapter(),))

    with pytest.raises(PromptCompilerError, match="exactly match"):
        compiler.compile(_scene(), bad_plan)
    database.dispose()


def test_unknown_model_family_requires_adapter(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)
    plan = _lora_plan().model_copy(update={"model_family": "future-family"})
    compiler = PromptCompiler(registry, (ILXLDanbooruAdapter(),))

    with pytest.raises(PromptCompilerError, match="no prompt adapter"):
        compiler.compile(_scene(), plan)
    database.dispose()


def test_duplicate_adapter_registration_is_rejected(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)

    with pytest.raises(ValueError, match="duplicate prompt adapter"):
        PromptCompiler(
            registry,
            (ILXLDanbooruAdapter(), ILXLDanbooruAdapter()),
        )
    database.dispose()
