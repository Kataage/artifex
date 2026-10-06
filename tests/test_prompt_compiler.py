from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifex.characters import CharacterRegistry
from artifex.db import Database
from artifex.domain import CharacterOutfit, CharacterProfile, LoRAPolicy
from artifex.loras import LoRAPlan, LoRAPlanEntry
from artifex.packs import ScenePlan, VisualSpecification
from artifex.prompts import (
    ILXLDanbooruAdapter,
    PromptCompiler,
    PromptCompilerError,
    ValidatedTagLexicon,
)


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
            aliases=("Amane Kanata",),
            model_families=("ilxl",),
            lora_policy=LoRAPolicy.REQUIRED,
            readiness=1.0,
        )
    )
    return database, registry


def _multi_registry(
    tmp_path: Path,
) -> tuple[Database, CharacterRegistry]:
    database = Database(f"sqlite:///{(tmp_path / 'corpus.sqlite3').as_posix()}")
    database.migrate()
    registry = CharacterRegistry(database)
    for index in range(1, 4):
        registry.upsert(
            CharacterProfile(
                id=f"char_{index}",
                display_name=f"Character {index}",
                namespace="hololive",
                canonical_tags=(f"char_{index}",),
                model_families=("ilxl",),
                lora_policy=LoRAPolicy.NONE,
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
    assert list(compiled.unresolved_concepts) == expected["unresolved_concepts"]
    assert compiled.provenance.adapter_id == "ilxl_danbooru"
    assert compiled.provenance.adapter_version == "2"
    assert compiled.provenance.lexicon_id == "artifex-ilxl-danbooru-core"
    assert compiled.provenance.lexicon_version == "2026-10-06.2"
    assert compiled.provenance.prompt_profile_id == "ilxl-danbooru-v2"
    assert compiled.provenance.lora_ids == ("kanata-lora",)
    assert compiled.provenance.lora_weights == (0.8,)
    database.dispose()


def test_conflicts_implications_and_category_order_are_deterministic(
    tmp_path: Path,
) -> None:
    database, registry = _registry(tmp_path)
    compiler = PromptCompiler(registry, (ILXLDanbooruAdapter(),))
    first = compiler.compile(_scene(), _lora_plan())
    second = compiler.compile(_scene(), _lora_plan())

    assert first == second
    assert "full_body" in first.positive_tags
    assert "upper_body" not in first.positive_tags
    assert "from_below" in first.positive_tags
    assert "from_above" not in first.positive_tags
    assert "white_dress" in first.positive_tags
    assert "dress" in first.positive_tags
    assert "full_body" not in first.negative_tags
    assert first.negative_tags.count("extra_digits") == 1
    assert first.positive_tags.index("full_body") < first.positive_tags.index("sitting")
    assert first.positive_tags.index("sitting") < first.positive_tags.index("smile")
    assert first.positive_tags.index("smile") < first.positive_tags.index("white_dress")
    database.dispose()


def test_unknown_prose_is_reported_and_never_invented_as_tag(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)
    scene = _scene().model_copy(
        update={
            "visual": _scene().visual.model_copy(
                update={"pose": "floating upside down on impossible moonbeam"}
            )
        }
    )
    compiled = PromptCompiler(
        registry,
        (ILXLDanbooruAdapter(unresolved_policy="report"),),
    ).compile(scene, _lora_plan())

    assert "floating_upside_down_on_impossible_moonbeam" not in compiled.positive_tags
    assert any(
        item.startswith("pose:floating upside down")
        for item in compiled.unresolved_concepts
    )

    strict = PromptCompiler(
        registry,
        (ILXLDanbooruAdapter(unresolved_policy="error"),),
    )
    with pytest.raises(ValueError, match="unresolved prompt concepts"):
        strict.compile(scene, _lora_plan())
    database.dispose()


def test_character_and_outfit_required_forbidden_tags_are_enforced(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'outfit.sqlite3').as_posix()}")
    database.migrate()
    registry = CharacterRegistry(database)
    registry.upsert(
        CharacterProfile(
            id="char_outfit",
            display_name="Outfit Character",
            namespace="hololive",
            canonical_tags=("char_outfit",),
            required_tags=("ribbon",),
            forbidden_tags=("watermark",),
            outfits=(
                CharacterOutfit(
                    id="summer",
                    display_name="Summer Outfit",
                    aliases=("summer outfit",),
                    canonical_tags=("white_dress",),
                    required_tags=("thighhighs",),
                    forbidden_tags=("jacket",),
                ),
            ),
            lora_policy=LoRAPolicy.NONE,
            readiness=1.0,
        )
    )
    scene = ScenePlan(
        ordinal=1,
        title="Outfit rules",
        purpose="regression",
        character_ids=("char_outfit",),
        visual=VisualSpecification(
            composition="full body",
            camera="eye level",
            pose="standing",
            expression="smile",
            clothing="summer outfit, jacket",
            setting="garden",
            lighting="sunlight",
            atmosphere="wind",
        ),
    )
    plan = LoRAPlan(
        model_family="ilxl",
        character_ids=("char_outfit",),
        entries=(),
    )
    compiled = PromptCompiler(
        registry,
        (ILXLDanbooruAdapter(),),
    ).compile(scene, plan)

    assert "char_outfit" in compiled.positive_tags
    assert "ribbon" in compiled.positive_tags
    assert "white_dress" in compiled.positive_tags
    assert "thighhighs" in compiled.positive_tags
    assert "jacket" not in compiled.positive_tags
    assert "jacket" in compiled.negative_tags
    assert "watermark" in compiled.negative_tags
    database.dispose()


def test_checkpoint_override_selects_versioned_prompt_profile(tmp_path: Path) -> None:
    database, registry = _registry(tmp_path)
    adapter = ILXLDanbooruAdapter(
        checkpoint="my-ilxl-checkpoint.safetensors",
        checkpoint_profile_overrides={
            "*my-ilxl-checkpoint*": "ilxl-danbooru-character-first-v2"
        },
    )
    compiled = PromptCompiler(registry, (adapter,)).compile(_scene(), _lora_plan())

    assert (
        compiled.provenance.prompt_profile_id
        == "ilxl-danbooru-character-first-v2"
    )
    assert compiled.provenance.checkpoint == "my-ilxl-checkpoint.safetensors"
    database.dispose()


def test_representative_ilxl_regression_corpus_is_vocabulary_valid(
    tmp_path: Path,
) -> None:
    database, registry = _multi_registry(tmp_path)
    lexicon = ValidatedTagLexicon.packaged()
    compiler = PromptCompiler(registry, (ILXLDanbooruAdapter(lexicon=lexicon),))
    fixture = (
        Path(__file__).parent / "fixtures" / "ilxl_prompt_regression_corpus.json"
    )
    cases = json.loads(fixture.read_text(encoding="utf-8"))

    for case in cases:
        character_ids = tuple(
            f"char_{index}"
            for index in range(1, int(case["characters"]) + 1)
        )
        scene = ScenePlan(
            ordinal=1,
            title=str(case["name"]),
            purpose="prompt regression corpus",
            character_ids=character_ids,
            visual=VisualSpecification(
                composition=str(case["composition"]),
                camera=str(case["camera"]),
                pose=str(case["pose"]),
                expression=str(case["expression"]),
                clothing=str(case["clothing"]),
                setting=str(case["setting"]),
                lighting=str(case["lighting"]),
                atmosphere=str(case["atmosphere"]),
                positive_constraints=tuple(case.get("positive_constraints", ())),
                negative_constraints=tuple(case.get("negative_constraints", ())),
            ),
        )
        plan = LoRAPlan(
            model_family="ilxl",
            character_ids=character_ids,
            entries=(),
        )

        compiled = compiler.compile(scene, plan)
        repeated = compiler.compile(scene, plan)

        assert compiled == repeated, case["name"]
        assert compiled.unresolved_concepts == (), (
            case["name"],
            compiled.unresolved_concepts,
        )
        for expected in case["contains"]:
            assert expected in compiled.positive_tags, (case["name"], expected)
        trusted = set(character_ids)
        for tag in compiled.positive_tags:
            assert tag in trusted or lexicon.is_valid(tag), (case["name"], tag)
        for tag in compiled.negative_tags:
            assert lexicon.is_valid(tag), (case["name"], tag)

    assert any(int(case["characters"]) == 2 for case in cases)
    assert any(int(case["characters"]) >= 3 for case in cases)
    assert len(cases) >= 30
    database.dispose()


def test_generated_vocabulary_validation_rejects_unknown_tag() -> None:
    lexicon = ValidatedTagLexicon.packaged()
    with pytest.raises(ValueError, match="outside validated vocabulary"):
        lexicon.validate_generated(("definitely_not_a_real_validated_tag",))


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
