from __future__ import annotations

from pathlib import Path

import pytest

from artifex.characters import CharacterRegistry
from artifex.config.models import CharacterRegistryConfig, LoRARegistryConfig
from artifex.db import Database
from artifex.domain import CharacterProfile, LoRAPolicy, LoRAProfile, LoRAState
from artifex.loras import LoRARegistry, LoRAResolutionError, LoRAResolver


def _setup(tmp_path: Path) -> tuple[Database, CharacterRegistry, LoRARegistry]:
    database = Database(f"sqlite:///{(tmp_path / 'resolver.sqlite3').as_posix()}")
    database.migrate()
    return database, CharacterRegistry(database), LoRARegistry(database)


def _character(
    character_id: str,
    *,
    policy: LoRAPolicy = LoRAPolicy.REQUIRED,
    preferred: tuple[str, ...] = (),
) -> CharacterProfile:
    return CharacterProfile(
        id=character_id,
        display_name=character_id,
        namespace="test",
        model_families=("ilxl",),
        lora_policy=policy,
        preferred_lora_ids=preferred,
        readiness=0.9,
    )


def _production_lora(
    tmp_path: Path,
    lora_id: str,
    character_ids: tuple[str, ...],
    *,
    readiness: float,
    conflicts: tuple[str, ...] = (),
    weight: float = 0.8,
) -> LoRAProfile:
    return LoRAProfile(
        id=lora_id,
        path=tmp_path / f"{lora_id}.safetensors",
        state=LoRAState.PRODUCTION,
        target_character_ids=character_ids,
        model_families=("ilxl",),
        recommended_weight=weight,
        identity_score=readiness,
        quality_score=readiness,
        flexibility_score=readiness,
        readiness=readiness,
        incompatible_lora_ids=conflicts,
    )


def test_required_policy_selects_preferred_production_lora(tmp_path: Path) -> None:
    database, characters, loras = _setup(tmp_path)
    characters.upsert(_character("char-a", preferred=("preferred",)))
    loras.upsert(_production_lora(tmp_path, "other", ("char-a",), readiness=0.99))
    loras.upsert(_production_lora(tmp_path, "preferred", ("char-a",), readiness=0.8))

    plan = LoRAResolver(
        characters,
        loras,
        CharacterRegistryConfig(),
        LoRARegistryConfig(),
    ).resolve(("char-a",), model_family="ilxl")

    assert len(plan.entries) == 1
    assert plan.entries[0].lora_id == "preferred"
    database.dispose()


def test_missing_required_lora_blocks_scene(tmp_path: Path) -> None:
    database, characters, loras = _setup(tmp_path)
    characters.upsert(_character("char-a"))

    resolver = LoRAResolver(
        characters,
        loras,
        CharacterRegistryConfig(),
        LoRARegistryConfig(),
    )
    with pytest.raises(LoRAResolutionError, match="required production LoRA"):
        resolver.resolve(("char-a",), model_family="ilxl")
    database.dispose()


def test_duo_conflicting_loras_are_rejected(tmp_path: Path) -> None:
    database, characters, loras = _setup(tmp_path)
    characters.upsert(_character("char-a"))
    characters.upsert(_character("char-b"))
    loras.upsert(
        _production_lora(
            tmp_path,
            "lora-a",
            ("char-a",),
            readiness=0.9,
            conflicts=("lora-b",),
        )
    )
    loras.upsert(_production_lora(tmp_path, "lora-b", ("char-b",), readiness=0.9))

    resolver = LoRAResolver(
        characters,
        loras,
        CharacterRegistryConfig(),
        LoRARegistryConfig(),
    )
    with pytest.raises(LoRAResolutionError, match="conflicts"):
        resolver.resolve(("char-a", "char-b"), model_family="ilxl")
    database.dispose()


def test_group_respects_maximum_character_lora_stack(tmp_path: Path) -> None:
    database, characters, loras = _setup(tmp_path)
    for index in range(3):
        character_id = f"char-{index}"
        lora_id = f"lora-{index}"
        characters.upsert(_character(character_id))
        loras.upsert(
            _production_lora(
                tmp_path,
                lora_id,
                (character_id,),
                readiness=0.9,
            )
        )

    resolver = LoRAResolver(
        characters,
        loras,
        CharacterRegistryConfig(),
        LoRARegistryConfig(maximum_character_loras_per_scene=2),
    )
    with pytest.raises(LoRAResolutionError, match="exceeds configured"):
        resolver.resolve(("char-0", "char-1", "char-2"), model_family="ilxl")
    database.dispose()


def test_preferred_policy_can_fall_back_to_base_model(tmp_path: Path) -> None:
    database, characters, loras = _setup(tmp_path)
    characters.upsert(_character("char-a", policy=LoRAPolicy.PREFERRED))

    plan = LoRAResolver(
        characters,
        loras,
        CharacterRegistryConfig(),
        LoRARegistryConfig(),
    ).resolve(("char-a",), model_family="ilxl")

    assert plan.entries == ()
    assert "preferred LoRA unavailable" in plan.warnings[0]
    database.dispose()
