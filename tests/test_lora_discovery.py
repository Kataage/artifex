from __future__ import annotations

import json
from pathlib import Path

from artifex.characters import CharacterRegistry
from artifex.db import Database
from artifex.domain import CharacterProfile, LoRAPolicy, LoRAState
from artifex.loras import LoRADiscovery, LoRARegistry


def _write_fake_safetensors(
    path: Path,
    metadata: dict[str, str],
    *,
    payload: bytes = b"",
) -> None:
    header = json.dumps(
        {"__metadata__": metadata},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)


def _registries(tmp_path: Path) -> tuple[Database, CharacterRegistry, LoRARegistry]:
    database = Database(f"sqlite:///{(tmp_path / 'lora.sqlite3').as_posix()}")
    database.migrate()
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="amane_kanata",
            display_name="天音かなた",
            namespace="hololive",
            aliases=("amane kanata", "天音かなた"),
            canonical_tags=("amane_kanata",),
            lora_policy=LoRAPolicy.REQUIRED,
            readiness=0.9,
        )
    )
    return database, characters, LoRARegistry(database)


def test_discovery_reads_metadata_checksum_and_character_match(tmp_path: Path) -> None:
    database, characters, loras = _registries(tmp_path)
    root = tmp_path / "loras"
    root.mkdir()
    path = root / "amane_kanata_ILXL.safetensors"
    _write_fake_safetensors(
        path,
        {
            "ss_sd_model_name": "Illustrious-XL-v2",
            "modelspec.trigger_phrase": "kanata, angel",
        },
    )

    discovery = LoRADiscovery(loras, characters)
    result = discovery.scan((root,))

    assert not result.failed
    assert len(result.discovered) == 1
    profile = result.discovered[0]
    assert profile.state is LoRAState.DISCOVERED
    assert profile.target_character_ids == ("amane_kanata",)
    assert profile.model_families == ("ilxl",)
    assert profile.trigger_tags == ("kanata", "angel")
    assert profile.checksum is not None

    second = discovery.scan((root,))
    assert second.discovered == ()
    assert len(second.unchanged) == 1
    database.dispose()


def test_changed_file_resets_previous_production_validation(tmp_path: Path) -> None:
    database, characters, loras = _registries(tmp_path)
    root = tmp_path / "loras"
    root.mkdir()
    path = root / "amane_kanata.safetensors"
    _write_fake_safetensors(path, {"ss_sd_model_name": "illustrious"})

    discovery = LoRADiscovery(loras, characters)
    first = discovery.scan((root,)).discovered[0]
    loras.upsert(
        first.model_copy(
            update={
                "state": LoRAState.PRODUCTION,
                "readiness": 0.95,
                "identity_score": 0.95,
                "quality_score": 0.9,
                "flexibility_score": 0.8,
            }
        )
    )

    _write_fake_safetensors(
        path,
        {"ss_sd_model_name": "illustrious", "version": "replacement"},
        payload=b"changed",
    )
    changed = discovery.scan((root,)).discovered[0]

    assert changed.id == first.id
    assert changed.state is LoRAState.DISCOVERED
    assert changed.readiness == 0.0
    assert changed.identity_score is None
    database.dispose()



def test_removed_production_asset_is_disabled_until_revalidated(tmp_path: Path) -> None:
    database, characters, loras = _registries(tmp_path)
    root = tmp_path / "loras"
    root.mkdir()
    path = root / "amane_kanata.safetensors"
    _write_fake_safetensors(path, {"ss_sd_model_name": "illustrious"})

    discovery = LoRADiscovery(loras, characters)
    first = discovery.scan((root,)).discovered[0]
    loras.upsert(
        first.model_copy(
            update={
                "state": LoRAState.PRODUCTION,
                "readiness": 0.95,
                "identity_score": 0.95,
                "quality_score": 0.90,
                "flexibility_score": 0.80,
            }
        )
    )

    original_bytes = path.read_bytes()
    path.unlink()
    removed = discovery.scan((root,))

    assert len(removed.removed) == 1
    disabled = loras.require(first.id)
    assert disabled.state is LoRAState.DISABLED
    assert disabled.readiness == 0.0
    assert disabled.missing_since is not None
    assert disabled.identity_score is None

    path.write_bytes(original_bytes)
    restored = discovery.scan((root,))

    assert len(restored.discovered) == 1
    profile = loras.require(first.id)
    assert profile.state is LoRAState.DISCOVERED
    assert profile.missing_since is None
    assert profile.readiness == 0.0
    database.dispose()


def test_corrupt_replacement_invalidates_previous_production_asset(tmp_path: Path) -> None:
    database, characters, loras = _registries(tmp_path)
    root = tmp_path / "loras"
    root.mkdir()
    path = root / "amane_kanata.safetensors"
    _write_fake_safetensors(path, {"ss_sd_model_name": "illustrious"})

    discovery = LoRADiscovery(loras, characters)
    first = discovery.scan((root,)).discovered[0]
    loras.upsert(
        first.model_copy(
            update={
                "state": LoRAState.PRODUCTION,
                "readiness": 0.95,
                "identity_score": 0.95,
                "quality_score": 0.90,
                "flexibility_score": 0.80,
            }
        )
    )

    path.write_bytes(b"not-a-safetensors-file")
    result = discovery.scan((root,))

    assert result.failed
    assert len(result.invalidated) == 1
    failed = loras.require(first.id)
    assert failed.state is LoRAState.FAILED
    assert failed.readiness == 0.0
    assert failed.identity_score is None
    database.dispose()
