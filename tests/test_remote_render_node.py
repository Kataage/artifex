from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from artifex.characters import CharacterRegistry
from artifex.db import Database
from artifex.domain import CharacterProfile, LoRAState
from artifex.loras import LoRARegistry, RemoteLoRADiscovery
from artifex.render_node import RenderLoRAInventoryItem, RenderNodeAttestation


def _attestation(
    *,
    loras: tuple[RenderLoRAInventoryItem, ...],
) -> RenderNodeAttestation:
    return RenderNodeAttestation(
        node_id="gpu-box",
        created_at=datetime.now(UTC),
        hostname="render-pc",
        os={
            "system": "Windows",
            "release": "11",
            "version": "test",
            "machine": "AMD64",
        },
        nvidia_gpus=(
            {
                "name": "RTX 3060",
                "uuid": "GPU-test",
                "driver_version": "999",
                "memory_total_mib": 12288,
            },
        ),
        comfyui_base_url="http://127.0.0.1:8188",
        loras=loras,
    )


def test_remote_lora_inventory_is_synced_without_copying_model_file(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'remote.sqlite3').as_posix()}")
    database.migrate()
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-1",
            display_name="Character 1",
            namespace="test",
            canonical_tags=("char_1",),
            readiness=1.0,
        )
    )
    registry = LoRARegistry(database)
    discovery = RemoteLoRADiscovery(registry, characters)
    item = RenderLoRAInventoryItem(
        name="char_1.safetensors",
        path=r"D:\ComfyUI\models\loras\characters\char_1.safetensors",
        relative_path="characters/char_1.safetensors",
        sha256="a" * 64,
        bytes=1234,
        metadata={
            "modelspec.trigger_phrase": "char_1",
            "modelspec.architecture": "illustrious",
        },
    )

    first = discovery.sync(_attestation(loras=(item,)))

    assert len(first.discovered) == 1
    profile = first.discovered[0]
    assert profile.state is LoRAState.DISCOVERED
    assert profile.checksum == "a" * 64
    assert profile.target_character_ids == ("char-1",)
    assert profile.metadata["asset_name"] == "characters/char_1.safetensors"
    assert profile.metadata["remote_node_id"] == "gpu-box"
    assert profile.source == "render-node:gpu-box"
    assert profile.path.is_file() is False

    second = discovery.sync(_attestation(loras=(item,)))
    assert [value.id for value in second.unchanged] == [profile.id]

    removed = discovery.sync(_attestation(loras=()))
    assert [value.id for value in removed.removed] == [profile.id]
    assert registry.require(profile.id).state is LoRAState.DISABLED
    database.dispose()


def test_partial_remote_inventory_failure_never_disables_missing_loras(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'partial.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    discovery = RemoteLoRADiscovery(registry, CharacterRegistry(database))
    first = RenderLoRAInventoryItem(
        name="first.safetensors",
        path="D:/loras/first.safetensors",
        relative_path="first.safetensors",
        sha256="a" * 64,
        bytes=1234,
    )
    second = RenderLoRAInventoryItem(
        name="second.safetensors",
        path="D:/loras/second.safetensors",
        relative_path="second.safetensors",
        sha256="b" * 64,
        bytes=1234,
    )
    registered = discovery.sync(_attestation(loras=(first, second)))
    assert len(registered.discovered) == 2

    partial = _attestation(loras=(first,)).model_copy(
        update={
            "inventory_errors": (
                {"path": "D:/loras/second.safetensors", "error": "access denied"},
            )
        }
    )
    result = discovery.sync(partial)
    assert result.failed
    assert not result.removed
    assert all(
        registry.require(profile.id).state is LoRAState.DISCOVERED
        for profile in registered.discovered
    )

    complete = discovery.sync(_attestation(loras=(first,)))
    assert len(complete.removed) == 1
    assert registry.require(complete.removed[0].id).state is LoRAState.DISABLED
    database.dispose()
