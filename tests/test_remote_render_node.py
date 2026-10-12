from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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


def test_resolver_never_dispatches_secondary_or_controller_lora_to_primary(
    tmp_path: Path,
) -> None:
    from artifex.config.models import (
        CharacterRegistryConfig,
        LoRARegistryConfig,
        RenderNodeConfig,
        RenderNodesConfig,
    )
    from artifex.domain import LoRAPolicy, LoRAProfile
    from artifex.loras import LoRAResolver

    database = Database(f"sqlite:///{(tmp_path / 'routing.sqlite3').as_posix()}")
    database.migrate()
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-1",
            display_name="Character One",
            namespace="test",
            canonical_tags=("char_1",),
            lora_policy=LoRAPolicy.REQUIRED,
            readiness=1.0,
        )
    )
    registry = LoRARegistry(database)
    for node in ("secondary", "primary"):
        registry.upsert(
            LoRAProfile(
                id=f"{node}-char",
                path=tmp_path / f"{node}.safetensors",
                state=LoRAState.PRODUCTION,
                lora_type="character",
                model_families=("ilxl",),
                target_character_ids=("char-1",),
                readiness=1.0,
                source=f"render-node:{node}",
                metadata={"asset_name": "char.safetensors"},
            )
        )
    renderer = RenderNodesConfig(
        primary="primary",
        nodes={
            "primary": RenderNodeConfig(base_url="http://primary.test:8188"),
            "secondary": RenderNodeConfig(base_url="http://secondary.test:8188"),
        },
    )
    resolver = LoRAResolver(
        characters, registry, CharacterRegistryConfig(), LoRARegistryConfig(),
        renderer,
    )
    plan = resolver.resolve(("char-1",), model_family="ilxl")
    assert [entry.lora_id for entry in plan.entries] == ["primary-char"]
    database.dispose()


def test_renderer_attestation_reports_duplicate_relative_lora_names(
    tmp_path: Path,
) -> None:
    import json

    from artifex.config.models import ArtifexSettings, RenderAgentConfig
    from artifex.render_node import build_attestation

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    header = json.dumps({"__metadata__": {"key": "value"}}).encode()
    data = len(header).to_bytes(8, "little") + header
    (first_root / "same.safetensors").write_bytes(data)
    (second_root / "same.safetensors").write_bytes(data)

    attestation = build_attestation(
        ArtifexSettings(
            render_agent=RenderAgentConfig(
                node_id="primary",
                lora_roots=(first_root, second_root),
            )
        )
    )
    assert len(attestation.loras) == 1
    assert any(
        error["error"] == "duplicate_relative_lora_asset"
        for error in attestation.inventory_errors
    )


def test_inventory_errors_do_not_expose_pc_b_private_paths_or_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful attestation can still contain partial scan failures."""
    from artifex.config.models import ArtifexSettings
    from artifex.render_node import attestation as module

    monkeypatch.setattr(module, "_nvidia_gpus", lambda: ())
    private = "PRIVATE_HOME_AUTH_TOKEN_DO_NOT_EXPOSE"
    missing_asset = tmp_path / private / "checkpoint.safetensors"
    missing_lora_root = tmp_path / private / "absent-loras"
    bad_root = tmp_path / "damaged-loras"
    bad_root.mkdir()
    (bad_root / f"{private}.safetensors").write_bytes(b"broken safetensors")

    settings = ArtifexSettings()
    settings.render_agent.asset_paths = {"checkpoint": missing_asset}
    settings.render_agent.lora_roots = (missing_lora_root, bad_root)
    attestation = module.build_attestation(settings)
    errors = {(e["path"], e["error"]) for e in attestation.inventory_errors}
    assert errors == {
        ("asset", "asset_unavailable"),
        ("lora_root", "lora_root_missing"),
        ("lora_asset", "lora_asset_unreadable"),
    }
    wire = attestation.model_dump_json()
    assert private not in wire
    assert str(tmp_path) not in wire
    # Missing inventory items must never silently disable previously known LoRAs.
    assert len(attestation.inventory_errors) == 3



@pytest.mark.parametrize("bad_clock", ["stale", "future", "naive"])
def test_untrusted_remote_inventory_cannot_disable_existing_loras(
    tmp_path: Path, bad_clock: str,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'stale-remote.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    remote = RemoteLoRADiscovery(registry, CharacterRegistry(database))
    item = RenderLoRAInventoryItem(
        name="working.safetensors",
        path=r"D:\ComfyUI\loras\working.safetensors",
        relative_path="working.safetensors",
        sha256="a" * 64,
        bytes=1,
    )
    original = remote.sync(_attestation(loras=(item,)))
    assert len(original.discovered) == 1
    profile = original.discovered[0]
    clock = datetime.now(UTC)
    if bad_clock == "stale":
        clock -= timedelta(minutes=4)
    elif bad_clock == "future":
        clock += timedelta(seconds=40)
    else:
        clock = clock.replace(tzinfo=None)
    evidence = _attestation(loras=()).model_copy(update={"created_at": clock})
    blocked = remote.sync(evidence)
    assert not blocked.discovered and not blocked.removed
    assert blocked.failed[0][1] == "stale_or_untrusted_inventory"
    assert registry.require(profile.id).state is LoRAState.DISCOVERED
    assert registry.require(profile.id).checksum == item.sha256
    database.dispose()


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.safetensors",
        "/absolute.safetensors",
        r"parent\windows.safetensors",
        "C:/windows.safetensors",
        "nested//empty.safetensors",
        "nested/./dot.safetensors",
        "nested/../escape.safetensors",
    ],
)
def test_untrusted_remote_paths_never_mutate_registry(
    tmp_path: Path, relative_path: str,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'unsafe-path.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    remote = RemoteLoRADiscovery(registry, CharacterRegistry(database))
    original = RenderLoRAInventoryItem(
        name="safe.safetensors", path="D:/loras/safe.safetensors",
        relative_path="safe.safetensors", sha256="a" * 64, bytes=1,
    )
    initial = remote.sync(_attestation(loras=(original,)))
    profile = initial.discovered[0]
    candidate = original.model_copy(update={
        "relative_path": relative_path,
        "name": relative_path.split("/")[-1],
        "sha256": "b" * 64,
    })
    blocked = remote.sync(_attestation(loras=(candidate,)))
    assert not blocked.discovered and not blocked.removed
    assert blocked.failed[0][1] == "stale_or_untrusted_inventory"
    assert registry.require(profile.id).checksum == original.sha256
    assert len(registry.list()) == 1
    database.dispose()


@pytest.mark.parametrize("node_id", ["../foreign", "/root", "nested/node", "C:foreign"])
def test_unsafe_remote_node_id_is_rejected_before_sync(
    tmp_path: Path, node_id: str,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'bad-node.sqlite3').as_posix()}")
    database.migrate()
    remote = RemoteLoRADiscovery(
        LoRARegistry(database), CharacterRegistry(database),
    )
    candidate = _attestation(loras=()).model_copy(update={"node_id": node_id})
    result = remote.sync(candidate)
    assert result.failed[0][1] == "stale_or_untrusted_inventory"
    assert not result.discovered and not result.removed
    database.dispose()


def test_casefold_duplicate_remote_asset_names_block_partial_updates(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'duplicates.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    remote = RemoteLoRADiscovery(registry, CharacterRegistry(database))
    first = RenderLoRAInventoryItem(
        name="Character.safetensors", path="D:/loras/Character.safetensors",
        relative_path="Character.safetensors", sha256="a" * 64, bytes=1,
    )
    second = first.model_copy(update={
        "name": "character.safetensors",
        "relative_path": "character.safetensors",
        "sha256": "b" * 64,
    })
    result = remote.sync(_attestation(loras=(first, second)))
    assert not result.discovered and not result.removed
    assert result.failed[0][1] == "stale_or_untrusted_inventory"
    assert not registry.list()
    database.dispose()


def test_remote_lora_scan_requests_fresh_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.render_node.client as client_module
    from artifex.config.models import RenderNodeConfig, RenderNodesConfig
    from artifex.loras.remote import RenderAwareLoRADiscovery

    database = Database(f"sqlite:///{(tmp_path / 'fresh-scan.sqlite3').as_posix()}")
    database.migrate()
    called: list[tuple[str, bool]] = []

    def get_attestation(node_id: str, config: RenderNodeConfig, *, fresh: bool = False):
        called.append((node_id, fresh))
        return _attestation(loras=()).model_copy(update={"node_id": node_id})

    monkeypatch.setattr(client_module, "fetch_render_attestation", get_attestation)
    nodes = RenderNodesConfig(
        primary="gpu-box",
        nodes={"gpu-box": RenderNodeConfig(
            base_url="http://127.0.0.1:8188",
            attestation_url="http://127.0.0.1:8190",
        )},
    )
    scanner = RenderAwareLoRADiscovery(
        LoRARegistry(database), CharacterRegistry(database), nodes,
    )
    result = scanner.scan([])
    assert not result.failed
    assert called == [("gpu-box", True)]
    database.dispose()


@pytest.mark.parametrize(
    "failure",
    ["regular_file_root", "root_walk_permission_error", "file_stat_race"],
)
def test_pc_b_incomplete_root_scan_preserves_registered_remote_loras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """A bad root or disappearing file must not look like a complete deletion."""
    import json

    from artifex.config.models import ArtifexSettings
    from artifex.render_node import attestation as module

    monkeypatch.setattr(module, "_nvidia_gpus", lambda: ())
    root = tmp_path / "loras"
    root.mkdir()
    safetensors = root / "character.safetensors"
    header = json.dumps({"__metadata__": {"trigger": "character"}}).encode()
    safetensors.write_bytes(len(header).to_bytes(8, "little") + header)
    settings = ArtifexSettings()
    settings.render_agent.lora_roots = (root,)
    first = module.build_attestation(settings)
    assert len(first.loras) == 1 and not first.inventory_errors

    database = Database(f"sqlite:///{(tmp_path / 'root-failure.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    remote = RemoteLoRADiscovery(registry, CharacterRegistry(database))
    profile = remote.sync(first).discovered[0]
    original_checksum = profile.checksum

    if failure == "regular_file_root":
        wrong_root = tmp_path / "mistaken-root.safetensors"
        wrong_root.write_bytes(b"not a folder")
        settings.render_agent.lora_roots = (wrong_root,)
        expected_code = "lora_root_not_directory"
    elif failure == "root_walk_permission_error":
        walk = Path.rglob

        def fail_root_walk(path: Path, pattern: str):
            if path == root:
                raise PermissionError("private root path")
            return walk(path, pattern)

        monkeypatch.setattr(Path, "rglob", fail_root_walk)
        expected_code = "lora_root_unreadable"
    else:
        checksum = module.sha256_file

        def vanish_after_checksum(path: Path) -> str:
            digest = checksum(path)
            path.unlink()
            return digest

        monkeypatch.setattr(module, "sha256_file", vanish_after_checksum)
        expected_code = "lora_asset_unreadable"

    incomplete = module.build_attestation(settings)
    assert not incomplete.loras
    assert len(incomplete.inventory_errors) == 1
    assert incomplete.inventory_errors[0]["error"] == expected_code
    assert str(tmp_path) not in incomplete.model_dump_json()

    result = remote.sync(incomplete)
    assert not result.removed
    assert result.failed
    assert registry.require(profile.id).state is LoRAState.DISCOVERED
    assert registry.require(profile.id).checksum == original_checksum
    database.dispose()



def test_remote_lora_sync_late_conflict_rolls_back_earlier_updates(
    tmp_path: Path,
) -> None:
    """Even a failure on the second row cannot partially update PC-A's DB."""
    from artifex.domain import LoRAProfile

    database = Database(f"sqlite:///{(tmp_path / 'atomic-remote.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    remote = RemoteLoRADiscovery(registry, CharacterRegistry(database))
    previous = RenderLoRAInventoryItem(
        name="a.safetensors", path="D:/loras/a.safetensors",
        relative_path="a.safetensors", sha256="a" * 64, bytes=128,
    )
    old = remote.sync(_attestation(loras=(previous,))).discovered[0]
    reserved_path = Path(".artifex-remote") / "gpu-box" / "z-conflict.safetensors"
    reserved = registry.upsert(
        LoRAProfile(id="reserved", path=reserved_path, source="local"),
    )
    changed = previous.model_copy(update={"sha256": "b" * 64})
    conflicting = RenderLoRAInventoryItem(
        name="z-conflict.safetensors", path="D:/loras/z-conflict.safetensors",
        relative_path="z-conflict.safetensors", sha256="c" * 64, bytes=128,
    )

    with pytest.raises(ValueError, match="LoRA path already belongs"):
        remote.sync(_attestation(loras=(changed, conflicting)))
    assert registry.require(old.id).checksum == "a" * 64
    assert registry.require(old.id).state is LoRAState.DISCOVERED
    assert registry.require(reserved.id).path == reserved.path
    assert len(registry.list()) == 2
    database.dispose()


def test_remote_lora_sync_duplicate_batch_rejected_without_any_commit(
    tmp_path: Path,
) -> None:
    """The registry's batch transaction validates all staged rows first."""
    from artifex.domain import LoRAProfile

    database = Database(f"sqlite:///{(tmp_path / 'duplicate-batch.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    a = LoRAProfile(id="one", path=tmp_path / "same.safetensors")
    b = LoRAProfile(id="two", path=tmp_path / "same.safetensors")
    with pytest.raises(ValueError, match="duplicate LoRA ID or path"):
        registry.upsert_many((a, b))
    assert not registry.list()
    database.dispose()


def test_remote_lora_sync_metadata_failure_has_no_partial_upserts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad later metadata/character match cannot commit earlier discoveries."""
    database = Database(f"sqlite:///{(tmp_path / 'matching-failure.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    characters = CharacterRegistry(database)
    remote = RemoteLoRADiscovery(registry, characters)
    counter = 0
    original_match = characters.match_text

    def failing_match(text: str) -> tuple[str, ...]:
        nonlocal counter
        counter += 1
        if counter == 2:
            raise ValueError("synthetic later item metadata failure")
        return original_match(text)

    monkeypatch.setattr(characters, "match_text", failing_match)
    one = RenderLoRAInventoryItem(
        name="first.safetensors", path="D:/loras/first.safetensors",
        relative_path="first.safetensors", sha256="a" * 64, bytes=1,
    )
    two = RenderLoRAInventoryItem(
        name="second.safetensors", path="D:/loras/second.safetensors",
        relative_path="second.safetensors", sha256="b" * 64, bytes=1,
    )
    with pytest.raises(ValueError, match="synthetic later item"):
        remote.sync(_attestation(loras=(one, two)))
    assert not registry.list()
    database.dispose()


def test_remote_lora_sync_removals_and_additions_commit_together(
    tmp_path: Path,
) -> None:
    """The normal complete-inventory behavior is preserved after batching."""
    database = Database(f"sqlite:///{(tmp_path / 'atomic-success.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    remote = RemoteLoRADiscovery(registry, CharacterRegistry(database))
    old = RenderLoRAInventoryItem(
        name="old.safetensors", path="D:/loras/old.safetensors",
        relative_path="old.safetensors", sha256="a" * 64, bytes=1,
    )
    current = remote.sync(_attestation(loras=(old,))).discovered[0]
    new = RenderLoRAInventoryItem(
        name="new.safetensors", path="D:/loras/new.safetensors",
        relative_path="new.safetensors", sha256="b" * 64, bytes=1,
    )
    result = remote.sync(_attestation(loras=(new,)))
    assert len(result.discovered) == 1
    assert len(result.removed) == 1
    assert registry.require(current.id).state is LoRAState.DISABLED
    assert registry.require(current.id).missing_since is not None
    assert registry.require(result.discovered[0].id).state is LoRAState.DISCOVERED
    database.dispose()
