from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from artifex.characters import CharacterRegistry
from artifex.db import Database
from artifex.db.models import PackRow
from artifex.domain import CharacterProfile


def test_character_registry_loads_yaml_matches_unicode_and_records_use(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'characters.sqlite3').as_posix()}")
    database.migrate()
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "kanata.yaml").write_text(
        """
id: amane_kanata
display_name: 天音かなた
namespace: hololive
branch: jp
generation: "4"
enabled: true
canonical_tags:
  - amane_kanata
aliases:
  - amane kanata
  - 天音かなた
model_families:
  - ilxl
lora_policy: required
readiness: 0.9
""",
        encoding="utf-8",
    )

    registry = CharacterRegistry(database)
    assert registry.load_directories((profiles,)) == 1

    profile = registry.require("amane_kanata")
    assert profile.display_name == "天音かなた"
    assert registry.match_text("LoRA_天音かなた_ILXL") == ("amane_kanata",)
    assert registry.match_text("amane_kanata character model") == ("amane_kanata",)

    registry.record_use(("amane_kanata",))
    used = registry.require("amane_kanata")
    assert used.total_generated == 1
    assert used.last_used_at is not None
    database.dispose()


def test_pack_scoped_character_use_is_exactly_once_and_atomic(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'usage.sqlite3').as_posix()}")
    database.migrate()
    registry = CharacterRegistry(database)
    registry.upsert(
        CharacterProfile(
            id="char-one",
            display_name="One",
            namespace="unit-test",
            canonical_tags=("char_one",),
            model_families=("ilxl",),
        )
    )
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            PackRow(
                id="pack-usage",
                state="evaluating",
                format_type="single_feature",
                payload_json={},
                checkpoint_json={"archive_complete": True},
                created_at=now,
                updated_at=now,
            )
        )

    registry.record_use(("char-one",), pack_id="pack-usage")
    first = registry.require("char-one")
    registry.record_use(("char-one",), pack_id="pack-usage")
    assert registry.require("char-one").total_generated == first.total_generated == 1
    assert registry.require("char-one").last_used_at == first.last_used_at
    with database.session() as session:
        pack = session.get(PackRow, "pack-usage")
        assert pack is not None
        assert pack.checkpoint_json["character_usage_ids"] == ["char-one"]
    with pytest.raises(ValueError, match="different character roster"):
        registry.record_use(("char-two",), pack_id="pack-usage")
    database.dispose()
