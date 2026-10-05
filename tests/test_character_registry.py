from __future__ import annotations

from pathlib import Path

from artifex.characters import CharacterRegistry
from artifex.db import Database


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
