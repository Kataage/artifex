from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from artifex.archive import PackArchive
from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow, SceneRow
from artifex.domain import PackState, SceneState


def test_archive_writes_reproduction_manifest_and_selected_output(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'archive.sqlite3').as_posix()}")
    database.migrate()
    now = datetime.now(UTC)
    image = tmp_path / "generated.png"
    image.write_bytes(b"image-bytes")

    with database.session() as session:
        session.add(
            PackRow(
                id="pack-1",
                state=PackState.EVALUATING.value,
                format_type="single_feature",
                payload_json={"plan": {"title": "test"}},
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            SceneRow(
                id="scene-1",
                pack_id="pack-1",
                ordinal=1,
                state=SceneState.ACCEPTED.value,
                publication_tier="member",
                payload_json={},
                selected_attempt_id="attempt-1",
            )
        )
        session.add(
            GenerationAttemptRow(
                id="attempt-1",
                scene_id="scene-1",
                ordinal=1,
                backend_status="completed",
                seed=123,
                prompt="prompt",
                negative_prompt="negative",
                provenance_json={
                    "workflow_template": "ilxl_base_v1",
                    "workflow_version": 1,
                    "checkpoint": "model.safetensors",
                    "output_paths": [str(image)],
                },
                error_json=None,
                created_at=now,
            )
        )

    result = PackArchive(database, tmp_path / "packs").finalize("pack-1")

    assert result.manifest_path.exists()
    assert len(result.copied_outputs) == 1
    assert result.copied_outputs[0].parent.name == "member"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["attempts"][0]["seed"] == 123
    assert manifest["attempts"][0]["provenance"]["workflow_template"] == "ilxl_base_v1"
    assert manifest["archived_outputs"][0]["selected"] is True

    with database.session() as session:
        pack = session.get(PackRow, "pack-1")
        assert pack is not None
        assert pack.checkpoint_json["archive_complete"] is True
        assert pack.payload_json["archive"]["manifest_sha256"] == result.manifest_sha256
    database.dispose()
