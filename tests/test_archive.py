from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

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


def _archive_fixture(tmp_path: Path) -> tuple[Database, Path, PackArchive]:
    database = Database(f"sqlite:///{(tmp_path / 'repeat.sqlite3').as_posix()}")
    database.migrate()
    now = datetime.now(UTC)
    source = tmp_path / "source.png"
    source.write_bytes(b"first-original-image")
    with database.session() as session:
        session.add(
            PackRow(
                id="pack-repeat",
                state=PackState.EVALUATING.value,
                format_type="single_feature",
                payload_json={"plan": {"title": "repeat"}},
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            SceneRow(
                id="scene-repeat",
                pack_id="pack-repeat",
                ordinal=1,
                state=SceneState.ACCEPTED.value,
                publication_tier="public",
                payload_json={},
                selected_attempt_id="attempt-repeat",
            )
        )
        session.add(
            GenerationAttemptRow(
                id="attempt-repeat",
                scene_id="scene-repeat",
                ordinal=1,
                backend_status="completed",
                seed=42,
                prompt="portrait",
                negative_prompt="",
                provenance_json={"output_paths": [str(source)]},
                error_json=None,
                created_at=now,
            )
        )
    return database, source, PackArchive(database, tmp_path / "packs")


def test_archive_finalize_twice_has_stable_manifest_and_one_image(
    tmp_path: Path,
) -> None:
    database, source, archive = _archive_fixture(tmp_path)
    first = archive.finalize("pack-repeat")
    original_bytes = first.manifest_path.read_bytes()
    # Original ComfyUI file may no longer be available after archiving.
    source.unlink()
    second = archive.finalize("pack-repeat")
    assert second.manifest_sha256 == first.manifest_sha256
    assert second.manifest_path.read_bytes() == original_bytes
    assert second.copied_outputs == first.copied_outputs
    assert len(list((second.root / "public").iterdir())) == 1
    database.dispose()


def test_archive_recovers_crash_after_output_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _, archive = _archive_fixture(tmp_path)
    original = archive._write_or_verify
    crashed = False

    def fail_before_manifest(path: Path, contents: bytes) -> None:
        nonlocal crashed
        if path.name == "pack.json" and not crashed:
            crashed = True
            raise RuntimeError("controller died before manifest commit")
        original(path, contents)

    monkeypatch.setattr(archive, "_write_or_verify", fail_before_manifest)
    with pytest.raises(RuntimeError, match="controller died"):
        archive.finalize("pack-repeat")
    monkeypatch.setattr(archive, "_write_or_verify", original)
    public = next((tmp_path / "packs").rglob("public"))
    existing = list(public.iterdir())
    assert len(existing) == 1
    second = archive.finalize("pack-repeat")
    assert list(public.iterdir()) == existing
    assert second.copied_outputs == tuple(existing)
    assert json.loads(second.manifest_path.read_text(encoding="utf-8"))[
        "archived_outputs"
    ][0]["sha256"]
    database.dispose()


def test_archive_fails_closed_when_partial_copy_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, _, archive = _archive_fixture(tmp_path)
    original = archive._write_or_verify

    def fail_before_manifest(path: Path, contents: bytes) -> None:
        if path.name == "pack.json":
            raise RuntimeError("controller died before manifest commit")
        original(path, contents)

    monkeypatch.setattr(archive, "_write_or_verify", fail_before_manifest)
    with pytest.raises(RuntimeError, match="controller died"):
        archive.finalize("pack-repeat")
    monkeypatch.setattr(archive, "_write_or_verify", original)
    public = next((tmp_path / "packs").rglob("public"))
    existing = list(public.iterdir())
    assert len(existing) == 1
    existing[0].write_bytes(b"modified-archive")
    with pytest.raises(ValueError, match="differs"):
        archive.finalize("pack-repeat")
    assert existing[0].read_bytes() == b"modified-archive"
    database.dispose()


def test_archive_fails_closed_when_selected_source_disappears(
    tmp_path: Path,
) -> None:
    database, source, archive = _archive_fixture(tmp_path)
    source.unlink()
    with pytest.raises(FileNotFoundError, match="source missing"):
        archive.finalize("pack-repeat")
    with database.session() as session:
        pack = session.get(PackRow, "pack-repeat")
        assert pack is not None
        assert pack.checkpoint_json.get("archive_complete") is not True
    database.dispose()


def test_completed_archive_rejects_tampered_manifest_and_image(
    tmp_path: Path,
) -> None:
    database, _, archive = _archive_fixture(tmp_path)
    result = archive.finalize("pack-repeat")
    result.copied_outputs[0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="altered"):
        archive.finalize("pack-repeat")
    result.copied_outputs[0].write_bytes(b"first-original-image")
    result.manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts"):
        archive.finalize("pack-repeat")
    database.dispose()
