from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.characters import CharacterRegistry
from artifex.config.models import (
    ArtifexSettings,
    QualificationConfig,
    RenderNodeConfig,
    RenderNodesConfig,
    StorageConfig,
)
from artifex.db import Database
from artifex.db.models import AgentEventRow, GenerationAttemptRow, PackRow, SceneRow
from artifex.domain import (
    CharacterProfile,
    LoRAPolicy,
    LoRAProfile,
    LoRAState,
    PackState,
    SceneState,
)
from artifex.loras import LoRARegistry
from artifex.operations.doctor import DoctorCheck, DoctorReport
from artifex.operations.health import ComponentHealth, ComponentState, HealthReport
from artifex.qualification import (
    QualificationService,
    QualificationStage,
    QualificationStatus,
)
from artifex.render_node import RenderAssetDigest, RenderNodeAttestation
from artifex.series import SeriesRepository


def _settings(tmp_path: Path, asset: Path) -> ArtifexSettings:
    return ArtifexSettings(
        storage=StorageConfig(
            database_url=f"sqlite:///{(tmp_path / 'qualification.sqlite3').as_posix()}",
            packs_dir=tmp_path / "packs",
            minimum_free_gib=0,
        ),
        qualification=QualificationConfig(
            evidence_dir=tmp_path / "qualification",
            minimum_soak_hours=8,
            asset_paths={"production_checkpoint": asset},
            required_asset_labels=("production_checkpoint",),
            require_native_windows=False,
            require_uv=False,
            require_nvidia_gpu=False,
            require_lora_validation_evidence=True,
            require_stable_asset_hashes=True,
            require_stable_workflow_snapshot=True,
        ),
    )


def _doctor(*, ready: bool = True) -> DoctorReport:
    state = ComponentState.HEALTHY if ready else ComponentState.UNHEALTHY
    return DoctorReport(
        health=HealthReport(
            components=(
                ComponentHealth(
                    name="baseline",
                    state=state,
                    detail="test baseline",
                    blocking=True,
                ),
            )
        ),
        checks=(
            DoctorCheck(
                name="test",
                ready=ready,
                detail="test doctor check",
            ),
        ),
    )


def _service(
    tmp_path: Path,
) -> tuple[
    QualificationService,
    Database,
    CharacterRegistry,
    LoRARegistry,
    ArtifexSettings,
    Path,
]:
    asset = tmp_path / "checkpoint.safetensors"
    asset.write_bytes(b"checkpoint-v1")
    settings = _settings(tmp_path, asset)
    database = Database(settings.storage.database_url)
    database.migrate()
    characters = CharacterRegistry(database)
    loras = LoRARegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-1",
            display_name="Character 1",
            namespace="qualification",
            canonical_tags=("char_1",),
            lora_policy=LoRAPolicy.REQUIRED,
            preferred_lora_ids=("lora-1",),
            readiness=1.0,
        )
    )
    for index in (2, 3):
        characters.upsert(
            CharacterProfile(
                id=f"char-{index}",
                display_name=f"Character {index}",
                namespace="qualification",
                canonical_tags=(f"char_{index}",),
                readiness=1.0,
            )
        )

    lora_path = tmp_path / "lora-1.safetensors"
    lora_path.write_bytes(b"validated-lora")
    loras.upsert(
        LoRAProfile(
            id="lora-1",
            path=lora_path,
            state=LoRAState.PRODUCTION,
            target_character_ids=("char-1",),
            model_families=("ilxl",),
            trigger_tags=("char_1",),
            readiness=1.0,
            checksum=hashlib.sha256(lora_path.read_bytes()).hexdigest(),
            last_validation_at=datetime.now(UTC),
            last_validation_run_id="validation-run-1",
        )
    )
    return (
        QualificationService(settings, database, characters, loras),
        database,
        characters,
        loras,
        settings,
        asset,
    )


def _seed_pack(
    database: Database,
    tmp_path: Path,
    pack_id: str,
    *,
    character_ids: tuple[str, ...],
    tiers: tuple[str, ...] = ("public",),
    series_id: str | None = None,
    retry: bool = False,
    prompt_id: str | None = None,
    lora_ids: tuple[str, ...] = (),
) -> str:
    now = datetime.now(UTC)
    manifest = tmp_path / f"{pack_id}-manifest.json"
    manifest.write_text(
        f'{{"pack_id":"{pack_id}","schema_version":1}}',
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with database.session() as session:
        session.add(
            PackRow(
                id=pack_id,
                concept_id=None,
                series_id=series_id,
                state=PackState.FINALIZED.value,
                format_type="qualification",
                payload_json={
                    "archive": {
                        "manifest_path": str(manifest),
                        "manifest_sha256": digest,
                    }
                },
                checkpoint_json={
                    "archive_complete": True,
                    "archive_manifest_sha256": digest,
                },
                created_at=now,
                updated_at=now,
            )
        )
        for ordinal, tier in enumerate(tiers, start=1):
            scene_id = f"{pack_id}-scene-{ordinal}"
            attempt_id = f"{pack_id}-attempt-{ordinal}"
            payload: dict[str, object] = {
                "plan": {"character_ids": list(character_ids)},
                "lora_plan": {
                    "entries": [
                        {"lora_id": lora_id}
                        for lora_id in lora_ids
                    ]
                },
            }
            if retry:
                payload["retry_history"] = [
                    {
                        "applied_actions": ["change_seed"],
                        "before_digest": "before",
                        "after_digest": "after",
                    }
                ]
            session.add(
                SceneRow(
                    id=scene_id,
                    pack_id=pack_id,
                    ordinal=ordinal,
                    state=SceneState.ACCEPTED.value,
                    publication_tier=tier,
                    payload_json=payload,
                    selected_attempt_id=attempt_id,
                )
            )
            session.add(
                GenerationAttemptRow(
                    id=attempt_id,
                    scene_id=scene_id,
                    parent_attempt_id=None,
                    ordinal=1,
                    backend_status="completed",
                    seed=ordinal,
                    prompt="qualification prompt",
                    negative_prompt="",
                    provenance_json=(
                        {"comfy_prompt_id": prompt_id}
                        if prompt_id is not None
                        else {}
                    ),
                    error_json=None,
                    created_at=now,
                )
            )
    return pack_id


def _event(
    database: Database,
    event_type: str,
    payload: dict[str, object],
    *,
    offset_seconds: int,
) -> None:
    with database.session() as session:
        session.add(
            AgentEventRow(
                event_type=event_type,
                severity="info",
                payload_json=payload,
                created_at=datetime.now(UTC) + timedelta(seconds=offset_seconds),
            )
        )


def test_full_real_machine_ladder_can_only_verify_with_persisted_evidence(
    tmp_path: Path,
) -> None:
    service, database, _, _, _, _ = _service(tmp_path)
    session = service.start(_doctor())
    assert session.doctor_ready is True

    SeriesRepository(database).create(
        title="Qualification Series",
        character_ids=("char-1",),
        series_id="series-qualification",
    )

    single = _seed_pack(
        database,
        tmp_path,
        "single",
        character_ids=("char-2",),
    )
    required = _seed_pack(
        database,
        tmp_path,
        "required",
        character_ids=("char-1",),
        lora_ids=("lora-1",),
    )
    duo = _seed_pack(
        database,
        tmp_path,
        "duo",
        character_ids=("char-1", "char-2"),
    )
    group = _seed_pack(
        database,
        tmp_path,
        "group",
        character_ids=("char-1", "char-2", "char-3"),
    )
    public_member = _seed_pack(
        database,
        tmp_path,
        "public-member",
        character_ids=("char-2",),
        tiers=("public", "member"),
    )
    series_a = _seed_pack(
        database,
        tmp_path,
        "series-a",
        character_ids=("char-1",),
        series_id="series-qualification",
    )
    series_b = _seed_pack(
        database,
        tmp_path,
        "series-b",
        character_ids=("char-1",),
        series_id="series-qualification",
    )
    retry = _seed_pack(
        database,
        tmp_path,
        "retry",
        character_ids=("char-2",),
        retry=True,
    )
    restart = _seed_pack(
        database,
        tmp_path,
        "restart",
        character_ids=("char-2",),
        prompt_id="prompt-restart",
    )
    backend = _seed_pack(
        database,
        tmp_path,
        "backend",
        character_ids=("char-2",),
    )

    _event(
        database,
        "recovery.pack_checked",
        {"pack_id": restart},
        offset_seconds=1,
    )
    for offset, component, state in (
        (2, "comfyui", "unhealthy"),
        (3, "comfyui", "healthy"),
        (4, "llm", "unhealthy"),
        (5, "llm", "healthy"),
    ):
        _event(
            database,
            "health.component_changed",
            {"component": component, "state": state},
            offset_seconds=offset,
        )

    records = (
        (QualificationStage.SINGLE_CHARACTER, (single,), {}),
        (QualificationStage.LORA_REQUIRED, (required,), {}),
        (QualificationStage.DUO, (duo,), {}),
        (QualificationStage.GROUP, (group,), {}),
        (QualificationStage.PUBLIC_MEMBER, (public_member,), {}),
        (
            QualificationStage.SERIES_CONTINUATION,
            (series_a, series_b),
            {},
        ),
        (QualificationStage.FORCED_RETRY, (retry,), {}),
        (QualificationStage.RESTART_GENERATION, (restart,), {}),
        (QualificationStage.BACKEND_RECOVERY, (backend,), {}),
        (
            QualificationStage.UNATTENDED_MULTI_PACK,
            (single, duo, group),
            {},
        ),
        (
            QualificationStage.OVERNIGHT_SOAK,
            (),
            {
                "duration_hours": 8.5,
                "fatal_errors": 0,
                "peak_vram_mib": 11000,
                "peak_ram_mib": 24000,
                "minimum_free_disk_gib": 100.0,
                "telemetry_rows": 5000,
            },
        ),
        (
            QualificationStage.ARCHIVE_REPRODUCTION,
            (single,),
            {
                "reproduction_verified": True,
                "hash_match": True,
            },
        ),
    )
    for stage, pack_ids, details in records:
        service.record(
            session.session_id,
            stage,
            status=QualificationStatus.PASS,
            pack_ids=pack_ids,
            details=details,
        )

    verified = service.verify(session.session_id)

    assert verified["ready"] is True
    assert verified["issues"] == []
    assert verified["stages"]["discord_controls"] == "skipped"
    database.dispose()


def test_verify_fails_if_qualified_asset_changes(tmp_path: Path) -> None:
    service, database, _, _, _, asset = _service(tmp_path)
    session = service.start(_doctor())
    asset.write_bytes(b"checkpoint-v2")

    verified = service.verify(session.session_id)

    assert verified["ready"] is False
    assert any("hash changed" in issue for issue in verified["issues"])
    database.dispose()


def test_archive_stage_rejects_tampered_manifest(tmp_path: Path) -> None:
    service, database, _, _, _, _ = _service(tmp_path)
    session = service.start(_doctor())
    pack_id = _seed_pack(
        database,
        tmp_path,
        "archive-tamper",
        character_ids=("char-2",),
    )
    with database.session() as db_session:
        row = db_session.get(PackRow, pack_id)
        assert row is not None
        manifest = Path(row.payload_json["archive"]["manifest_path"])
    manifest.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="manifest SHA"):
        service.record(
            session.session_id,
            QualificationStage.ARCHIVE_REPRODUCTION,
            status=QualificationStatus.PASS,
            pack_ids=(pack_id,),
            details={
                "reproduction_verified": True,
                "hash_match": True,
            },
        )
    database.dispose()


def test_lora_required_stage_rejects_unvalidated_production_lora(
    tmp_path: Path,
) -> None:
    service, database, _, loras, _, _ = _service(tmp_path)
    session = service.start(_doctor())
    current = loras.require("lora-1")
    loras.upsert(
        current.model_copy(
            update={
                "last_validation_at": None,
                "last_validation_run_id": None,
            }
        )
    )
    pack_id = _seed_pack(
        database,
        tmp_path,
        "required-invalid",
        character_ids=("char-1",),
        lora_ids=("lora-1",),
    )

    with pytest.raises(ValueError, match="validation timestamp"):
        service.record(
            session.session_id,
            QualificationStage.LORA_REQUIRED,
            status=QualificationStatus.PASS,
            pack_ids=(pack_id,),
        )
    database.dispose()


def test_restart_stage_rejects_duplicate_prompt_submission(tmp_path: Path) -> None:
    service, database, _, _, _, _ = _service(tmp_path)
    session = service.start(_doctor())
    now = datetime.now(UTC)
    pack_id = _seed_pack(
        database,
        tmp_path,
        "restart-duplicate",
        character_ids=("char-2",),
        prompt_id="same-prompt",
    )
    with database.session() as db_session:
        scene = db_session.get(SceneRow, f"{pack_id}-scene-1")
        assert scene is not None
        db_session.add(
            GenerationAttemptRow(
                id="restart-duplicate-attempt-2",
                scene_id=scene.id,
                parent_attempt_id=scene.selected_attempt_id,
                ordinal=2,
                backend_status="completed",
                seed=2,
                prompt="qualification prompt retry",
                negative_prompt="",
                provenance_json={"comfy_prompt_id": "same-prompt"},
                error_json=None,
                created_at=now,
            )
        )
    _event(
        database,
        "recovery.pack_checked",
        {"pack_id": pack_id},
        offset_seconds=1,
    )

    with pytest.raises(ValueError, match="duplicate Comfy prompt IDs"):
        service.record(
            session.session_id,
            QualificationStage.RESTART_GENERATION,
            status=QualificationStatus.PASS,
            pack_ids=(pack_id,),
        )
    database.dispose()


def test_remote_render_asset_is_attested_and_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ArtifexSettings(
        render_nodes=RenderNodesConfig(
            primary="gpu-box",
            nodes={
                "gpu-box": RenderNodeConfig(
                    base_url="http://render.test:8188",
                    output_mode="api",
                    attestation_url="http://render.test:8190",
                )
            },
        ),
        storage=StorageConfig(
            database_url=f"sqlite:///{(tmp_path / 'remote-qualification.sqlite3').as_posix()}",
            packs_dir=tmp_path / "packs",
            minimum_free_gib=0,
        ),
        qualification=QualificationConfig(
            evidence_dir=tmp_path / "qualification",
            asset_paths={},
            required_asset_labels=("production_checkpoint",),
            require_native_windows=False,
            require_uv=False,
            require_nvidia_gpu=False,
            require_lora_validation_evidence=True,
            require_stable_asset_hashes=True,
            require_stable_workflow_snapshot=True,
        ),
    )
    database = Database(settings.storage.database_url)
    database.migrate()
    characters = CharacterRegistry(database)
    loras = LoRARegistry(database)
    current_digest = {"value": "a" * 64}

    def fake_attestation(
        node_id: str,
        _config: RenderNodeConfig,
    ) -> RenderNodeAttestation:
        assert node_id == "gpu-box"
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
            assets=(
                RenderAssetDigest(
                    label="production_checkpoint",
                    path=r"D:\ComfyUI\models\checkpoints\production.safetensors",
                    sha256=current_digest["value"],
                    bytes=123456,
                ),
            ),
        )

    monkeypatch.setattr(
        "artifex.qualification.service.fetch_render_attestation",
        fake_attestation,
    )
    service = QualificationService(settings, database, characters, loras)
    session = service.start(_doctor())

    assert session.doctor_ready is True
    remote = next(
        asset for asset in session.assets if asset.label == "production_checkpoint"
    )
    assert remote.source == "render_node"
    assert remote.node_id == "gpu-box"
    assert remote.sha256 == "a" * 64

    current_digest["value"] = "b" * 64
    verified = service.verify(session.session_id)

    assert verified["ready"] is False
    assert any(
        "production_checkpoint hash changed during qualification" in issue
        for issue in verified["issues"]
    )
    database.dispose()
