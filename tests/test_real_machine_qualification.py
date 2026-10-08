from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifex.characters import CharacterRegistry
from artifex.config.models import (
    ArtifexSettings,
    DiscordConfig,
    QualificationConfig,
    RenderNodeConfig,
    RenderNodesConfig,
    StorageConfig,
)
from artifex.db import Database
from artifex.db.models import AgentEventRow, GenerationAttemptRow, PackRow, SceneRow
from artifex.discord.audit import DiscordQualificationAudit
from artifex.discord.models import CommandResponse, OperatorContext
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
from artifex.production import GeneratedBatch
from artifex.qualification import (
    QualificationService,
    QualificationStage,
    QualificationStatus,
)
from artifex.qualification.archive_reproduction import (
    file_sha256,
    reproduce_archived_attempt,
)
from artifex.qualification.collector import QualificationEvidenceCollector
from artifex.qualification.soak_observer import SoakSample, observe_soak
from artifex.render_node import RenderAssetDigest, RenderNodeAttestation
from artifex.series import SeriesRepository
from artifex.telemetry import TelemetryRepository


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


def _bound_soak_trace(
    service: QualificationService,
    settings: ArtifexSettings,
    session_id: str,
    *,
    evidence_name: str = "soak.jsonl",
) -> tuple[str, Path]:
    """Synthetic 8-hour clock trace for binding tests, never hardware evidence."""
    # This fixture contains no actual CIM/Task Scheduler/PC-B measurements.
    # Production two-PC soak-run retains the strict owner policy.
    settings.qualification.require_renderer_owner_observation = False
    settings.render_nodes = RenderNodesConfig(
        primary="gpu-b",
        nodes={
            "gpu-b": RenderNodeConfig(
                base_url="http://test-renderer:8188",
                attestation_url="http://test-renderer:8190",
                output_mode="api",
            )
        },
    )
    started = datetime.now(UTC) - timedelta(hours=8, minutes=3)
    session = service.load(session_id)
    # Only this deterministic fixture backdates the session and snapshot.
    session = session.model_copy(update={
        "created_at": started - timedelta(minutes=1),
        "configuration": service._configuration_snapshot(),
    })
    service._write(session)
    clock = [0.0]

    def now() -> datetime:
        return started + timedelta(seconds=clock[0])

    def sample(_: ArtifexSettings, elapsed: float) -> SoakSample:
        hours = int(elapsed // 3600)
        return SoakSample(
            observed_at=started + timedelta(seconds=elapsed),
            elapsed_seconds=elapsed,
            llm_ok=True,
            comfyui_ok=True,
            renderer_ok=True,
            gpu_used_mib=5000,
            ram_used_mib=10000,
            free_disk_gib=50,
            telemetry_rows=10 + hours,
            severe_events=0,
            finalized_packs=10 + hours,
            asset_sha256={
                "production_checkpoint": "a" * 64,
                "refiner_checkpoint": "b" * 64,
                "vae": "c" * 64,
                "upscale_model": "d" * 64,
            },
        )

    path = settings.qualification.evidence_dir / evidence_name
    report = observe_soak(
        settings,
        output=path,
        duration_hours=8,
        interval_seconds=3600,
        qualification_session_id=session_id,
        sample=sample,
        monotonic=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        now=now,
    )
    assert report.ready_for_soak_review
    return report.evidence_sha256, path


def _reproduce_fixture(
    service: QualificationService,
    database: Database,
    tmp_path: Path,
    settings: ArtifexSettings,
    session_id: str,
    pack_id: str,
    *,
    match: bool = True,
) -> tuple[Path, object]:
    """Fake replay is test-only: production CLI uses real ComfyUI."""
    from PIL import Image

    image_path = tmp_path / f"{pack_id}-archived.png"
    Image.new("RGB", (5, 5), (35, 55, 75)).save(image_path)
    compiled = {
        "positive_prompt": "qualification prompt",
        "negative_prompt": "",
        "positive_tags": [],
        "negative_tags": [],
        "provenance": {
            "adapter_id": "test", "adapter_version": "1",
            "model_family": "ilxl", "character_ids": ["char-2"],
            "lora_ids": [], "lora_weights": [],
        },
    }
    lora = {"model_family": "ilxl", "character_ids": ["char-2"], "entries": []}
    backend_provenance = {
        "backend": "comfyui", "base_url": "http://test", "render_node_id": "legacy",
        "checkpoint": "checkpoint.safetensors", "width": 1024, "height": 1024,
        "batch_size": 1, "model_family": "ilxl",
        "refiner_checkpoint": "refiner", "upscale_model": "upscaler",
        "base_steps": 48, "base_cfg": 5.5, "base_sampler": "sampler",
        "base_scheduler": "karras", "workflow_template": "illust_main_v1",
        "workflow_version": 1,
    }
    with database.session() as db_session:
        pack = db_session.get(PackRow, pack_id)
        assert pack is not None
        manifest_path = Path(pack.payload_json["archive"]["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scene_id = f"{pack_id}-scene-1"
        attempt_id = f"{pack_id}-attempt-1"
        manifest["archived_outputs"] = [{
            "selected": True, "scene_id": scene_id, "attempt_id": attempt_id,
            "archived_path": str(image_path.resolve()),
        }]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_hash = file_sha256(manifest_path)
        pack.payload_json = {
            **pack.payload_json,
            "archive": {**pack.payload_json["archive"], "manifest_sha256": manifest_hash},
        }
        pack.checkpoint_json = {
            **pack.checkpoint_json, "archive_manifest_sha256": manifest_hash,
        }
        row = db_session.get(GenerationAttemptRow, attempt_id)
        assert row is not None
        row.provenance_json = {
            **backend_provenance,
            "compiled_prompt": compiled,
            "lora_plan": lora,
            "comfy_prompt_id": "original-" + pack_id,
        }

    class FakeReplayBackend:
        def provenance(self) -> dict[str, object]:
            return backend_provenance

        async def generate(self, request: object, *, on_submitted: object) -> GeneratedBatch:
            from artifex.production import GenerationRequest
            assert isinstance(request, GenerationRequest)
            assert request.seed == 1
            assert request.compiled.positive_prompt == "qualification prompt"
            assert request.workflow_template_id == "illust_main_v1"
            assert callable(on_submitted)
            replay_id = "replay-" + pack_id
            on_submitted(replay_id)
            output = tmp_path / f"{request.attempt_id}.png"
            if match:
                shutil.copyfile(image_path, output)
            else:
                Image.new("RGB", (5, 5), (255, 0, 0)).save(output)
            return GeneratedBatch(prompt_id=replay_id, output_paths=(output,))

    path, proof = asyncio.run(reproduce_archived_attempt(
        settings, database, FakeReplayBackend(),
        session_id=session_id, pack_id=pack_id,
        output_root=settings.qualification.evidence_dir,
    ))
    return path, proof


def test_collect_qualifications_uses_only_new_finalized_packs_and_is_idempotent(
    tmp_path: Path,
) -> None:
    service, database, _, _, _, _ = _service(tmp_path)
    session = service.start(_doctor())
    SeriesRepository(database).create(
        title="Auto Collect Series",
        character_ids=("char-1",),
        series_id="auto-series",
    )
    old = _seed_pack(database, tmp_path, "pre-session", character_ids=("char-2",))
    with database.session() as db_session:
        row = db_session.get(PackRow, old)
        assert row is not None
        row.created_at = session.created_at - timedelta(minutes=1)
    new = {
        "single": _seed_pack(database, tmp_path, "auto-single", character_ids=("char-2",)),
        "required": _seed_pack(
            database, tmp_path, "auto-lora",
            character_ids=("char-1",), lora_ids=("lora-1",),
        ),
        "duo": _seed_pack(
            database, tmp_path, "auto-duo", character_ids=("char-1", "char-2"),
        ),
        "group": _seed_pack(
            database, tmp_path, "auto-group",
            character_ids=("char-1", "char-2", "char-3"),
        ),
        "public_member": _seed_pack(
            database, tmp_path, "auto-tiers",
            character_ids=("char-2",), tiers=("public", "member"),
        ),
        "series_a": _seed_pack(
            database, tmp_path, "auto-series-a",
            character_ids=("char-1",), series_id="auto-series",
        ),
        "series_b": _seed_pack(
            database, tmp_path, "auto-series-b",
            character_ids=("char-1",), series_id="auto-series",
        ),
        "retry": _seed_pack(
            database, tmp_path, "auto-retry", character_ids=("char-2",), retry=True,
        ),
        "restart": _seed_pack(
            database, tmp_path, "auto-restart", character_ids=("char-2",),
            prompt_id="unique-restart",
        ),
        "backend": _seed_pack(
            database, tmp_path, "auto-backend", character_ids=("char-2",),
        ),
    }
    collector = QualificationEvidenceCollector(service, database)

    preview = collector.collect(session.session_id)
    by_stage = {item["stage"]: item for item in preview["stages"]}
    assert preview["mode"] == "preview"
    assert preview["finalized_packs_scanned"] == len(new)
    assert preview["scan_truncated"] is False
    assert by_stage["single_character"]["state"] == "ready"
    assert old not in by_stage["single_character"]["pack_ids"]
    assert by_stage["series_continuation"]["pack_ids"] == [
        new["series_a"], new["series_b"],
    ]
    assert by_stage["restart_generation"]["state"] == "missing"
    assert by_stage["backend_recovery"]["state"] == "missing"
    assert service.load(session.session_id).stage(QualificationStage.SINGLE_CHARACTER).status is QualificationStatus.PENDING

    first = collector.collect(session.session_id, apply=True)
    done = {item["stage"]: item for item in first["stages"]}
    assert done["single_character"]["state"] == "recorded"
    assert done["lora_required"]["state"] == "recorded"
    assert done["unattended_multi_pack"]["state"] == "recorded"
    assert done["restart_generation"]["state"] == "missing"
    assert done["backend_recovery"]["state"] == "missing"
    assert service.load(session.session_id).stage(QualificationStage.OVERNIGHT_SOAK).status is QualificationStatus.PENDING
    assert service.load(session.session_id).stage(QualificationStage.ARCHIVE_REPRODUCTION).status is QualificationStatus.PENDING

    _event(database, "recovery.pack_checked", {"pack_id": new["restart"]}, offset_seconds=1)
    for offset, component, state in (
        (2, "comfyui", "unhealthy"),
        (3, "comfyui", "healthy"),
        (4, "llm", "unhealthy"),
        (5, "llm", "healthy"),
    ):
        _event(
            database, "health.component_changed",
            {"component": component, "state": state},
            offset_seconds=offset,
        )
    second = collector.collect(session.session_id, apply=True)
    stages = {item["stage"]: item for item in second["stages"]}
    assert stages["single_character"]["state"] == "already_recorded"
    assert stages["restart_generation"]["state"] == "recorded"
    assert stages["backend_recovery"]["state"] == "recorded"
    assert second["production_qualified"] is False
    database.dispose()


def test_collect_qualification_excludes_bad_manifests_and_rejects_stale_baselines(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    session = service.start(_doctor())
    bad = _seed_pack(
        database, tmp_path, "auto-broken", character_ids=("char-2",),
    )
    with database.session() as db_session:
        row = db_session.get(PackRow, bad)
        assert row is not None
        path = Path(row.payload_json["archive"]["manifest_path"])
    path.write_text('{"tampered":true}', encoding="utf-8")
    collector = QualificationEvidenceCollector(service, database)
    preview = collector.collect(session.session_id)
    assert preview["valid_packs"] == 0
    assert preview["invalid_packs"][0]["pack_id"] == bad
    assert all(
        item["state"] != "ready"
        for item in preview["stages"]
    )
    with pytest.raises(ValueError, match="max_packs"):
        collector.collect(session.session_id, max_packs=0)
    settings.production.checkpoint = "changed-checkpoint"
    with pytest.raises(ValueError, match="configuration changed"):
        collector.collect(session.session_id, apply=True)
    assert service.load(session.session_id).stage(QualificationStage.SINGLE_CHARACTER).status is QualificationStatus.PENDING
    database.dispose()


def test_collect_discord_only_after_real_telemetry(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    settings.discord = DiscordConfig(
        enabled=True, guild_id=111, channel_id=222, allowed_user_ids=(333,),
    )
    session = service.start(_doctor())
    collector = QualificationEvidenceCollector(service, database)
    initial = collector.collect(session.session_id, apply=True)
    assert next(item for item in initial["stages"] if item["stage"] == "discord_controls")["state"] == "missing"

    audit = DiscordQualificationAudit(settings.discord, TelemetryRepository(database))
    for index, command in enumerate(("status", "pause", "resume", "approve", "reject", "retry")):
        message = (
            "Artifex paused." if command == "pause"
            else "Artifex resumed." if command == "resume"
            else "Success."
        )
        assert audit.record_completed(
            command=command,
            response=CommandResponse(ok=True, message=message),
            operator=OperatorContext(user_id=333),
            interaction_id=400 + index, guild_id=111, channel_id=222,
        )
    result = collector.collect(session.session_id, apply=True)
    discord = next(item for item in result["stages"] if item["stage"] == "discord_controls")
    assert discord["state"] == "recorded"
    assert discord["pack_ids"] == []
    assert service.load(session.session_id).stage(QualificationStage.DISCORD_CONTROLS).status is QualificationStatus.PASS
    database.dispose()


def test_full_real_machine_ladder_can_only_verify_with_persisted_evidence(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    # This older deterministic test validates the original 14 stage
    # validators with simulated evidence, not the new live PC-B owner check.
    # Production two-PC qualification keeps the owner check enabled.
    settings.qualification.require_renderer_owner_observation = False
    session = service.start(_doctor())
    assert session.doctor_ready is True
    digest, evidence_path = _bound_soak_trace(
        service, settings, session.session_id,
    )

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

    reproduction_path, reproduction_proof = _reproduce_fixture(
        service, database, tmp_path, settings, session.session_id, single,
    )
    assert reproduction_proof.exact_match

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
                "evidence_path": str(evidence_path),
                "evidence_sha256": digest,
            },
        ),
        (
            QualificationStage.ARCHIVE_REPRODUCTION,
            (single,),
            {
                "evidence_path": str(reproduction_path),
                "evidence_sha256": file_sha256(reproduction_path),
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

    assert verified["ready"] is True, verified["issues"]
    assert verified["issues"] == []
    assert verified["stages"]["discord_controls"] == "skipped"
    database.dispose()



@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("manual", "manually supplied claims"),
        ("missing", "requires evidence_path"),
        ("wrong_digest", "SHA-256 mismatch"),
        ("wrong_session", "another qualification session"),
        ("missing_session", "another qualification session"),
        ("external_path", "qualification.evidence_dir"),
    ],
)
def test_overnight_soak_rejects_unbound_and_unverified_claims(
    tmp_path: Path,
    case: str,
    expected: str,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    session = service.start(_doctor())
    digest, path = _bound_soak_trace(service, settings, session.session_id)
    details: dict[str, object] = {
        "evidence_path": str(path),
        "evidence_sha256": digest,
    }
    if case == "manual":
        details = {"duration_hours": 8, "peak_vram_mib": 7000}
    elif case == "missing":
        details = {"evidence_sha256": digest}
    elif case == "wrong_digest":
        details["evidence_sha256"] = "0" * 64
    elif case in ("wrong_session", "missing_session"):
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        header["qualification_session_id"] = (
            "other-session" if case == "wrong_session" else None
        )
        lines[0] = json.dumps(header)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        details["evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif case == "external_path":
        external = tmp_path / "external-soak.jsonl"
        external.write_bytes(path.read_bytes())
        details["evidence_path"] = str(external)
    with pytest.raises(ValueError, match=expected):
        service.record(
            session.session_id,
            QualificationStage.OVERNIGHT_SOAK,
            status=QualificationStatus.PASS,
            details=details,
        )
    assert service.load(session.session_id).stage(
        QualificationStage.OVERNIGHT_SOAK
    ).status is QualificationStatus.PENDING
    database.dispose()


def test_overnight_soak_is_rechecked_and_revoked_after_tampering(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    session = service.start(_doctor())
    digest, path = _bound_soak_trace(service, settings, session.session_id)
    service.record(
        session.session_id, QualificationStage.OVERNIGHT_SOAK,
        status=QualificationStatus.PASS,
        details={"evidence_path": str(path), "evidence_sha256": digest},
    )
    recorded = service.load(session.session_id).stage(
        QualificationStage.OVERNIGHT_SOAK
    )
    assert recorded.details["observed_metrics"]["verified_completed_packs"] == 8
    assert recorded.details["observed_metrics"]["production_qualified"] is False

    path.write_bytes(path.read_bytes() + b"\n")
    rechecked = service.verify(session.session_id)
    assert rechecked["ready"] is False
    assert any(
        "overnight_soak" in issue and "SHA-256 mismatch" in issue
        for issue in rechecked["issues"]
    )
    database.dispose()


def test_overnight_soak_detects_session_metric_and_configuration_drift(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    session = service.start(_doctor())
    digest, path = _bound_soak_trace(service, settings, session.session_id)
    service.record(
        session.session_id, QualificationStage.OVERNIGHT_SOAK,
        status=QualificationStatus.PASS,
        details={"evidence_path": str(path), "evidence_sha256": digest},
    )
    recorded = service.load(session.session_id)
    evidence = recorded.stage(QualificationStage.OVERNIGHT_SOAK)
    mutated = dict(evidence.details)
    metrics = dict(mutated["observed_metrics"])
    metrics["verified_completed_packs"] = 999
    mutated["observed_metrics"] = metrics
    stages = dict(recorded.stages)
    stages[QualificationStage.OVERNIGHT_SOAK.value] = evidence.model_copy(
        update={"details": mutated},
    )
    service._write(recorded.model_copy(update={"stages": stages}))
    issues = service.verify(session.session_id)["issues"]
    assert any("recorded metrics were modified" in issue for issue in issues)

    # Even intact evidence cannot qualify after controller config drift.
    stages[QualificationStage.OVERNIGHT_SOAK.value] = evidence
    service._write(recorded.model_copy(update={"stages": stages}))
    settings.llm.base_url = "http://unexpected-llm:1234"
    issues = service.verify(session.session_id)["issues"]
    assert any("observed a different configuration" in issue for issue in issues)
    database.dispose()


def test_soak_candidate_preflight_rejects_stale_or_completed_sessions(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    session = service.start(_doctor())
    service.require_soak_candidate(session)

    with pytest.raises(ValueError, match="doctor baseline"):
        service.require_soak_candidate(
            session.model_copy(update={"doctor_ready": False})
        )
    with pytest.raises(ValueError, match="another controller"):
        service.require_soak_candidate(
            session.model_copy(update={"hostname": "another-pc"})
        )
    with pytest.raises(ValueError, match="native requirements"):
        environment = dict(session.environment)
        environment["requirements"] = {"native_windows": False}
        service.require_soak_candidate(
            session.model_copy(update={"environment": environment})
        )

    stages = dict(session.stages)
    soak = session.stage(QualificationStage.OVERNIGHT_SOAK)
    stages[QualificationStage.OVERNIGHT_SOAK.value] = soak.model_copy(
        update={"status": QualificationStatus.PASS}
    )
    with pytest.raises(ValueError, match="already passed"):
        service.require_soak_candidate(
            session.model_copy(update={"stages": stages})
        )
    settings.llm.base_url = "http://wrong-llm:8899"
    with pytest.raises(ValueError, match="configuration changed"):
        service.require_soak_candidate(session)
    database.dispose()


def test_qualification_accepts_lora_promoted_after_session_started(
    tmp_path: Path,
) -> None:
    service, database, _, loras, _, _ = _service(tmp_path)
    session = service.start(_doctor())
    new_path = tmp_path / "promoted.safetensors"
    new_path.write_bytes(b"promoted-during-qualification")
    loras.upsert(
        LoRAProfile(
            id="new-promoted",
            path=new_path,
            state=LoRAState.PRODUCTION,
            target_character_ids=("char-1",),
            model_families=("ilxl",),
            readiness=1.0,
            checksum=hashlib.sha256(new_path.read_bytes()).hexdigest(),
            last_validation_at=datetime.now(UTC),
            last_validation_run_id="validation-after-start",
        )
    )

    assert service._verify_production_loras(session, {}) == []
    database.dispose()


def test_qualification_rejects_unvalidated_new_production_lora(
    tmp_path: Path,
) -> None:
    service, database, _, loras, _, _ = _service(tmp_path)
    session = service.start(_doctor())
    new_path = tmp_path / "unvalidated.safetensors"
    new_path.write_bytes(b"unvalidated")
    loras.upsert(
        LoRAProfile(
            id="new-unvalidated",
            path=new_path,
            state=LoRAState.PRODUCTION,
            target_character_ids=("char-1",),
            model_families=("ilxl",),
            readiness=1.0,
            checksum=hashlib.sha256(new_path.read_bytes()).hexdigest(),
            last_validation_at=session.created_at - timedelta(hours=1),
            last_validation_run_id="old-validation",
        )
    )

    issues = service._verify_production_loras(session, {})
    assert any("new-unvalidated" in issue and "validation evidence" in issue for issue in issues)
    database.dispose()


def test_verify_rejects_qualification_policy_downgrade(tmp_path: Path) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    session = service.start(_doctor())
    settings.qualification.require_stable_asset_hashes = False
    verified = service.verify(session.session_id)
    assert verified["ready"] is False
    assert "production configuration changed during qualification" in verified["issues"]
    database.dispose()


def test_verify_rejects_production_lora_disk_drift(tmp_path: Path) -> None:
    service, database, _, loras, _, _ = _service(tmp_path)
    session = service.start(_doctor())
    profile = loras.require("lora-1")
    profile.path.write_bytes(b"changed-lora-after-qualification-start")
    verified = service.verify(session.session_id)
    assert verified["ready"] is False
    assert any("production LoRA lora-1 on-disk hash changed" in item for item in verified["issues"])
    database.dispose()


def test_verify_fails_if_qualified_asset_changes(tmp_path: Path) -> None:
    service, database, _, _, _, asset = _service(tmp_path)
    session = service.start(_doctor())
    asset.write_bytes(b"checkpoint-v2")

    verified = service.verify(session.session_id)

    assert verified["ready"] is False
    assert any("hash changed" in issue for issue in verified["issues"])
    database.dispose()


def test_archive_reproduction_refuses_claims_and_mismatching_actual_image(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    session = service.start(_doctor())
    pack_id = _seed_pack(database, tmp_path, "archive-claim", character_ids=("char-2",))
    with pytest.raises(ValueError, match="manually supplied claims"):
        service.record(
            session.session_id, QualificationStage.ARCHIVE_REPRODUCTION,
            status=QualificationStatus.PASS, pack_ids=(pack_id,),
            details={"reproduction_verified": True, "hash_match": True},
        )
    path, proof = _reproduce_fixture(
        service, database, tmp_path, settings, session.session_id, pack_id,
        match=False,
    )
    assert not proof.exact_match
    with pytest.raises(ValueError, match="hash mismatch"):
        service.record(
            session.session_id, QualificationStage.ARCHIVE_REPRODUCTION,
            status=QualificationStatus.PASS, pack_ids=(pack_id,),
            details={"evidence_path": str(path), "evidence_sha256": file_sha256(path)},
        )
    database.dispose()


def test_archive_reproduction_rechecked_after_file_and_database_tampering(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    session = service.start(_doctor())
    pack_id = _seed_pack(database, tmp_path, "archive-proof", character_ids=("char-2",))
    path, proof = _reproduce_fixture(
        service, database, tmp_path, settings, session.session_id, pack_id,
    )
    assert proof.exact_match
    details = {"evidence_path": str(path), "evidence_sha256": file_sha256(path)}
    recorded = service.record(
        session.session_id, QualificationStage.ARCHIVE_REPRODUCTION,
        status=QualificationStatus.PASS, pack_ids=(pack_id,), details=details,
    )
    assert recorded.stage(QualificationStage.ARCHIVE_REPRODUCTION).status is QualificationStatus.PASS
    replay_path = Path(proof.reproduced_path)
    replay_path.write_bytes(b"tampered")
    checked = service.verify(session.session_id)
    assert not checked["ready"]
    assert any("archive_reproduction" in issue for issue in checked["issues"])
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


def test_discord_qualification_requires_real_delivered_interactions(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    settings.discord = DiscordConfig(
        enabled=True, guild_id=123, channel_id=456, allowed_user_ids=(789,),
    )
    session = service.start(_doctor())
    audit = DiscordQualificationAudit(settings.discord, TelemetryRepository(database))

    # Supplying historical manual assertions is never proof.
    with pytest.raises(ValueError, match="manually supplied claims"):
        service.record(
            session.session_id,
            QualificationStage.DISCORD_CONTROLS,
            status=QualificationStatus.PASS,
            details={"verified_commands": ["status", "pause", "resume", "approve", "reject", "retry"], "fake": True},
        )
    with pytest.raises(ValueError, match="lacks delivered authorized interactions"):
        service.record(
            session.session_id,
            QualificationStage.DISCORD_CONTROLS,
            status=QualificationStatus.PASS,
        )

    commands = ("status", "pause", "resume", "approve", "reject", "retry")
    for index, command in enumerate(commands, start=1):
        message = (
            "Artifex paused." if command == "pause" else
            "Artifex resumed." if command == "resume" else "Operation succeeded."
        )
        assert audit.record_completed(
            command=command,
            response=CommandResponse(ok=True, message=message),
            operator=OperatorContext(user_id=789),
            interaction_id=1000 + index,
            guild_id=123, channel_id=456,
        )
    recorded = service.record(
        session.session_id,
        QualificationStage.DISCORD_CONTROLS,
        status=QualificationStatus.PASS,
    )
    evidence = recorded.stage(QualificationStage.DISCORD_CONTROLS)
    assert evidence.details["verified_commands"] == sorted(commands)
    assert len(evidence.details["observed_interactions"]) == 6
    assert len({x["event_id"] for x in evidence.details["observed_interactions"]}) == 6

    # Revalidation re-queries the DB, not the supplied command strings.
    command_record = evidence.details["observed_interactions"][0]
    with database.session() as db_session:
        row = db_session.get(AgentEventRow, command_record["event_id"])
        assert row is not None
        row.payload_json = {**row.payload_json, "channel_id": 999}
    verified = service.verify(session.session_id)
    assert not verified["ready"]
    assert any("discord_controls" in item for item in verified["issues"])
    database.dispose()


def test_discord_qualification_filters_invalid_or_reused_interactions(
    tmp_path: Path,
) -> None:
    service, database, _, _, settings, _ = _service(tmp_path)
    settings.discord = DiscordConfig(
        enabled=True, guild_id=123, channel_id=456,
        allowed_user_ids=(789,), allowed_role_ids=(222,),
    )
    session = service.start(_doctor())
    audit = DiscordQualificationAudit(settings.discord, TelemetryRepository(database))

    assert not audit.record_completed(
        command="pause", response=CommandResponse(ok=True, message="Artifex is already paused."),
        operator=OperatorContext(user_id=789), interaction_id=1,
        guild_id=123, channel_id=456,
    )
    assert not audit.record_completed(
        command="pause", response=CommandResponse(ok=False, message="Failed"),
        operator=OperatorContext(user_id=789), interaction_id=2,
        guild_id=123, channel_id=456,
    )
    assert not audit.record_completed(
        command="status", response=CommandResponse(ok=True, message="OK"),
        operator=OperatorContext(user_id=789), interaction_id=3,
        guild_id=123, channel_id=999,
    )
    assert not audit.record_completed(
        command="status", response=CommandResponse(ok=True, message="OK"),
        operator=OperatorContext(user_id=999), interaction_id=4,
        guild_id=123, channel_id=456,
    )
    assert audit.record_completed(
        command="status", response=CommandResponse(ok=True, message="OK"),
        operator=OperatorContext(user_id=999, role_ids=(222,)), interaction_id=5,
        guild_id=123, channel_id=456,
    )
    # Reusing the same Discord interaction is not six distinct controls.
    for command in ("pause", "resume", "approve", "reject", "retry"):
        assert audit.record_completed(
            command=command,
            response=CommandResponse(
                ok=True,
                message=("Artifex paused." if command == "pause" else
                         "Artifex resumed." if command == "resume" else "OK"),
            ),
            operator=OperatorContext(user_id=789), interaction_id=5,
            guild_id=123, channel_id=456,
        )
    with pytest.raises(ValueError, match="lacks delivered authorized interactions"):
        service.record(
            session.session_id,
            QualificationStage.DISCORD_CONTROLS,
            status=QualificationStatus.PASS,
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
        *,
        fresh: bool = False,
    ) -> RenderNodeAttestation:
        assert fresh is True
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
