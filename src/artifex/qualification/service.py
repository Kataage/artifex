from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from artifex.characters import CharacterRegistry
from artifex.comfy import WorkflowTemplateRegistry
from artifex.config.models import ArtifexSettings
from artifex.db import Database
from artifex.db.models import PackRow, SceneRow
from artifex.domain import PackState
from artifex.loras import LoRARegistry
from artifex.operations.doctor import DoctorReport
from artifex.qualification.models import (
    REQUIRED_STAGES,
    AssetDigest,
    QualificationSession,
    QualificationStage,
    QualificationStageEvidence,
    QualificationStatus,
)


_PACK_STAGES = frozenset(
    {
        QualificationStage.SINGLE_CHARACTER,
        QualificationStage.LORA_REQUIRED,
        QualificationStage.DUO,
        QualificationStage.GROUP,
        QualificationStage.PUBLIC_MEMBER,
        QualificationStage.SERIES_CONTINUATION,
        QualificationStage.FORCED_RETRY,
        QualificationStage.RESTART_GENERATION,
        QualificationStage.BACKEND_RECOVERY,
        QualificationStage.UNATTENDED_MULTI_PACK,
        QualificationStage.ARCHIVE_REPRODUCTION,
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _run_text(args: list[str], *, timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=(
                subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                if platform.system() == "Windows"
                else 0
            ),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _nvidia_gpus() -> tuple[dict[str, object], ...]:
    raw = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if raw is None:
        return ()
    result: list[dict[str, object]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=4)]
        if len(parts) != 5:
            continue
        index, name, driver, memory_mib, uuid = parts
        try:
            memory: object = int(memory_mib)
        except ValueError:
            memory = memory_mib
        result.append(
            {
                "index": index,
                "name": name,
                "driver_version": driver,
                "memory_total_mib": memory,
                "uuid": uuid,
            }
        )
    return tuple(result)


def _hash_file(path: Path, digest: Any) -> tuple[int, int]:
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, 1


def _digest_path(label: str, path: Path) -> AssetDigest:
    absolute = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    if absolute.is_file():
        total_bytes, file_count = _hash_file(absolute, digest)
    elif absolute.is_dir():
        files = sorted(
            (item for item in absolute.rglob("*") if item.is_file()),
            key=lambda item: item.as_posix().casefold(),
        )
        if not files:
            raise ValueError(f"asset directory is empty: {absolute}")
        for item in files:
            relative = item.relative_to(absolute).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            size, count = _hash_file(item, digest)
            total_bytes += size
            file_count += count
    else:
        raise ValueError(f"asset path is neither file nor directory: {absolute}")
    return AssetDigest(
        label=label,
        path=str(absolute),
        sha256=digest.hexdigest(),
        bytes=total_bytes,
        file_count=file_count,
    )


class QualificationService:
    def __init__(
        self,
        settings: ArtifexSettings,
        database: Database,
        characters: CharacterRegistry,
        loras: LoRARegistry,
    ) -> None:
        self._settings = settings
        self._database = database
        self._characters = characters
        self._loras = loras
        self._root = settings.qualification.evidence_dir.expanduser().resolve()

    def start(self, doctor: DoctorReport) -> QualificationSession:
        now = _utcnow()
        session_id = (
            now.strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid4().hex[:8]
        )
        assets: list[AssetDigest] = []
        asset_errors: dict[str, str] = {}
        for label, path in sorted(self._settings.qualification.asset_paths.items()):
            try:
                assets.append(_digest_path(label, path))
            except (OSError, ValueError) as exc:
                asset_errors[label] = str(exc)

        calibration = self._settings.evaluation.semantic_calibration_path
        if calibration.is_file():
            try:
                assets.append(_digest_path("semantic_calibration", calibration))
            except (OSError, ValueError) as exc:
                asset_errors["semantic_calibration"] = str(exc)

        environment = self._environment()
        present_labels = {asset.label for asset in assets}
        missing_labels = sorted(
            set(self._settings.qualification.required_asset_labels)
            - present_labels
        )
        requirements = {
            "native_windows": (
                not self._settings.qualification.require_native_windows
                or environment["os"]["system"] == "Windows"
            ),
            "uv": (
                not self._settings.qualification.require_uv
                or environment.get("uv_version") is not None
            ),
            "nvidia_gpu": (
                not self._settings.qualification.require_nvidia_gpu
                or bool(environment.get("nvidia_gpus"))
            ),
            "asset_hashes": not missing_labels and not asset_errors,
        }
        environment["requirements"] = requirements
        environment["missing_asset_labels"] = missing_labels
        environment["asset_errors"] = asset_errors

        stages = {
            stage.value: QualificationStageEvidence(stage=stage)
            for stage in REQUIRED_STAGES
        }
        stages[QualificationStage.DOCTOR.value] = QualificationStageEvidence(
            stage=QualificationStage.DOCTOR,
            status=(
                QualificationStatus.PASS
                if doctor.ready
                else QualificationStatus.FAIL
            ),
            recorded_at=now,
            details={
                "doctor_ready": doctor.ready,
                "environment_requirements": requirements,
            },
            note=(
                "doctor and native baseline passed"
                if doctor.ready and all(requirements.values())
                else "doctor or native baseline is not ready"
            ),
        )
        if not self._settings.discord.enabled:
            stages[
                QualificationStage.DISCORD_CONTROLS.value
            ] = QualificationStageEvidence(
                stage=QualificationStage.DISCORD_CONTROLS,
                status=QualificationStatus.SKIPPED,
                recorded_at=now,
                note="Discord is disabled in production configuration.",
                details={"discord_enabled": False},
            )

        session = QualificationSession(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            hostname=socket.gethostname(),
            environment=environment,
            configuration=self._configuration_snapshot(),
            workflow=self._workflow_snapshot(),
            assets=tuple(assets),
            loras=tuple(self._lora_snapshot()),
            doctor_ready=doctor.ready and all(requirements.values()),
            doctor=doctor.model_dump(mode="json"),
            stages=stages,
        )
        self._write(session)
        return session

    def load(self, session_id: str) -> QualificationSession:
        path = self._path(session_id)
        if not path.is_file():
            raise KeyError(f"unknown qualification session: {session_id}")
        return QualificationSession.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def record(
        self,
        session_id: str,
        stage: QualificationStage,
        *,
        status: QualificationStatus,
        pack_ids: tuple[str, ...] = (),
        note: str | None = None,
        details: dict[str, object] | None = None,
    ) -> QualificationSession:
        if stage is QualificationStage.DOCTOR:
            raise ValueError("doctor stage is captured by qualify start")
        session = self.load(session_id)
        payload = dict(details or {})
        if status is QualificationStatus.PASS:
            if stage in _PACK_STAGES and not pack_ids:
                raise ValueError(
                    f"{stage.value} PASS requires at least one finalized Pack ID"
                )
            verified = self._verify_stage(stage, pack_ids, payload)
            if verified:
                payload["verified_packs"] = verified
        elif status is QualificationStatus.SKIPPED:
            if not (
                stage is QualificationStage.DISCORD_CONTROLS
                and not self._settings.discord.enabled
            ):
                raise ValueError(
                    "only disabled Discord qualification may be skipped"
                )

        evidence = QualificationStageEvidence(
            stage=stage,
            status=status,
            recorded_at=_utcnow(),
            pack_ids=tuple(dict.fromkeys(pack_ids)),
            note=note,
            details=payload,
        )
        stages = dict(session.stages)
        stages[stage.value] = evidence
        updated = session.model_copy(
            update={
                "updated_at": _utcnow(),
                "stages": stages,
            }
        )
        self._write(updated)
        return updated

    def verify(self, session_id: str) -> dict[str, object]:
        session = self.load(session_id)
        issues: list[str] = []
        requirements = session.environment.get("requirements")
        if not isinstance(requirements, dict) or not all(
            value is True for value in requirements.values()
        ):
            issues.append("native environment requirements are not all satisfied")
        if not session.doctor_ready:
            issues.append("doctor baseline is not ready")

        for stage in REQUIRED_STAGES:
            evidence = session.stage(stage)
            if stage is QualificationStage.DISCORD_CONTROLS:
                acceptable = (
                    evidence.status is QualificationStatus.PASS
                    or (
                        not self._settings.discord.enabled
                        and evidence.status is QualificationStatus.SKIPPED
                    )
                )
            else:
                acceptable = evidence.status is QualificationStatus.PASS
            if not acceptable:
                issues.append(
                    f"stage {stage.value} is {evidence.status.value}"
                )
                continue
            if evidence.status is QualificationStatus.PASS:
                try:
                    self._verify_stage(
                        stage,
                        evidence.pack_ids,
                        dict(evidence.details),
                    )
                except (KeyError, OSError, ValueError) as exc:
                    issues.append(f"stage {stage.value}: {exc}")

        asset_labels = {asset.label for asset in session.assets}
        for label in self._settings.qualification.required_asset_labels:
            if label not in asset_labels:
                issues.append(f"required asset hash is missing: {label}")

        return {
            "session_id": session.session_id,
            "ready": not issues,
            "issues": issues,
            "stages": {
                key: value.status.value
                for key, value in session.stages.items()
            },
            "evidence_path": str(self._path(session.session_id)),
        }

    def _verify_stage(
        self,
        stage: QualificationStage,
        pack_ids: tuple[str, ...],
        details: dict[str, object],
    ) -> list[dict[str, object]]:
        packs = [self._pack_evidence(pack_id) for pack_id in pack_ids]
        if stage is QualificationStage.SINGLE_CHARACTER:
            self._require_character_count(packs, exact=1)
        elif stage is QualificationStage.LORA_REQUIRED:
            self._require_lora_evidence(packs)
        elif stage is QualificationStage.DUO:
            self._require_character_count(packs, exact=2)
        elif stage is QualificationStage.GROUP:
            self._require_character_count(packs, minimum=3)
        elif stage is QualificationStage.PUBLIC_MEMBER:
            tiers = {
                tier
                for pack in packs
                for tier in pack["publication_tiers"]
            }
            if not {"public", "member"} <= tiers:
                raise ValueError(
                    "public_member requires both public and member Scenes"
                )
        elif stage is QualificationStage.SERIES_CONTINUATION:
            if len(packs) < 2:
                raise ValueError(
                    "series_continuation requires at least two finalized Packs"
                )
            series_ids = {pack["series_id"] for pack in packs}
            if None in series_ids or len(series_ids) != 1:
                raise ValueError(
                    "series_continuation Packs must share one non-null Series ID"
                )
        elif stage is QualificationStage.FORCED_RETRY:
            if not any(pack["retry_count"] > 0 for pack in packs):
                raise ValueError("forced_retry has no persisted retry history")
        elif stage is QualificationStage.RESTART_GENERATION:
            if details.get("recovered_after_restart") is not True:
                raise ValueError(
                    "restart_generation requires recovered_after_restart=true"
                )
            if details.get("duplicate_submissions") != 0:
                raise ValueError(
                    "restart_generation requires duplicate_submissions=0"
                )
        elif stage is QualificationStage.BACKEND_RECOVERY:
            if details.get("comfy_recovered") is not True:
                raise ValueError("backend_recovery requires comfy_recovered=true")
            if details.get("llm_recovered") is not True:
                raise ValueError("backend_recovery requires llm_recovered=true")
        elif stage is QualificationStage.UNATTENDED_MULTI_PACK:
            if len(packs) < 3:
                raise ValueError(
                    "unattended_multi_pack requires at least three finalized Packs"
                )
        elif stage is QualificationStage.OVERNIGHT_SOAK:
            duration = details.get("duration_hours")
            if not isinstance(duration, int | float):
                raise ValueError("overnight_soak requires numeric duration_hours")
            if duration < self._settings.qualification.minimum_soak_hours:
                raise ValueError(
                    "overnight_soak duration is below configured minimum"
                )
            if details.get("fatal_errors", 0) != 0:
                raise ValueError("overnight_soak requires fatal_errors=0")
            for key in (
                "peak_vram_mib",
                "peak_ram_mib",
                "minimum_free_disk_gib",
                "telemetry_rows",
            ):
                if key not in details:
                    raise ValueError(
                        f"overnight_soak requires resource evidence: {key}"
                    )
        elif stage is QualificationStage.DISCORD_CONTROLS:
            if self._settings.discord.enabled:
                commands = details.get("verified_commands")
                required = {
                    "status",
                    "pause",
                    "resume",
                    "approve",
                    "reject",
                    "retry",
                }
                if not isinstance(commands, list | tuple):
                    raise ValueError(
                        "discord_controls requires verified_commands"
                    )
                if not required <= {str(item) for item in commands}:
                    raise ValueError(
                        "discord_controls is missing required verified commands"
                    )
        elif stage is QualificationStage.ARCHIVE_REPRODUCTION:
            if details.get("reproduction_verified") is not True:
                raise ValueError(
                    "archive_reproduction requires reproduction_verified=true"
                )
            if details.get("hash_match") is not True:
                raise ValueError(
                    "archive_reproduction requires hash_match=true"
                )
        return packs

    def _pack_evidence(self, pack_id: str) -> dict[str, object]:
        with self._database.session() as session:
            pack = session.get(PackRow, pack_id)
            if pack is None:
                raise KeyError(f"unknown Pack: {pack_id}")
            if pack.state != PackState.FINALIZED.value:
                raise ValueError(
                    f"Pack {pack_id} is not finalized: {pack.state}"
                )
            if pack.checkpoint_json.get("archive_complete") is not True:
                raise ValueError(
                    f"Pack {pack_id} has no archive_complete checkpoint"
                )
            archive = pack.payload_json.get("archive")
            if not isinstance(archive, dict):
                raise ValueError(f"Pack {pack_id} has no archive metadata")
            manifest_raw = archive.get("manifest_path")
            if not isinstance(manifest_raw, str):
                raise ValueError(
                    f"Pack {pack_id} has no archive manifest path"
                )
            manifest = Path(manifest_raw).expanduser()
            if not manifest.is_file():
                raise OSError(
                    f"Pack {pack_id} archive manifest is missing: {manifest}"
                )
            scenes = session.scalars(
                select(SceneRow)
                .where(SceneRow.pack_id == pack_id)
                .order_by(SceneRow.ordinal.asc())
            ).all()
            character_ids: set[str] = set()
            tiers: list[str] = []
            retry_count = 0
            lora_ids: set[str] = set()
            for scene in scenes:
                tiers.append(scene.publication_tier)
                plan = scene.payload_json.get("plan")
                if isinstance(plan, dict):
                    raw_ids = plan.get("character_ids")
                    if isinstance(raw_ids, list | tuple):
                        character_ids.update(str(value) for value in raw_ids)
                history = scene.payload_json.get("retry_history")
                if isinstance(history, list):
                    retry_count += len(history)
                lora_plan = scene.payload_json.get("lora_plan")
                if isinstance(lora_plan, dict):
                    entries = lora_plan.get("entries")
                    if isinstance(entries, list):
                        for entry in entries:
                            if isinstance(entry, dict) and entry.get("lora_id"):
                                lora_ids.add(str(entry["lora_id"]))
            return {
                "pack_id": pack.id,
                "series_id": pack.series_id,
                "concept_id": pack.concept_id,
                "format": pack.format_type,
                "character_ids": sorted(character_ids),
                "publication_tiers": tiers,
                "retry_count": retry_count,
                "lora_ids": sorted(lora_ids),
                "manifest_path": str(manifest.resolve()),
            }

    @staticmethod
    def _require_character_count(
        packs: list[dict[str, object]],
        *,
        exact: int | None = None,
        minimum: int | None = None,
    ) -> None:
        for pack in packs:
            raw = pack["character_ids"]
            count = len(raw) if isinstance(raw, list) else 0
            if exact is not None and count != exact:
                raise ValueError(
                    f"Pack {pack['pack_id']} has {count} characters; "
                    f"expected exactly {exact}"
                )
            if minimum is not None and count < minimum:
                raise ValueError(
                    f"Pack {pack['pack_id']} has {count} characters; "
                    f"expected at least {minimum}"
                )

    def _require_lora_evidence(
        self,
        packs: list[dict[str, object]],
    ) -> None:
        for pack in packs:
            raw_ids = pack["character_ids"]
            character_ids = (
                [str(value) for value in raw_ids]
                if isinstance(raw_ids, list)
                else []
            )
            if not character_ids:
                raise ValueError(
                    f"Pack {pack['pack_id']} has no character evidence"
                )
            required = [
                character_id
                for character_id in character_ids
                if self._characters.require(character_id).lora_policy.value
                == "required"
            ]
            if not required:
                raise ValueError(
                    f"Pack {pack['pack_id']} has no REQUIRED-LoRA character"
                )
            if not pack["lora_ids"]:
                raise ValueError(
                    f"Pack {pack['pack_id']} has no persisted LoRA plan"
                )

    def _environment(self) -> dict[str, object]:
        uv_version = _run_text(["uv", "--version"])
        git_commit = _run_text(["git", "rev-parse", "HEAD"])
        return {
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
            },
            "uv_version": uv_version,
            "git_commit": git_commit,
            "nvidia_gpus": list(_nvidia_gpus()),
            "container_required": False,
        }

    def _configuration_snapshot(self) -> dict[str, object]:
        settings = self._settings
        return {
            "llm": {
                "backend": settings.llm.backend,
                "base_url": settings.llm.base_url,
                "model": settings.llm.model,
                "context_window_tokens": settings.llm.context_window_tokens,
            },
            "comfyui": {
                "base_url": settings.comfyui.base_url,
                "output_dir": (
                    None
                    if settings.comfyui.output_dir is None
                    else str(settings.comfyui.output_dir)
                ),
                "default_template": settings.comfyui.default_template,
                "release_vram_after_attempt": (
                    settings.comfyui.release_vram_after_attempt
                ),
            },
            "production": {
                "model_family": settings.production.model_family,
                "checkpoint": settings.production.checkpoint,
                "repair_workflow_template": (
                    settings.production.repair_workflow_template
                ),
                "width": settings.production.width,
                "height": settings.production.height,
                "batch_size": settings.production.batch_size,
            },
            "evaluation": {
                "semantic_provider": settings.evaluation.semantic_provider,
                "semantic_model": settings.evaluation.semantic_model,
                "semantic_revision": settings.evaluation.semantic_revision,
                "semantic_device": settings.evaluation.semantic_device,
                "semantic_calibration_path": str(
                    settings.evaluation.semantic_calibration_path
                ),
                "vision_base_url": settings.evaluation.vision_base_url,
                "vision_model": settings.evaluation.vision_model,
            },
            "catalog": {
                "enabled_characters": len(
                    self._characters.list(enabled_only=True)
                ),
                "profile_dirs": [
                    str(path) for path in settings.characters.profile_dirs
                ],
            },
            "loras": {
                "roots": [str(path) for path in settings.loras.roots],
                "production_count": sum(
                    1
                    for profile in self._loras.list()
                    if profile.state.value == "production"
                ),
            },
            "research": {
                "provider_order": list(settings.research.provider_order),
                "required_for_ideation": settings.research.required_for_ideation,
            },
            "trends": {
                "current_web_enabled": settings.trends.current_web_enabled,
                "tag_trends_enabled": settings.trends.tag_trends_enabled,
                "seasonal_enabled": settings.trends.seasonal_enabled,
            },
            "discord": {
                "enabled": settings.discord.enabled,
                "guild_id": settings.discord.guild_id,
                "channel_id": settings.discord.channel_id,
            },
            "patreon": {
                "enabled": settings.patreon.enabled,
                "default_archetype": settings.patreon.default_archetype,
                "performance_enabled": settings.patreon.performance.enabled,
            },
            "rights": {
                "default_profile": settings.rights.default_profile,
                "platform_profile": settings.rights.platform_profile,
                "profile_dirs": [
                    str(path) for path in settings.rights.profile_dirs
                ],
            },
        }

    def _workflow_snapshot(self) -> dict[str, object]:
        registry = WorkflowTemplateRegistry.with_packaged_templates()
        result: dict[str, object] = {}
        for template_id in dict.fromkeys(
            (
                self._settings.comfyui.default_template,
                self._settings.production.repair_workflow_template,
            )
        ):
            template = registry.require(template_id)
            result[template_id] = {
                "version": template.version,
                "model_family": template.model_family,
                "source_sha256": getattr(template, "source_sha256", None),
            }
        return result

    def _lora_snapshot(self) -> list[dict[str, object]]:
        return [
            {
                "id": profile.id,
                "path": str(profile.path),
                "checksum": profile.checksum,
                "state": profile.state.value,
                "type": profile.lora_type,
                "target_character_ids": list(profile.target_character_ids),
                "readiness": profile.readiness,
                "last_validation_at": (
                    None
                    if profile.last_validation_at is None
                    else profile.last_validation_at.isoformat()
                ),
                "last_validation_run_id": profile.last_validation_run_id,
            }
            for profile in self._loras.list()
        ]

    def _path(self, session_id: str) -> Path:
        if not session_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in session_id
        ):
            raise ValueError("invalid qualification session ID")
        return self._root / session_id / "qualification.json"

    def _write(self, session: QualificationSession) -> None:
        path = self._path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                session.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
