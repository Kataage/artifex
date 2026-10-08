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
from artifex.db.models import (
    AgentEventRow,
    GenerationAttemptRow,
    PackRow,
    SceneRow,
)
from artifex.domain import LoRAState, PackState
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
from artifex.render_node import RenderNodeAttestation, fetch_render_attestation

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
        session_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        assets: list[AssetDigest] = []
        asset_errors: dict[str, str] = {}

        for label, path in sorted(self._settings.qualification.asset_paths.items()):
            try:
                assets.append(_digest_path(label, path))
            except (OSError, ValueError) as exc:
                asset_errors[label] = str(exc)

        llm_model = self._llm_model_path()
        if (
            llm_model is not None
            and "llm_model" not in {asset.label for asset in assets}
        ):
            try:
                assets.append(_digest_path("llm_model", llm_model))
            except (OSError, ValueError) as exc:
                asset_errors["llm_model"] = str(exc)

        calibration = self._settings.evaluation.semantic_calibration_path
        if calibration.is_file():
            try:
                assets.append(_digest_path("semantic_calibration", calibration))
            except (OSError, ValueError) as exc:
                asset_errors["semantic_calibration"] = str(exc)

        remote_attestations, remote_errors = self._render_attestations()
        asset_errors.update(remote_errors)
        existing_labels = {asset.label for asset in assets}
        for node_id, attestation in remote_attestations.items():
            for remote in attestation.assets:
                if remote.label in existing_labels:
                    asset_errors[remote.label] = (
                        "asset label is provided by both controller and render node "
                        f"{node_id}"
                    )
                    continue
                assets.append(
                    AssetDigest(
                        label=remote.label,
                        path=remote.path,
                        sha256=remote.sha256,
                        bytes=remote.bytes,
                        file_count=remote.file_count,
                        source="render_node",
                        node_id=node_id,
                        attested_at=attestation.created_at,
                    )
                )
                existing_labels.add(remote.label)

        environment = self._environment()
        environment["render_nodes"] = {
            node_id: self._attestation_environment(attestation)
            for node_id, attestation in remote_attestations.items()
        }
        present_labels = {asset.label for asset in assets}
        missing_labels = sorted(
            set(self._settings.qualification.required_asset_labels)
            - present_labels
        )
        controller_os = environment.get("os")
        controller_windows = (
            isinstance(controller_os, dict)
            and controller_os.get("system") == "Windows"
        )
        primary = self._settings.render_nodes.primary_node()
        if primary is None:
            production_windows = controller_windows
            production_gpu = bool(environment.get("nvidia_gpus"))
        else:
            primary_id, _ = primary
            primary_attestation = remote_attestations.get(primary_id)
            production_windows = (
                controller_windows
                and primary_attestation is not None
                and primary_attestation.os.get("system") == "Windows"
            )
            production_gpu = bool(
                primary_attestation is not None
                and primary_attestation.nvidia_gpus
            )

        requirements = {
            "native_windows": (
                not self._settings.qualification.require_native_windows
                or production_windows
            ),
            "uv": (
                not self._settings.qualification.require_uv
                or environment.get("uv_version") is not None
            ),
            "nvidia_gpu": (
                not self._settings.qualification.require_nvidia_gpu
                or production_gpu
            ),
            "render_attestation": not remote_errors,
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
            verified = self._verify_stage(
                stage,
                pack_ids,
                payload,
                since=session.created_at,
            )
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
        # Freeze the production configuration used for the qualification run.
        # The production LoRA count can increase as validation completes, but
        # changing the backend, assets, identity policies or safety gates cannot.
        original_configuration = json.loads(json.dumps(session.configuration))
        current_configuration = self._configuration_snapshot()
        for snapshot in (original_configuration, current_configuration):
            if isinstance(snapshot.get("loras"), dict):
                snapshot["loras"].pop("production_count", None)
        if original_configuration != current_configuration:
            issues.append("production configuration changed during qualification")

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
                        since=session.created_at,
                    )
                except (KeyError, OSError, ValueError) as exc:
                    issues.append(f"stage {stage.value}: {exc}")

        asset_labels = {asset.label for asset in session.assets}
        for label in self._settings.qualification.required_asset_labels:
            if label not in asset_labels:
                issues.append(f"required asset hash is missing: {label}")

        remote_cache: dict[str, RenderNodeAttestation] = {}
        if self._settings.qualification.require_stable_asset_hashes:
            for asset in session.assets:
                if asset.source == "render_node":
                    if asset.node_id is None:
                        issues.append(
                            f"asset {asset.label} has no render node provenance"
                        )
                        continue
                    try:
                        attestation = remote_cache.get(asset.node_id)
                        if attestation is None:
                            node = self._settings.render_nodes.nodes.get(asset.node_id)
                            if node is None:
                                raise ValueError(
                                    f"render node is no longer configured: {asset.node_id}"
                                )
                            attestation = fetch_render_attestation(asset.node_id, node, fresh=True)
                            remote_cache[asset.node_id] = attestation
                        candidates = {
                            item.label: item for item in attestation.assets
                        }
                        remote = candidates.get(asset.label)
                        if remote is None:
                            raise ValueError(
                                f"asset disappeared from render node {asset.node_id}"
                            )
                        current = AssetDigest(
                            label=remote.label,
                            path=remote.path,
                            sha256=remote.sha256,
                            bytes=remote.bytes,
                            file_count=remote.file_count,
                            source="render_node",
                            node_id=asset.node_id,
                            attested_at=attestation.created_at,
                        )
                    except Exception as exc:  # noqa: BLE001
                        issues.append(
                            f"asset {asset.label} cannot be revalidated: {exc}"
                        )
                        continue
                else:
                    try:
                        current = _digest_path(asset.label, Path(asset.path))
                    except (OSError, ValueError) as exc:
                        issues.append(
                            f"asset {asset.label} cannot be revalidated: {exc}"
                        )
                        continue

                if current.sha256 != asset.sha256:
                    issues.append(
                        f"asset {asset.label} hash changed during qualification: "
                        f"{asset.sha256} -> {current.sha256}"
                    )
                if current.bytes != asset.bytes:
                    issues.append(
                        f"asset {asset.label} byte size changed during qualification"
                    )
                if current.file_count != asset.file_count:
                    issues.append(
                        f"asset {asset.label} file count changed during qualification"
                    )

        issues.extend(self._verify_production_loras(session, remote_cache))

        if self._settings.qualification.require_stable_workflow_snapshot:
            try:
                current_workflow = self._workflow_snapshot()
            except (KeyError, ValueError) as exc:
                issues.append(f"workflow snapshot cannot be revalidated: {exc}")
            else:
                if current_workflow != session.workflow:
                    issues.append(
                        "workflow template snapshot changed during qualification"
                    )

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
        *,
        since: datetime,
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
            tiers: set[str] = set()
            for pack in packs:
                raw_tiers = pack.get("publication_tiers")
                if not isinstance(raw_tiers, list):
                    raise TypeError(
                        f"Pack {pack.get('pack_id')} has invalid publication tier evidence"
                    )
                tiers.update(str(tier) for tier in raw_tiers)
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
            retry_counts = [pack.get("retry_count") for pack in packs]
            if not any(
                isinstance(value, int) and value > 0
                for value in retry_counts
            ):
                raise ValueError("forced_retry has no persisted retry history")
        elif stage is QualificationStage.RESTART_GENERATION:
            repeated: set[str] = set()
            for pack in packs:
                raw_prompt_ids = pack.get("repeated_prompt_ids")
                if not isinstance(raw_prompt_ids, list):
                    raise TypeError(
                        f"Pack {pack.get('pack_id')} has invalid prompt ID evidence"
                    )
                repeated.update(str(value) for value in raw_prompt_ids)
            if repeated:
                raise ValueError(
                    "restart_generation detected duplicate Comfy prompt IDs: "
                    + ", ".join(sorted(str(value) for value in repeated))
                )
            recovered_pack_ids = self._recovery_checked_pack_ids(
                since=since
            )
            missing = set(pack_ids) - recovered_pack_ids
            if missing:
                raise ValueError(
                    "restart_generation lacks recovery.pack_checked telemetry for: "
                    + ", ".join(sorted(missing))
                )
            details["recovery_pack_checked"] = sorted(recovered_pack_ids)
            details["duplicate_prompt_ids"] = []
        elif stage is QualificationStage.BACKEND_RECOVERY:
            for component in ("comfyui", "llm"):
                transitions = self._health_transitions(
                    component,
                    since=since,
                )
                states = [state for _, state in transitions]
                try:
                    unhealthy_index = states.index("unhealthy")
                except ValueError as exc:
                    raise ValueError(
                        f"backend_recovery has no {component} unhealthy transition"
                    ) from exc
                if "healthy" not in states[unhealthy_index + 1 :]:
                    raise ValueError(
                        f"backend_recovery has no {component} recovery transition"
                    )
                details[f"{component}_transitions"] = [
                    {"at": at.isoformat(), "state": state}
                    for at, state in transitions
                ]
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
            if not all(pack.get("archive_manifest_verified") is True for pack in packs):
                raise ValueError(
                    "archive_reproduction requires intact archived manifest provenance"
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
                raise TypeError(f"Pack {pack_id} has no archive metadata")
            manifest_raw = archive.get("manifest_path")
            if not isinstance(manifest_raw, str):
                raise TypeError(
                    f"Pack {pack_id} has no archive manifest path"
                )
            manifest = Path(manifest_raw).expanduser()
            if not manifest.is_file():
                raise OSError(
                    f"Pack {pack_id} archive manifest is missing: {manifest}"
                )
            manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            payload_digest = archive.get("manifest_sha256")
            checkpoint_digest = pack.checkpoint_json.get(
                "archive_manifest_sha256"
            )
            if payload_digest != manifest_digest:
                raise ValueError(
                    f"Pack {pack_id} archive manifest SHA does not match payload provenance"
                )
            if checkpoint_digest != manifest_digest:
                raise ValueError(
                    f"Pack {pack_id} archive manifest SHA does not match checkpoint provenance"
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
                "manifest_sha256": manifest_digest,
                "archive_manifest_verified": True,
                **self._attempt_evidence(
                    session,
                    tuple(scene.id for scene in scenes),
                ),
            }

    @staticmethod
    def _attempt_evidence(
        session: Any,
        scene_ids: tuple[str, ...],
    ) -> dict[str, object]:
        if not scene_ids:
            return {
                "attempt_count": 0,
                "prompt_ids": [],
                "repeated_prompt_ids": [],
                "recovered_from_history_count": 0,
            }
        rows = session.scalars(
            select(GenerationAttemptRow)
            .where(GenerationAttemptRow.scene_id.in_(scene_ids))
            .order_by(
                GenerationAttemptRow.created_at.asc(),
                GenerationAttemptRow.id.asc(),
            )
        ).all()
        prompt_ids: list[str] = []
        recovered = 0
        for row in rows:
            prompt_id = row.provenance_json.get("comfy_prompt_id")
            if isinstance(prompt_id, str) and prompt_id:
                prompt_ids.append(prompt_id)
            if row.provenance_json.get("recovered_from_history") is True:
                recovered += 1
        seen: set[str] = set()
        repeated: set[str] = set()
        for prompt_id in prompt_ids:
            if prompt_id in seen:
                repeated.add(prompt_id)
            seen.add(prompt_id)
        return {
            "attempt_count": len(rows),
            "prompt_ids": prompt_ids,
            "repeated_prompt_ids": sorted(repeated),
            "recovered_from_history_count": recovered,
        }

    def _recovery_checked_pack_ids(
        self,
        *,
        since: datetime,
    ) -> set[str]:
        with self._database.session() as session:
            rows = session.scalars(
                select(AgentEventRow)
                .where(
                    AgentEventRow.event_type == "recovery.pack_checked",
                    AgentEventRow.created_at >= since,
                )
                .order_by(AgentEventRow.created_at.asc(), AgentEventRow.id.asc())
            ).all()
            return {
                str(row.payload_json["pack_id"])
                for row in rows
                if row.payload_json.get("pack_id")
            }

    def _health_transitions(
        self,
        component: str,
        *,
        since: datetime,
    ) -> list[tuple[datetime, str]]:
        with self._database.session() as session:
            rows = session.scalars(
                select(AgentEventRow)
                .where(
                    AgentEventRow.event_type == "health.component_changed",
                    AgentEventRow.created_at >= since,
                )
                .order_by(AgentEventRow.created_at.asc(), AgentEventRow.id.asc())
            ).all()
            return [
                (row.created_at, str(row.payload_json.get("state")))
                for row in rows
                if row.payload_json.get("component") == component
            ]

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
            raw_lora_ids = pack["lora_ids"]
            if not isinstance(raw_lora_ids, list) or not raw_lora_ids:
                raise ValueError(
                    f"Pack {pack['pack_id']} has no persisted LoRA plan"
                )
            for lora_id in raw_lora_ids:
                profile = self._loras.require(str(lora_id))
                if profile.state is not LoRAState.PRODUCTION:
                    raise ValueError(
                        f"LoRA {profile.id} is not in production state"
                    )
                if not set(required) & set(profile.target_character_ids):
                    raise ValueError(
                        f"LoRA {profile.id} does not map to a REQUIRED-LoRA "
                        f"character in Pack {pack['pack_id']}"
                    )
                if self._settings.qualification.require_lora_validation_evidence:
                    if profile.checksum is None:
                        raise ValueError(
                            f"LoRA {profile.id} is missing checksum evidence"
                        )
                    if profile.last_validation_at is None:
                        raise ValueError(
                            f"LoRA {profile.id} is missing validation timestamp"
                        )
                    if profile.last_validation_run_id is None:
                        raise ValueError(
                            f"LoRA {profile.id} is missing validation run evidence"
                        )

    def _verify_production_loras(
        self,
        session: QualificationSession,
        remote_cache: dict[str, RenderNodeAttestation],
    ) -> list[str]:
        issues: list[str] = []
        baseline = {
            str(item["id"]): item
            for item in session.loras
            if item.get("state") == LoRAState.PRODUCTION.value
        }
        current = {
            profile.id: profile
            for profile in self._loras.list(states=(LoRAState.PRODUCTION,))
        }
        for lora_id, evidence in baseline.items():
            profile = current.get(lora_id)
            if profile is None:
                issues.append(f"production LoRA {lora_id} is missing or no longer validated")
            elif profile.checksum != evidence.get("checksum"):
                issues.append(f"production LoRA {lora_id} checksum changed during qualification")

        primary = self._settings.render_nodes.primary_node()
        for lora_id, profile in current.items():
            if lora_id not in baseline:
                issues.append(f"production LoRA {lora_id} was added during qualification")
                continue
            if not self._settings.qualification.require_stable_asset_hashes:
                continue
            if profile.source is not None and profile.source.startswith("render-node:"):
                node_id = profile.source.removeprefix("render-node:")
                if primary is None or node_id != primary[0]:
                    issues.append(
                        f"production LoRA {lora_id} is not on the primary render node"
                    )
                    continue
                try:
                    attestation = remote_cache.get(node_id)
                    if attestation is None:
                        attestation = fetch_render_attestation(node_id, primary[1], fresh=True)
                        remote_cache[node_id] = attestation
                    if attestation.inventory_errors:
                        raise ValueError("render node inventory is incomplete")
                    matches = [
                        item for item in attestation.loras
                        if item.relative_path.replace("\\\\", "/")
                        == str(profile.metadata.get("asset_name", "")).replace("\\\\", "/")
                    ]
                    if len(matches) != 1 or matches[0].sha256 != profile.checksum:
                        raise ValueError("render node LoRA is absent or hash mismatched")
                except Exception as exc:  # noqa: BLE001
                    issues.append(f"production LoRA {lora_id} cannot be attested: {exc}")
            else:
                try:
                    checksum = _digest_path(lora_id, profile.path).sha256
                except (OSError, ValueError) as exc:
                    issues.append(f"production LoRA {lora_id} cannot be hashed: {exc}")
                    continue
                if checksum != profile.checksum:
                    issues.append(f"production LoRA {lora_id} on-disk hash changed")
        return issues

    def _llm_model_path(self) -> Path | None:
        bootstrap = self._settings.llm.bootstrap
        if not bootstrap.enabled:
            return None
        try:
            path = bootstrap.model_path().expanduser()
        except ValueError:
            return None
        return path if path.is_file() else None

    def _render_attestations(
        self,
    ) -> tuple[dict[str, RenderNodeAttestation], dict[str, str]]:
        primary = self._settings.render_nodes.primary_node()
        if primary is None:
            return {}, {}

        primary_id, primary_config = primary
        errors: dict[str, str] = {}
        if not primary_config.attestation_url:
            errors[f"render_node:{primary_id}"] = (
                "primary render node has no attestation_url configured"
            )
            return {}, errors
        try:
            attestation = fetch_render_attestation(primary_id, primary_config, fresh=True)
        except Exception as exc:  # noqa: BLE001
            errors[f"render_node:{primary_id}"] = str(exc)
            return {}, errors

        for index, item in enumerate(attestation.inventory_errors):
            errors[f"render_node:{primary_id}:inventory:{index}"] = (
                f"{item.get('path', '-')}: {item.get('error', 'unknown inventory error')}"
            )
        return {primary_id: attestation}, errors

    @staticmethod
    def _attestation_environment(
        attestation: RenderNodeAttestation,
    ) -> dict[str, object]:
        return {
            "hostname": attestation.hostname,
            "os": dict(attestation.os),
            "nvidia_gpus": [dict(item) for item in attestation.nvidia_gpus],
            "comfyui_base_url": attestation.comfyui_base_url,
            "attested_at": attestation.created_at.isoformat(),
            "asset_labels": sorted(asset.label for asset in attestation.assets),
            "lora_count": len(attestation.loras),
        }

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
                "bootstrap": {
                    "enabled": settings.llm.bootstrap.enabled,
                    "auto_download": settings.llm.bootstrap.auto_download,
                    "profile": settings.llm.bootstrap.profile,
                    "model_path": str(settings.llm.bootstrap.model_path())
                    if settings.llm.bootstrap.profiles
                    else None,
                },
            },
            "render_nodes": {
                "primary": settings.render_nodes.primary,
                "nodes": {
                    node_id: {
                        "type": node.type,
                        "enabled": node.enabled,
                        "base_url": node.base_url,
                        "output_mode": node.output_mode,
                        "download_dir": str(node.download_dir),
                        "attestation_url": node.attestation_url,
                    }
                    for node_id, node in sorted(settings.render_nodes.nodes.items())
                },
            },
            "comfyui": {
                "base_url": settings.comfyui.base_url,
                "output_mode": settings.comfyui.output_mode,
                "output_dir": (
                    None
                    if settings.comfyui.output_dir is None
                    else str(settings.comfyui.output_dir)
                ),
                "download_dir": str(settings.comfyui.download_dir),
                "render_node_id": settings.comfyui.render_node_id,
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
            "qualification": {
                "required_asset_labels": list(settings.qualification.required_asset_labels),
                "require_native_windows": settings.qualification.require_native_windows,
                "require_uv": settings.qualification.require_uv,
                "require_nvidia_gpu": settings.qualification.require_nvidia_gpu,
                "require_lora_validation_evidence": (
                    settings.qualification.require_lora_validation_evidence
                ),
                "require_stable_asset_hashes": (
                    settings.qualification.require_stable_asset_hashes
                ),
                "require_stable_workflow_snapshot": (
                    settings.qualification.require_stable_workflow_snapshot
                ),
                "minimum_soak_hours": settings.qualification.minimum_soak_hours,
                "asset_paths": {
                    key: str(value)
                    for key, value in sorted(settings.qualification.asset_paths.items())
                },
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
                "asset_name": profile.metadata.get("asset_name"),
                "remote_node_id": profile.metadata.get("remote_node_id"),
                "source": profile.source,
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
