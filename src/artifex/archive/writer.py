from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from artifex.config.models import PatreonConfig
from artifex.db import Database
from artifex.db.models import (
    ConceptRow,
    EvaluationRow,
    GenerationAttemptRow,
    PackRow,
    PolicyDecisionRow,
    SceneRow,
)
from artifex.packs import ContentPackPlan


class ArchiveResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pack_id: str
    root: Path
    manifest_path: Path
    manifest_sha256: str
    copied_outputs: tuple[Path, ...]
    post_package_path: Path | None = None


class PackArchive:
    def __init__(
        self,
        database: Database,
        root: Path,
        patreon: PatreonConfig | None = None,
    ) -> None:
        self._database = database
        self._root = root
        self._patreon = patreon or PatreonConfig(enabled=False)


    def finalize(self, pack_id: str) -> ArchiveResult:
        manifest, output_records, created_at = self._build_manifest(pack_id)
        pack_root = (
            self._root.expanduser().resolve()
            / f"{created_at.year:04d}"
            / f"{created_at.month:02d}"
            / f"PACK-{pack_id}"
        )
        directories = {
            name: pack_root / name
            for name in ("public", "member", "review", "rejected", "metadata", "source")
        }
        manifest_path = directories["metadata"] / "pack.json"

        # The archive checkpoint is durable BEFORE the coordinator's Pack
        # FINALIZED transition. A restart in between must reuse the exact
        # original manifest instead of hashing a mutated SQLite snapshot.
        with self._database.session() as session:
            pack = session.get(PackRow, pack_id)
            if pack is None:
                raise KeyError(f"unknown pack: {pack_id}")
            if pack.checkpoint_json.get("archive_complete") is True:
                return self._restore_completed(pack, pack_root, manifest_path)

        for directory in directories.values():
            if directory.is_symlink():
                raise ValueError(f"archive directory is a symlink: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        for index, record in enumerate(output_records):
            source = Path(record["source_path"]).expanduser()
            tier = str(record["archive_tier"])
            destination_dir = directories.get(tier, directories["review"])
            # Stable across retries; separate different attempts and batch
            # outputs even if they share the same basename.
            destination = destination_dir / (
                f"{record['scene_id']}_{record['attempt_id']}_{index:04d}_"
                f"{source.name}"
            )
            file_digest = self._copy_or_verify(source, destination)
            copied.append(destination)
            record["archived_path"] = str(destination)
            record["sha256"] = file_digest

        manifest["archived_outputs"] = output_records
        post_package_path: Path | None = None
        if self._patreon.enabled and self._patreon.generate_post_package:
            post_package = self._build_post_package(manifest, output_records)
            post_package_path = directories["metadata"] / "patreon-post-package.json"
            self._write_or_verify(
                post_package_path,
                json.dumps(
                    post_package,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8"),
            )
            manifest["post_package"] = {
                "platform": "patreon",
                "path": str(post_package_path),
                "schema_version": post_package["schema_version"],
            }

        encoded = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        self._write_or_verify(manifest_path, encoded)
        digest = hashlib.sha256(encoded).hexdigest()

        with self._database.session() as session:
            pack = session.get(PackRow, pack_id)
            if pack is None:
                raise KeyError(f"unknown pack: {pack_id}")
            pack.payload_json = {
                **pack.payload_json,
                "archive": {
                    "root": str(pack_root),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": digest,
                    "copied_output_count": len(copied),
                    "post_package_path": (
                        str(post_package_path)
                        if post_package_path is not None
                        else None
                    ),
                },
            }
            pack.checkpoint_json = {
                **pack.checkpoint_json,
                "archive_complete": True,
                "archive_manifest_sha256": digest,
            }

        return ArchiveResult(
            pack_id=pack_id,
            root=pack_root,
            manifest_path=manifest_path,
            manifest_sha256=digest,
            copied_outputs=tuple(copied),
            post_package_path=post_package_path,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _copy_or_verify(cls, source: Path, destination: Path) -> str:
        if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"archive source missing, unsafe or empty: {source}")
        source_digest = cls._sha256(source)
        if destination.is_symlink():
            raise ValueError(f"archive destination is a symlink: {destination}")
        if destination.exists():
            if (
                not destination.is_file()
                or destination.stat().st_size == 0
                or cls._sha256(destination) != source_digest
            ):
                raise ValueError(
                    f"existing archived output differs from original: {destination}"
                )
            return source_digest

        # Publish only a complete file. A crash leaves at most an orphaned
        # temporary copy, never a partial file at the canonical path.
        fd, temp_name = tempfile.mkstemp(prefix=".artifex-", dir=destination.parent)
        os.close(fd)
        temporary = Path(temp_name)
        try:
            shutil.copy2(source, temporary)
            if cls._sha256(temporary) != source_digest:
                raise OSError(f"source changed while archiving: {source}")
            if destination.is_symlink() or destination.exists():
                raise FileExistsError(
                    f"archive destination appeared during copy: {destination}"
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return source_digest

    @staticmethod
    def _write_or_verify(path: Path, contents: bytes) -> None:
        if path.is_symlink():
            raise ValueError(f"archive metadata is a symlink: {path}")
        if path.exists():
            if not path.is_file() or path.read_bytes() != contents:
                raise ValueError(f"archive metadata already differs: {path}")
            return
        fd, temp_name = tempfile.mkstemp(prefix=".artifex-", dir=path.parent)
        temporary = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            if path.is_symlink() or path.exists():
                raise FileExistsError(f"archive metadata appeared during write: {path}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _restore_completed(
        cls, pack: PackRow, pack_root: Path, manifest_path: Path
    ) -> ArchiveResult:
        archive = pack.payload_json.get("archive")
        if not isinstance(archive, dict):
            raise TypeError("completed archive checkpoint lacks provenance")
        digest = archive.get("manifest_sha256")
        if (
            not isinstance(digest, str)
            or digest != pack.checkpoint_json.get("archive_manifest_sha256")
            or archive.get("root") != str(pack_root)
            or archive.get("manifest_path") != str(manifest_path)
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
            or cls._sha256(manifest_path) != digest
        ):
            raise ValueError("saved archive manifest is absent or conflicts with SQLite")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("pack", {}).get("id") != pack.id:
            raise ValueError("completed archive has wrong Pack identity")
        outputs = manifest.get("archived_outputs")
        if not isinstance(outputs, list):
            raise TypeError("completed archive has no output index")
        copied: list[Path] = []
        for record in outputs:
            if not isinstance(record, dict):
                raise TypeError("completed archive output record is malformed")
            raw_path = record.get("archived_path")
            if not isinstance(raw_path, str):
                raise TypeError("completed archive output path is missing")
            archived_path = Path(raw_path)
            if (
                archived_path.is_symlink()
                or not archived_path.is_file()
                or not archived_path.resolve().is_relative_to(pack_root.resolve())
                or archived_path.stat().st_size == 0
            ):
                raise ValueError(f"completed archive output unavailable: {archived_path}")
            recorded_hash = record.get("sha256")
            if recorded_hash is not None and cls._sha256(archived_path) != recorded_hash:
                raise ValueError(f"completed archive output altered: {archived_path}")
            copied.append(archived_path)
        if archive.get("copied_output_count") != len(copied):
            raise ValueError("completed archive output count conflicts with SQLite")
        post_raw = archive.get("post_package_path")
        post_package_path = Path(post_raw) if isinstance(post_raw, str) else None
        if post_package_path is not None and (
            post_package_path.is_symlink()
            or not post_package_path.is_file()
            or post_package_path.resolve()
            != (pack_root / "metadata" / "patreon-post-package.json").resolve()
        ):
            raise ValueError("completed Patreon post package is missing or unsafe")
        return ArchiveResult(
            pack_id=pack.id,
            root=pack_root,
            manifest_path=manifest_path,
            manifest_sha256=digest,
            copied_outputs=tuple(copied),
            post_package_path=post_package_path,
        )

    def _build_manifest(
        self,
        pack_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
        with self._database.session() as session:
            pack = session.get(PackRow, pack_id)
            if pack is None:
                raise KeyError(f"unknown pack: {pack_id}")
            scenes = session.scalars(
                select(SceneRow)
                .where(SceneRow.pack_id == pack_id)
                .order_by(SceneRow.ordinal.asc(), SceneRow.id.asc())
            ).all()
            scene_ids = [scene.id for scene in scenes]
            attempts = session.scalars(
                select(GenerationAttemptRow)
                .where(GenerationAttemptRow.scene_id.in_(scene_ids))
                .order_by(
                    GenerationAttemptRow.scene_id.asc(),
                    GenerationAttemptRow.ordinal.asc(),
                )
            ).all()
            attempt_ids = [attempt.id for attempt in attempts]
            evaluations = session.scalars(
                select(EvaluationRow)
                .where(EvaluationRow.attempt_id.in_(attempt_ids))
                .order_by(EvaluationRow.created_at.asc(), EvaluationRow.id.asc())
            ).all()
            policy = session.scalars(
                select(PolicyDecisionRow)
                .where(
                    PolicyDecisionRow.subject_type == "scene",
                    PolicyDecisionRow.subject_id.in_(scene_ids),
                )
                .order_by(PolicyDecisionRow.created_at.asc(), PolicyDecisionRow.id.asc())
            ).all()
            concept = (
                session.get(ConceptRow, pack.concept_id)
                if pack.concept_id is not None
                else None
            )

            manifest: dict[str, Any] = {
                "schema_version": 1,
                "pack": {
                    "id": pack.id,
                    "concept_id": pack.concept_id,
                    "series_id": pack.series_id,
                    "state": pack.state,
                    "format_type": pack.format_type,
                    "payload": pack.payload_json,
                    "checkpoint": pack.checkpoint_json,
                    "created_at": pack.created_at.isoformat(),
                    "updated_at": pack.updated_at.isoformat(),
                },
                "concept": (
                    {
                        "id": concept.id,
                        "status": concept.status,
                        "payload": concept.payload_json,
                        "score": concept.score,
                        "similarity_score": concept.similarity_score,
                        "created_at": concept.created_at.isoformat(),
                    }
                    if concept is not None
                    else None
                ),
                "scenes": [
                    {
                        "id": scene.id,
                        "ordinal": scene.ordinal,
                        "state": scene.state,
                        "publication_tier": scene.publication_tier,
                        "selected_attempt_id": scene.selected_attempt_id,
                        "payload": scene.payload_json,
                    }
                    for scene in scenes
                ],
                "attempts": [
                    {
                        "id": attempt.id,
                        "scene_id": attempt.scene_id,
                        "parent_attempt_id": attempt.parent_attempt_id,
                        "ordinal": attempt.ordinal,
                        "backend_status": attempt.backend_status,
                        "seed": attempt.seed,
                        "prompt": attempt.prompt,
                        "negative_prompt": attempt.negative_prompt,
                        "provenance": attempt.provenance_json,
                        "error": attempt.error_json,
                        "created_at": attempt.created_at.isoformat(),
                    }
                    for attempt in attempts
                ],
                "evaluations": [
                    {
                        "id": evaluation.id,
                        "attempt_id": evaluation.attempt_id,
                        "result_state": evaluation.result_state,
                        "scores": evaluation.scores_json,
                        "reasons": evaluation.reasons_json,
                        "classification": evaluation.classification_json,
                        "created_at": evaluation.created_at.isoformat(),
                    }
                    for evaluation in evaluations
                ],
                "policy_decisions": [
                    {
                        "id": decision.id,
                        "subject_id": decision.subject_id,
                        "policy_name": decision.policy_name,
                        "policy_version": decision.policy_version,
                        "decision": decision.decision,
                        "reason": decision.reason,
                        "payload": decision.payload_json,
                        "created_at": decision.created_at.isoformat(),
                    }
                    for decision in policy
                ],
            }

            selected_by_scene = {
                scene.id: scene.selected_attempt_id
                for scene in scenes
                if scene.selected_attempt_id is not None
            }
            tiers_by_scene = {
                scene.id: self._archive_tier(scene.publication_tier)
                for scene in scenes
            }
            output_records: list[dict[str, Any]] = []
            for attempt in attempts:
                raw_paths = attempt.provenance_json.get("output_paths", ())
                if not isinstance(raw_paths, list | tuple):
                    continue
                selected = selected_by_scene.get(attempt.scene_id) == attempt.id
                for raw_path in raw_paths:
                    if not isinstance(raw_path, str) or not raw_path:
                        continue
                    output_records.append(
                        {
                            "attempt_id": attempt.id,
                            "scene_id": attempt.scene_id,
                            "selected": selected,
                            "source_path": raw_path,
                            "archive_tier": (
                                tiers_by_scene.get(attempt.scene_id, "review")
                                if selected
                                else "rejected"
                            ),
                        }
                    )
            created_at = pack.created_at

        return manifest, output_records, created_at

    def _build_post_package(
        self,
        manifest: dict[str, Any],
        output_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pack_data = manifest["pack"]
        payload = pack_data["payload"]
        if not isinstance(payload, dict):
            raise TypeError("pack payload must be an object")
        raw_plan = payload.get("plan")
        if not isinstance(raw_plan, dict):
            raise TypeError("pack archive is missing ContentPackPlan")
        plan = ContentPackPlan.model_validate(raw_plan)

        selected_paths: dict[str, list[str]] = {}
        for record in output_records:
            if not record.get("selected"):
                continue
            archived_path = record.get("archived_path")
            scene_id = record.get("scene_id")
            if isinstance(scene_id, str) and isinstance(archived_path, str):
                selected_paths.setdefault(scene_id, []).append(archived_path)

        scene_rows = manifest.get("scenes", [])
        if not isinstance(scene_rows, list):
            raise TypeError("archived scenes must be a list")
        public_scene_ids: list[str] = []
        member_scene_ids: list[str] = []
        review_scene_ids: list[str] = []
        package_scenes: list[dict[str, Any]] = []

        plan_by_ordinal = {scene.ordinal: scene for scene in plan.scenes}
        for scene_row in scene_rows:
            if not isinstance(scene_row, dict):
                continue
            scene_id = str(scene_row["id"])
            ordinal = int(scene_row["ordinal"])
            tier = str(scene_row["publication_tier"])
            scene_payload = scene_row.get("payload")
            if not isinstance(scene_payload, dict):
                scene_payload = {}
            scene_plan = plan_by_ordinal[ordinal]

            if tier == "public":
                public_scene_ids.append(scene_id)
            elif tier == "member":
                member_scene_ids.append(scene_id)
            else:
                review_scene_ids.append(scene_id)

            package_scenes.append(
                {
                    "scene_id": scene_id,
                    "ordinal": ordinal,
                    "title": scene_plan.title,
                    "purpose": scene_plan.purpose,
                    "role": scene_plan.role.value,
                    "publication_tier": tier,
                    "planned_content_rating": (
                        scene_plan.planned_content_rating.value
                    ),
                    "content_rating": scene_payload.get(
                        "content_rating",
                        scene_plan.planned_content_rating.value,
                    ),
                    "content_labels": scene_payload.get("content_labels", []),
                    "files": selected_paths.get(scene_id, []),
                }
            )

        tags = list(self._patreon.default_tags)
        if self._patreon.include_character_tags:
            tags.extend(plan.character_ids)
        tags.extend((plan.format.value, plan.editorial_archetype.value))
        tags = list(dict.fromkeys(tag for tag in tags if tag))

        preview_scene = next(
            (
                item
                for item in package_scenes
                if item["publication_tier"] == "public"
            ),
            package_scenes[0],
        )
        preview_copy = (
            f"{plan.logline} Preview: {preview_scene['title']} — "
            f"{preview_scene['purpose']}"
        )
        if member_scene_ids:
            preview_copy += (
                f" Member continuation: {len(member_scene_ids)} scene(s)."
            )

        return {
            "schema_version": 1,
            "platform": "patreon",
            "pack_id": pack_data["id"],
            "concept_id": pack_data.get("concept_id"),
            "series_id": pack_data.get("series_id"),
            "title": plan.title,
            "caption": plan.logline,
            "preview_copy": preview_copy,
            "tags": tags,
            "format": plan.format.value,
            "editorial_archetype": plan.editorial_archetype.value,
            "public_scene_ids": public_scene_ids,
            "member_scene_ids": member_scene_ids,
            "review_scene_ids": review_scene_ids,
            "scenes": package_scenes,
            "publication_ready": not review_scene_ids,
        }

    @staticmethod
    def _archive_tier(publication_tier: str) -> str:
        return {
            "public": "public",
            "member": "member",
            "private_review": "review",
            "blocked": "review",
        }.get(publication_tier, "review")
