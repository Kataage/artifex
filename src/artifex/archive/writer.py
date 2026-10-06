from __future__ import annotations

import hashlib
import json
import shutil
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
        self._patreon = patreon or PatreonConfig()

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
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        for record in output_records:
            source = Path(record["source_path"]).expanduser()
            if not source.exists() or not source.is_file():
                continue
            tier = str(record["archive_tier"])
            destination_dir = directories.get(tier, directories["review"])
            destination = self._unique_destination(
                destination_dir,
                f"{record['scene_id']}_{source.name}",
            )
            shutil.copy2(source, destination)
            copied.append(destination)
            record["archived_path"] = str(destination)

        manifest["archived_outputs"] = output_records
        post_package_path: Path | None = None
        if self._patreon.enabled and self._patreon.generate_post_package:
            post_package = self._build_post_package(manifest, output_records)
            post_package_path = directories["metadata"] / "patreon-post-package.json"
            post_package_path.write_text(
                json.dumps(
                    post_package,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            manifest["post_package"] = {
                "platform": "patreon",
                "path": str(post_package_path),
                "schema_version": post_package["schema_version"],
            }

        manifest_path = directories["metadata"] / "pack.json"
        encoded = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        manifest_path.write_bytes(encoded)
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

    @staticmethod
    def _unique_destination(directory: Path, filename: str) -> Path:
        candidate = directory / filename
        if not candidate.exists():
            return candidate
        stem = candidate.stem
        suffix = candidate.suffix
        index = 2
        while True:
            alternative = directory / f"{stem}_{index}{suffix}"
            if not alternative.exists():
                return alternative
            index += 1
