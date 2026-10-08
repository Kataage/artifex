from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from artifex.config.models import ArtifexSettings
from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow, SceneRow
from artifex.loras import LoRAPlan
from artifex.production import GenerationBackend, GenerationRequest
from artifex.prompts import CompiledPrompt
from artifex.telemetry import EventSeverity, TelemetryRepository


class ArchiveReproductionProof(BaseModel):
    """Independently inspectable report of an actual second ComfyUI submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    pack_id: str
    scene_id: str
    selected_attempt_id: str
    original_prompt_id: str
    replay_prompt_id: str
    manifest_sha256: str
    workflow_template: str
    seed: int = Field(ge=0)
    compiled_prompt_sha256: str
    lora_plan_sha256: str
    original_path: str
    original_sha256: str
    reproduced_path: str
    reproduced_sha256: str
    reproduced_at: datetime
    exact_match: bool


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(data: object) -> str:
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_regular_image(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"reproduction requires a nonempty regular image: {path}")
    # Qualification compares the encoded output bytes, not just a thumbnail.
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"reproduction output is not a valid image: {path}") from exc


async def reproduce_archived_attempt(
    settings: ArtifexSettings,
    database: Database,
    backend: GenerationBackend,
    *,
    session_id: str,
    pack_id: str,
    output_root: Path,
) -> tuple[Path, ArchiveReproductionProof]:
    """Re-submit a persisted selected attempt to real ComfyUI, never the daemon.

    An output SHA mismatch is persisted for diagnosis but never qualifies PASS.
    The first selected archived output is the comparison target; one new image
    is generated and copied to a collision-resistant evidence folder.
    """
    if not session_id or not pack_id:
        raise ValueError("session and Pack IDs are required")
    with database.session() as db_session:
        pack = db_session.get(PackRow, pack_id)
        if pack is None or pack.state != "finalized":
            raise ValueError("archive replay needs an existing finalized Pack")
        archive = pack.payload_json.get("archive")
        if not isinstance(archive, dict):
            raise ValueError("Pack has no archived manifest")
        manifest_path = Path(str(archive.get("manifest_path", ""))).expanduser()
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("archive manifest missing or symlinked")
        manifest_sha = file_sha256(manifest_path)
        if (
            manifest_sha != archive.get("manifest_sha256")
            or manifest_sha != pack.checkpoint_json.get("archive_manifest_sha256")
        ):
            raise ValueError("archive manifest does not match SQLite provenance")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_rows = manifest.get("archived_outputs")
        if not isinstance(output_rows, list):
            raise ValueError("archive manifest has no archived outputs")
        selected = [
            row for row in output_rows
            if isinstance(row, dict) and row.get("selected") is True
            and isinstance(row.get("archived_path"), str)
        ]
        if not selected:
            raise ValueError("archive contains no selected archived image")
        selected_row = selected[0]
        scene_id = selected_row.get("scene_id")
        attempt_id = selected_row.get("attempt_id")
        if not isinstance(scene_id, str) or not isinstance(attempt_id, str):
            raise ValueError("archive selected image has no scene and attempt IDs")
        scene = db_session.get(SceneRow, scene_id)
        attempt = db_session.get(GenerationAttemptRow, attempt_id)
        if (
            scene is None or scene.pack_id != pack_id
            or scene.selected_attempt_id != attempt_id
            or attempt is None or attempt.scene_id != scene_id
            or attempt.backend_status != "completed"
            or attempt.seed is None or attempt.seed < 0
        ):
            raise ValueError("archived selected attempt differs from live SQLite selection")
        provenance = attempt.provenance_json
        template_name = provenance.get("workflow_template")
        old_prompt_id = provenance.get("comfy_prompt_id")
        compiled_raw = provenance.get("compiled_prompt")
        lora_raw = provenance.get("lora_plan")
        if (
            not isinstance(template_name, str) or not template_name
            or not isinstance(old_prompt_id, str) or not old_prompt_id
            or not isinstance(compiled_raw, dict)
            or not isinstance(lora_raw, dict)
        ):
            raise ValueError("selected attempt lacks full archived ComfyUI inputs")
        compiled = CompiledPrompt.model_validate(compiled_raw)
        plan = LoRAPlan.model_validate(lora_raw)
        if (
            compiled.positive_prompt != attempt.prompt
            or compiled.negative_prompt != attempt.negative_prompt
        ):
            raise ValueError("archived compiled prompt diverged from selected attempt")
        recorded = dict(backend.provenance())
        expected_keys = (
            "backend", "base_url", "render_node_id", "checkpoint",
            "width", "height", "batch_size", "model_family",
            "refiner_checkpoint", "upscale_model", "base_steps",
            "base_cfg", "base_sampler", "base_scheduler",
        )
        for key in expected_keys:
            if recorded.get(key) != provenance.get(key):
                raise ValueError(f"archived production provenance drift: {key}")
        original = Path(selected_row["archived_path"]).expanduser().resolve()
        if original == manifest_path or not original.is_relative_to(manifest_path.parent.parent):
            raise ValueError("selected archive output is outside archived Pack")
        _require_regular_image(original)
        original_sha = file_sha256(original)
        seed = attempt.seed

    token = uuid4().hex
    run_dir = (output_root / session_id / "archive-reproduction" / token).resolve()
    if not run_dir.is_relative_to(output_root.resolve()):
        raise ValueError("reproduction evidence path escapes configured root")
    run_dir.mkdir(parents=True, exist_ok=False)
    request = GenerationRequest(
        attempt_id=f"qualify-{token}",
        scene_id=f"qualify-{token}",
        compiled=compiled,
        lora_plan=plan,
        seed=seed,
        output_prefix=f"ARTIFEX/qualification/{token}",
        workflow_template_id=template_name,
    )
    submitted: list[str] = []
    result = await backend.generate(request, on_submitted=submitted.append)
    if (
        not result.prompt_id or submitted != [result.prompt_id]
        or result.prompt_id == old_prompt_id
        or len(result.output_paths) != 1
    ):
        raise ValueError("replay did not produce exactly one new ComfyUI output")
    produced = result.output_paths[0].expanduser().resolve()
    _require_regular_image(produced)
    if produced == original:
        raise ValueError("replay output aliases original archived image")
    copied = run_dir / ("reproduced" + produced.suffix.lower())
    if copied.exists() or copied.is_symlink():
        raise FileExistsError("replay proof output already exists")
    with produced.open("rb") as source, copied.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    _require_regular_image(copied)
    replay_sha = file_sha256(copied)
    proof = ArchiveReproductionProof(
        session_id=session_id,
        pack_id=pack_id,
        scene_id=scene_id,
        selected_attempt_id=attempt_id,
        original_prompt_id=old_prompt_id,
        replay_prompt_id=result.prompt_id,
        manifest_sha256=manifest_sha,
        workflow_template=template_name,
        seed=seed,
        compiled_prompt_sha256=_json_sha256(compiled_raw),
        lora_plan_sha256=_json_sha256(lora_raw),
        original_path=str(original),
        original_sha256=original_sha,
        reproduced_path=str(copied),
        reproduced_sha256=replay_sha,
        reproduced_at=datetime.now(UTC),
        exact_match=original_sha == replay_sha,
    )
    proof_path = run_dir / "proof.json"
    with proof_path.open("x", encoding="utf-8") as stream:
        stream.write(proof.model_dump_json(indent=2) + "\n")
        stream.flush()
    # An event is emitted by this execution path, not by manual stage input.
    TelemetryRepository(database).record(
        "qualification.archive_replayed", EventSeverity.INFO,
        {
            "session_id": session_id, "pack_id": pack_id,
            "proof_path": str(proof_path),
            "proof_sha256": file_sha256(proof_path),
            "replay_prompt_id": result.prompt_id,
        },
    )
    return proof_path, proof
