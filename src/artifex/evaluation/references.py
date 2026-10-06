from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from artifex.characters import CharacterRegistry
from artifex.config.models import EvaluationConfig
from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow, SceneRow
from artifex.domain import PackState
from artifex.packs import ScenePlan

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_WORD = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True, slots=True)
class EvaluationReferenceSets:
    identity_by_character: dict[str, tuple[Path, ...]]
    duplicate_paths: tuple[Path, ...]
    novelty_paths: tuple[Path, ...]
    historical_candidates_scanned: int


def _dedupe(paths: list[Path], *, limit: int) -> tuple[Path, ...]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        result.append(path)
        if len(result) >= limit:
            break
    return tuple(result)


def _visual_family(plan: ScenePlan) -> str:
    composition = "_".join(_WORD.findall(plan.visual.composition.casefold())[:4])
    camera = "_".join(_WORD.findall(plan.visual.camera.casefold())[:4])
    return f"{plan.role.value}|{composition}|{camera}"


def _output_paths(attempt: GenerationAttemptRow) -> tuple[Path, ...]:
    raw = attempt.provenance_json.get("output_paths", ())
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(Path(value) for value in raw if isinstance(value, str) and value)


class EvaluationReferenceResolver:
    """Build purpose-specific reference sets without a last-20 history cutoff."""

    def __init__(
        self,
        database: Database,
        characters: CharacterRegistry,
        config: EvaluationConfig,
    ) -> None:
        self._database = database
        self._characters = characters
        self._config = config

    def resolve(self, scene_id: str, plan: ScenePlan) -> EvaluationReferenceSets:
        identity = {
            character_id: self._character_reference_paths(character_id)
            for character_id in plan.character_ids
        }
        current_signature = tuple(sorted(plan.character_ids))
        current_family = _visual_family(plan)

        with self._database.session() as session:
            rows = session.execute(
                select(SceneRow, PackRow, GenerationAttemptRow)
                .join(PackRow, SceneRow.pack_id == PackRow.id)
                .join(
                    GenerationAttemptRow,
                    SceneRow.selected_attempt_id == GenerationAttemptRow.id,
                )
                .where(
                    SceneRow.id != scene_id,
                    SceneRow.selected_attempt_id.is_not(None),
                    PackRow.state == PackState.FINALIZED.value,
                )
                .order_by(PackRow.updated_at.desc(), SceneRow.ordinal.desc())
                .limit(self._config.historical_reference_scan_limit)
            ).all()

        same_family: list[Path] = []
        same_characters: list[Path] = []
        global_diversity: list[Path] = []
        global_keys: set[tuple[tuple[str, ...], str]] = set()

        for scene, _, attempt in rows:
            raw_plan = scene.payload_json.get("plan")
            if not isinstance(raw_plan, dict):
                continue
            try:
                historical_plan = ScenePlan.model_validate(raw_plan)
            except ValueError:
                continue
            paths = [path for path in _output_paths(attempt) if path.exists()]
            if not paths:
                continue
            signature = tuple(sorted(historical_plan.character_ids))
            family = _visual_family(historical_plan)
            if signature == current_signature:
                same_characters.extend(paths)
                if family == current_family:
                    same_family.extend(paths)
                continue
            diversity_key = (signature, family)
            if (
                diversity_key not in global_keys
                and len(global_diversity)
                < self._config.global_diversity_reference_limit
            ):
                global_keys.add(diversity_key)
                global_diversity.append(paths[0])

        duplicate = _dedupe(
            [*same_family, *same_characters],
            limit=self._config.duplicate_reference_limit,
        )
        novelty = _dedupe(
            [*same_family, *same_characters, *global_diversity],
            limit=self._config.novelty_reference_limit,
        )
        return EvaluationReferenceSets(
            identity_by_character=identity,
            duplicate_paths=duplicate,
            novelty_paths=novelty,
            historical_candidates_scanned=len(rows),
        )

    def _character_reference_paths(self, character_id: str) -> tuple[Path, ...]:
        profile = self._characters.require(character_id)
        paths: list[Path] = []
        for root in profile.reference_image_dirs:
            if root.is_file() and root.suffix.casefold() in _IMAGE_SUFFIXES:
                paths.append(root)
                continue
            if not root.exists() or not root.is_dir():
                continue
            paths.extend(
                sorted(
                    (
                        path
                        for path in root.rglob("*")
                        if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
                    ),
                    key=lambda path: path.as_posix().casefold(),
                )
            )
        return _dedupe(
            paths,
            limit=self._config.identity_reference_limit_per_character,
        )
