from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from artifex.characters import CharacterRegistry
from artifex.config.models import EvaluationConfig
from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow, SceneRow
from artifex.domain import CharacterProfile, PackState, PublicationTier, SceneState
from artifex.evaluation import (
    EmbeddingModelDescriptor,
    EvaluationContext,
    EvaluationEngine,
    EvaluationReferenceResolver,
    RawEvaluationSignals,
    SemanticEmbeddingRepository,
    SemanticIndex,
    SimilarityService,
)
from artifex.packs import ScenePlan, VisualSpecification


class CountingEmbeddings:
    def __init__(self) -> None:
        self.text_calls = 0
        self.image_calls = 0

    @property
    def descriptor(self) -> EmbeddingModelDescriptor:
        return EmbeddingModelDescriptor(
            provider="test",
            model="semantic-test",
            revision="r1",
            quality_tier="production",
        )

    async def embed_text(self, text: str) -> tuple[float, ...]:
        self.text_calls += 1
        return (1.0, float(len(text) % 7 + 1))

    async def embed_image(self, path: Path) -> tuple[float, ...]:
        self.image_calls += 1
        return (1.0, float(path.stat().st_size % 11 + 1))


class MappingEmbeddings:
    def __init__(self, images: dict[str, tuple[float, ...]]) -> None:
        self.images = images

    async def embed_text(self, text: str) -> tuple[float, ...]:
        del text
        return (1.0, 0.0)

    async def embed_image(self, path: Path) -> tuple[float, ...]:
        return self.images[str(path)]


class StrongVision:
    async def evaluate(self, context: EvaluationContext) -> RawEvaluationSignals:
        del context
        return RawEvaluationSignals(
            identity=0.98,
            alignment=0.98,
            face_quality=0.98,
            technical_quality=0.98,
            aesthetic=0.98,
            continuity=0.98,
            integrity=0.99,
        )


def _scene_plan(character_id: str) -> ScenePlan:
    return ScenePlan(
        ordinal=1,
        title="feature",
        purpose="semantic regression",
        character_ids=(character_id,),
        visual=VisualSpecification(
            composition="full body centered",
            camera="eye level portrait",
            pose="standing",
            expression="smile",
            clothing="canonical outfit",
            setting="studio",
            lighting="soft light",
            atmosphere="clean",
        ),
        publication_tier=PublicationTier.PUBLIC,
    )


@pytest.mark.asyncio
async def test_semantic_index_persists_and_invalidates_by_content_hash(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'semantic.sqlite3').as_posix()}")
    database.migrate()
    provider = CountingEmbeddings()
    repository = SemanticEmbeddingRepository(database)
    index = SemanticIndex(repository, provider)

    first = await index.embed_text("concept", "concept-1", "angel at an open window")
    second = await index.embed_text("concept", "concept-1", "angel at an open window")

    assert first == second
    assert provider.text_calls == 1
    assert repository.count() == 1

    changed = await index.embed_text(
        "concept",
        "concept-1",
        "winged girl sitting halfway outside an open window",
    )
    assert changed != ()
    assert provider.text_calls == 2
    assert repository.count() == 1
    database.dispose()


def test_reference_resolver_scans_beyond_twenty_and_separates_characters(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'references.sqlite3').as_posix()}")
    database.migrate()
    characters = CharacterRegistry(database)

    ref_dir = tmp_path / "refs-a"
    ref_dir.mkdir()
    Image.new("RGB", (8, 8), (200, 180, 160)).save(ref_dir / "canonical.png")
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="Character A",
            namespace="test",
            readiness=1.0,
            reference_image_dirs=(ref_dir,),
        )
    )
    characters.upsert(
        CharacterProfile(
            id="char-b",
            display_name="Character B",
            namespace="test",
            readiness=1.0,
        )
    )

    now = datetime(2026, 10, 6, 1, tzinfo=UTC)
    with database.session() as session:
        for index in range(30):
            pack_id = f"pack-a-{index:02d}"
            scene_id = f"scene-a-{index:02d}"
            attempt_id = f"attempt-a-{index:02d}"
            path = tmp_path / f"hist-a-{index:02d}.png"
            Image.new("RGB", (8, 8), (120, index, 180)).save(path)
            plan = _scene_plan("char-a")
            session.add(
                PackRow(
                    id=pack_id,
                    concept_id=None,
                    series_id=None,
                    state=PackState.FINALIZED.value,
                    format_type="single_feature",
                    payload_json={},
                    checkpoint_json={},
                    created_at=now - timedelta(days=index + 1),
                    updated_at=now - timedelta(days=index + 1),
                )
            )
            session.add(
                SceneRow(
                    id=scene_id,
                    pack_id=pack_id,
                    ordinal=1,
                    state=SceneState.ACCEPTED.value,
                    publication_tier=PublicationTier.PUBLIC.value,
                    payload_json={"plan": plan.model_dump(mode="json")},
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
                    seed=index,
                    prompt="char-a",
                    negative_prompt="",
                    provenance_json={"output_paths": [str(path)]},
                    error_json=None,
                    created_at=now - timedelta(days=index + 1),
                )
            )

        for index in range(3):
            pack_id = f"pack-b-{index:02d}"
            scene_id = f"scene-b-{index:02d}"
            attempt_id = f"attempt-b-{index:02d}"
            path = tmp_path / f"hist-b-{index:02d}.png"
            Image.new("RGB", (8, 8), (120, index, 180)).save(path)
            plan = _scene_plan("char-b")
            session.add(
                PackRow(
                    id=pack_id,
                    concept_id=None,
                    series_id=None,
                    state=PackState.FINALIZED.value,
                    format_type="single_feature",
                    payload_json={},
                    checkpoint_json={},
                    created_at=now - timedelta(hours=index + 1),
                    updated_at=now - timedelta(hours=index + 1),
                )
            )
            session.add(
                SceneRow(
                    id=scene_id,
                    pack_id=pack_id,
                    ordinal=1,
                    state=SceneState.ACCEPTED.value,
                    publication_tier=PublicationTier.PUBLIC.value,
                    payload_json={"plan": plan.model_dump(mode="json")},
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
                    seed=100 + index,
                    prompt="char-b",
                    negative_prompt="",
                    provenance_json={"output_paths": [str(path)]},
                    error_json=None,
                    created_at=now - timedelta(hours=index + 1),
                )
            )

    resolver = EvaluationReferenceResolver(
        database,
        characters,
        EvaluationConfig(
            identity_reference_required=True,
            duplicate_reference_limit=64,
            novelty_reference_limit=64,
            global_diversity_reference_limit=8,
            historical_reference_scan_limit=100,
        ),
    )
    refs = resolver.resolve("current-scene", _scene_plan("char-a"))

    assert refs.historical_candidates_scanned == 33
    assert len(refs.duplicate_paths) == 30
    assert all("hist-a-" in path.name for path in refs.duplicate_paths)
    assert any("hist-b-" in path.name for path in refs.novelty_paths)
    assert refs.identity_by_character["char-a"] == (ref_dir / "canonical.png",)
    database.dispose()


@pytest.mark.asyncio
async def test_reference_grounded_identity_hard_gate_is_separate_from_duplicate_gate(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current.png"
    correct_ref = tmp_path / "char-a.png"
    wrong_character_same_palette = tmp_path / "char-b.png"
    for path in (current, correct_ref, wrong_character_same_palette):
        Image.new("RGB", (8, 8), (120, 160, 200)).save(path)

    embeddings = MappingEmbeddings(
        {
            str(current): (1.0, 0.0),
            str(correct_ref): (0.0, 1.0),
            str(wrong_character_same_palette): (1.0, 0.0),
        }
    )
    engine = EvaluationEngine(
        StrongVision(),
        SimilarityService(embeddings),
        EvaluationConfig(
            identity_reference_required=True,
            identity_reference_hard_min=0.50,
            identity_reference_accept_min=0.60,
            similarity_hard_max=0.95,
        ),
    )
    result = await engine.evaluate(
        EvaluationContext(
            attempt_id="attempt-1",
            scene_id="scene-1",
            image_path=current,
            character_ids=("char-a",),
            positive_prompt="char-a, standing",
            negative_prompt="",
            identity_reference_image_paths={"char-a": (correct_ref,)},
            novelty_reference_image_paths=(wrong_character_same_palette,),
        )
    )

    assert result.scores.identity_reference == pytest.approx(0.0)
    assert result.scores.image_similarity == pytest.approx(0.0)
    assert result.scores.novelty_similarity == pytest.approx(1.0)
    assert "identity_reference_hard_failure" in result.reasons
