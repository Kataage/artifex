from __future__ import annotations

from pathlib import Path

import pytest

from artifex.config.models import EvaluationConfig
from artifex.domain import ResultState
from artifex.evaluation import (
    EvaluationContext,
    EvaluationEngine,
    RawEvaluationSignals,
    SimilarityService,
)


class FakeEvaluator:
    def __init__(self, signals: RawEvaluationSignals) -> None:
        self.signals = signals

    async def evaluate(self, context: EvaluationContext) -> RawEvaluationSignals:
        del context
        return self.signals


class FakeEmbeddings:
    def __init__(
        self,
        *,
        images: dict[str, tuple[float, ...]] | None = None,
        texts: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        self.images = images or {}
        self.texts = texts or {}

    async def embed_text(self, text: str) -> tuple[float, ...]:
        return self.texts[text]

    async def embed_image(self, path: Path) -> tuple[float, ...]:
        return self.images[str(path)]


def _context(tmp_path: Path, refs: tuple[Path, ...] = ()) -> EvaluationContext:
    return EvaluationContext(
        attempt_id="attempt-1",
        scene_id="scene-1",
        image_path=tmp_path / "current.png",
        character_ids=("char-a",),
        positive_prompt="char-a, smile",
        negative_prompt="bad_hands",
        reference_image_paths=refs,
    )


def _strong(**updates: float) -> RawEvaluationSignals:
    values = {
        "identity": 0.95,
        "alignment": 0.95,
        "face_quality": 0.95,
        "technical_quality": 0.95,
        "aesthetic": 0.95,
        "continuity": 0.95,
        "integrity": 1.0,
    }
    values.update(updates)
    return RawEvaluationSignals(**values)


@pytest.mark.asyncio
async def test_identity_hard_gate_overrides_strong_global_score(tmp_path: Path) -> None:
    engine = EvaluationEngine(
        FakeEvaluator(_strong(identity=0.2)),
        SimilarityService(
            FakeEmbeddings(images={str(tmp_path / "current.png"): (1.0, 0.0)})
        ),
        EvaluationConfig(identity_reference_required=False),
    )

    result = await engine.evaluate(_context(tmp_path))

    assert result.scores.aggregate > 0.7
    assert result.state is ResultState.REJECTED
    assert "identity_hard_failure" in result.reasons


@pytest.mark.asyncio
async def test_image_similarity_hard_gate_rejects_duplicate(tmp_path: Path) -> None:
    current = tmp_path / "current.png"
    old = tmp_path / "old.png"
    engine = EvaluationEngine(
        FakeEvaluator(_strong()),
        SimilarityService(
            FakeEmbeddings(
                images={
                    str(current): (1.0, 0.0),
                    str(old): (1.0, 0.0),
                }
            )
        ),
        EvaluationConfig(identity_reference_required=False),
    )

    result = await engine.evaluate(_context(tmp_path, (old,)))

    assert result.scores.image_similarity == pytest.approx(1.0)
    assert result.scores.novelty == pytest.approx(0.0)
    assert result.state is ResultState.REJECTED
    assert "image_similarity_hard_failure" in result.reasons


@pytest.mark.asyncio
async def test_strong_result_is_accepted(tmp_path: Path) -> None:
    engine = EvaluationEngine(
        FakeEvaluator(_strong()),
        SimilarityService(
            FakeEmbeddings(images={str(tmp_path / "current.png"): (1.0, 0.0)})
        ),
        EvaluationConfig(identity_reference_required=False),
    )

    result = await engine.evaluate(_context(tmp_path))

    assert result.state is ResultState.ACCEPTED
    assert result.scores.aggregate >= 0.75


@pytest.mark.asyncio
async def test_low_face_quality_routes_to_review(tmp_path: Path) -> None:
    engine = EvaluationEngine(
        FakeEvaluator(_strong(face_quality=0.4)),
        SimilarityService(
            FakeEmbeddings(images={str(tmp_path / "current.png"): (1.0, 0.0)})
        ),
        EvaluationConfig(identity_reference_required=False),
    )

    result = await engine.evaluate(_context(tmp_path))

    assert result.state is ResultState.REVIEW
    assert "face_quality_needs_review" in result.reasons
