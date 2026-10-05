from __future__ import annotations

from pathlib import Path

import pytest

from artifex.config.models import LoRARegistryConfig
from artifex.db import Database
from artifex.domain import LoRAProfile, LoRAState
from artifex.loras import LoRARegistry, LoRAValidationReport, LoRAValidationService


def test_validation_lifecycle_and_thresholds(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'validation.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    registry.upsert(
        LoRAProfile(
            id="lora-a",
            path=tmp_path / "a.safetensors",
            target_character_ids=("char-a",),
            model_families=("ilxl",),
        )
    )
    service = LoRAValidationService(registry, LoRARegistryConfig())

    validated = service.record(
        "lora-a",
        LoRAValidationReport(
            identity_score=0.9,
            quality_score=0.8,
            flexibility_score=0.7,
            recommended_weight=0.85,
            validated_min_weight=0.7,
            validated_max_weight=1.0,
        ),
    )
    assert validated.state is LoRAState.VALIDATED
    assert validated.readiness == pytest.approx(0.83)
    assert validated.recommended_weight == 0.85

    production = service.promote("lora-a")
    assert production.state is LoRAState.PRODUCTION
    database.dispose()


def test_promotion_rejects_below_threshold(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'threshold.sqlite3').as_posix()}")
    database.migrate()
    registry = LoRARegistry(database)
    registry.upsert(LoRAProfile(id="weak", path=tmp_path / "weak.safetensors"))
    service = LoRAValidationService(registry, LoRARegistryConfig())

    service.record(
        "weak",
        LoRAValidationReport(
            identity_score=0.5,
            quality_score=0.9,
            flexibility_score=0.9,
        ),
    )
    with pytest.raises(ValueError, match="identity score"):
        service.promote("weak")
    database.dispose()


def test_validation_report_rejects_inverted_weight_range() -> None:
    with pytest.raises(ValueError):
        LoRAValidationReport(
            identity_score=0.9,
            quality_score=0.9,
            flexibility_score=0.9,
            validated_min_weight=1.1,
            validated_max_weight=0.8,
        )
