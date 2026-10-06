from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifex.characters import CharacterRegistry
from artifex.config.models import (
    ArtifexSettings,
    ComfyUiConfig,
    EvaluationConfig,
    ProductionConfig,
    StorageConfig,
)
from artifex.db import Database
from artifex.domain import CharacterProfile
from artifex.operations.doctor import DoctorService
from artifex.operations.health import ComponentHealth, ComponentState, HealthChecker
from artifex.telemetry import TelemetryRepository


def _write_calibration(path: Path) -> None:
    threshold = {
        "threshold": 0.8,
        "balanced_accuracy": 0.95,
        "positive_count": 8,
        "negative_count": 8,
        "positive_min": 0.75,
        "positive_mean": 0.9,
        "negative_max": 0.5,
        "negative_mean": 0.3,
        "margin": 0.25,
        "validated": True,
    }
    path.write_text(
        json.dumps(
            {
                "profile_id": "siglip2-hololive-ilxl-v1",
                "corpus_id": "doctor-test",
                "provider": "transformers_siglip2",
                "model": "google/siglip2-base-patch16-224",
                "revision": "test-revision",
                "generated_at": "2026-10-06T00:00:00Z",
                "identity_hard_min": 0.65,
                "identity_accept_min": 0.80,
                "similarity_hard_max": 0.85,
                "planner_hard_similarity_threshold": 0.82,
                "identity": {"purpose": "identity", **threshold},
                "duplicate": {"purpose": "duplicate", **threshold},
                "text_paraphrase": {
                    "purpose": "text_paraphrase",
                    **threshold,
                },
                "validated": True,
            }
        ),
        encoding="utf-8",
    )


class HealthyComfy:
    async def health(self):
        from artifex.comfy import ComfyHealth

        return ComfyHealth(available=True, version="test", devices=("gpu",))

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_doctor_reports_complete_production_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "artifex.operations.doctor.find_spec",
        lambda _name: object(),
    )
    calibration_path = tmp_path / "semantic-calibration.json"
    _write_calibration(calibration_path)
    database = Database(f"sqlite:///{(tmp_path / 'doctor.sqlite3').as_posix()}")
    database.migrate()
    settings = ArtifexSettings(
        production=ProductionConfig(checkpoint="ilxl.safetensors"),
        comfyui=ComfyUiConfig(output_dir=tmp_path / "comfy-output"),
        evaluation=EvaluationConfig(
            vision_base_url="http://vision.test",
            vision_model="vision-model",
            semantic_calibration_path=calibration_path,
        ),
        storage=StorageConfig(
            database_url=database.database_url,
            packs_dir=tmp_path / "packs",
            minimum_free_gib=0,
        ),
    )
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="Character A",
            namespace="test",
            readiness=1.0,
        )
    )
    telemetry = TelemetryRepository(database)

    async def healthy_llm() -> ComponentHealth:
        return ComponentHealth(
            name="llm",
            state=ComponentState.HEALTHY,
            detail="ok",
            blocking=True,
        )

    async def healthy_evaluator() -> ComponentHealth:
        return ComponentHealth(
            name="evaluator",
            state=ComponentState.HEALTHY,
            detail="ok",
            blocking=True,
        )

    health = HealthChecker(
        settings,
        database,
        telemetry,
        comfy=HealthyComfy(),  # type: ignore[arg-type]
        llm_probe=healthy_llm,
        evaluator_probe=healthy_evaluator,
    )
    report = await DoctorService(settings, health, characters).run()

    assert report.ready is True
    assert report.health.require("database").state is ComponentState.HEALTHY
    assert report.health.require("storage").state is ComponentState.HEALTHY
    assert report.health.require("discord").state is ComponentState.DISABLED
    assert all(check.ready for check in report.checks)
    database.dispose()


@pytest.mark.asyncio
async def test_doctor_fails_when_required_production_configuration_is_missing(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'doctor-missing.sqlite3').as_posix()}")
    database.migrate()
    settings = ArtifexSettings(
        storage=StorageConfig(
            database_url=database.database_url,
            packs_dir=tmp_path / "packs",
            minimum_free_gib=0,
        )
    )
    characters = CharacterRegistry(database)
    telemetry = TelemetryRepository(database)

    async def healthy_llm() -> ComponentHealth:
        return ComponentHealth(
            name="llm",
            state=ComponentState.HEALTHY,
            detail="ok",
            blocking=True,
        )

    async def healthy_evaluator() -> ComponentHealth:
        return ComponentHealth(
            name="evaluator",
            state=ComponentState.HEALTHY,
            detail="ok",
            blocking=True,
        )

    health = HealthChecker(
        settings,
        database,
        telemetry,
        comfy=HealthyComfy(),  # type: ignore[arg-type]
        llm_probe=healthy_llm,
        evaluator_probe=healthy_evaluator,
    )
    report = await DoctorService(settings, health, characters).run()
    checks = {check.name: check for check in report.checks}

    assert report.ready is False
    assert checks["production_checkpoint"].ready is False
    assert checks["comfy_output_dir"].ready is False
    assert checks["vision_evaluator"].ready is False
    assert checks["character_catalog"].ready is False
    database.dispose()
