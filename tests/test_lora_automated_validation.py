from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from artifex.config.models import LoRARegistryConfig
from artifex.db import Database
from artifex.domain import LoRAProfile, LoRAState
from artifex.loras import (
    LoRARegistry,
    LoRAValidationCase,
    LoRAValidationMatrixRunner,
    LoRAValidationRunRepository,
    LoRAValidationSample,
    LoRAValidationService,
)


class FakeProbe:
    def __init__(self, *, weak: bool = False) -> None:
        self.weak = weak
        self.calls: list[tuple[str, float, int]] = []

    def provenance(self) -> dict[str, Any]:
        return {"probe": "fake", "weak": self.weak}

    async def evaluate(
        self,
        profile: LoRAProfile,
        case: LoRAValidationCase,
        *,
        weight: float,
        seed: int,
        run_id: str,
    ) -> LoRAValidationSample:
        self.calls.append((case.id, weight, seed))
        if self.weak:
            identity = 0.40
            quality = 0.85
            aggregate = 0.60
        elif weight >= 0.99:
            identity = 0.95
            quality = 0.90
            aggregate = 0.92
        else:
            identity = 0.82
            quality = 0.75
            aggregate = 0.80
        return LoRAValidationSample(
            case_id=case.id,
            weight=weight,
            seed=seed,
            character_id="char-a",
            prompt_id=f"prompt-{case.id}-{weight}",
            image_path=Path(f"/tmp/{run_id}-{case.id}-{weight}.png"),
            identity_score=identity,
            quality_score=quality,
            aggregate_score=aggregate,
            result_state="accepted",
            scores={"identity": identity, "aggregate": aggregate},
            prompt="char-a, smile",
            negative_prompt="bad_hands",
        )


def _setup(
    tmp_path: Path,
    *,
    weak: bool = False,
) -> tuple[
    Database,
    LoRARegistry,
    LoRAValidationRunRepository,
    LoRAValidationMatrixRunner,
    FakeProbe,
]:
    database = Database(
        f"sqlite:///{(tmp_path / 'automated-lora.sqlite3').as_posix()}"
    )
    database.migrate()
    registry = LoRARegistry(database)
    path = tmp_path / "candidate.safetensors"
    path.write_bytes(b"candidate")
    registry.upsert(
        LoRAProfile(
            id="candidate",
            path=path,
            state=LoRAState.DISCOVERED,
            lora_type="character",
            target_character_ids=("char-a",),
            model_families=("ilxl",),
            checksum="checksum-v1",
        )
    )
    config = LoRARegistryConfig(
        validation_weights=(0.70, 1.00),
        production_identity_threshold=0.80,
        production_quality_threshold=0.70,
        production_flexibility_threshold=0.50,
    )
    runs = LoRAValidationRunRepository(database)
    probe = FakeProbe(weak=weak)
    runner = LoRAValidationMatrixRunner(
        registry,
        LoRAValidationService(registry, config),
        runs,
        probe,
        config,
    )
    return database, registry, runs, runner, probe


@pytest.mark.asyncio
async def test_validation_matrix_promotes_from_reproducible_evidence(
    tmp_path: Path,
) -> None:
    database, registry, runs, runner, probe = _setup(tmp_path)

    run = await runner.run("candidate")

    assert run.status == "promoted"
    assert len(probe.calls) == 6
    assert len(run.evidence["samples"]) == 6
    assert run.report is not None
    assert run.report["recommended_weight"] == 1.0
    assert run.report["validated_min_weight"] == 0.7
    assert run.report["validated_max_weight"] == 1.0

    profile = registry.require("candidate")
    assert profile.state is LoRAState.PRODUCTION
    assert profile.last_validation_at is not None
    assert profile.last_validation_run_id == run.id
    assert profile.recommended_weight == 1.0
    assert runs.require(run.id).status == "promoted"
    database.dispose()


@pytest.mark.asyncio
async def test_validation_matrix_rejects_weak_identity_and_persists_failure(
    tmp_path: Path,
) -> None:
    database, registry, runs, runner, _ = _setup(tmp_path, weak=True)

    run = await runner.run("candidate")

    assert run.status == "rejected"
    assert run.report is not None
    assert run.error is not None
    assert "identity score" in run.error["reason"]
    assert registry.require("candidate").state is LoRAState.FAILED
    assert runs.require(run.id).status == "rejected"
    database.dispose()


@pytest.mark.asyncio
async def test_run_pending_only_consumes_bounded_assets(tmp_path: Path) -> None:
    database, registry, _, runner, probe = _setup(tmp_path)
    second_path = tmp_path / "candidate-2.safetensors"
    second_path.write_bytes(b"candidate-2")
    registry.upsert(
        LoRAProfile(
            id="candidate-2",
            path=second_path,
            state=LoRAState.DISCOVERED,
            target_character_ids=("char-a",),
            checksum="checksum-v2",
        )
    )

    completed = await runner.run_pending(limit=1)

    assert len(completed) == 1
    states = {
        profile.id: profile.state
        for profile in registry.list()
    }
    assert list(states.values()).count(LoRAState.PRODUCTION) == 1
    assert list(states.values()).count(LoRAState.DISCOVERED) == 1
    assert len(probe.calls) == 6
    database.dispose()
