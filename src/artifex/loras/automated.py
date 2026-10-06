from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from artifex.characters import CharacterRegistry
from artifex.config.models import LoRARegistryConfig
from artifex.domain import LoRAProfile, LoRAState
from artifex.evaluation import EvaluationContext, EvaluationEngine
from artifex.loras.registry import LoRARegistry
from artifex.loras.resolver import (
    LoRAPlan,
    LoRAPlanEntry,
    LoRAResolutionError,
    LoRAResolver,
)
from artifex.loras.runs import LoRAValidationRun, LoRAValidationRunRepository
from artifex.loras.validation import LoRAValidationReport, LoRAValidationService
from artifex.packs import ScenePlan, VisualSpecification
from artifex.production.backend import GenerationBackend, GenerationRequest
from artifex.prompts import PromptCompiler


class AutomatedValidationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoRAValidationCase(AutomatedValidationModel):
    id: str = Field(min_length=1)
    composition: str
    camera: str
    pose: str
    expression: str
    clothing: str
    setting: str
    lighting: str
    atmosphere: str


class LoRAValidationSample(AutomatedValidationModel):
    case_id: str
    weight: float
    seed: int
    character_id: str
    prompt_id: str
    image_path: Path
    identity_score: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    aggregate_score: float = Field(ge=0, le=1)
    result_state: str
    scores: dict[str, float] = Field(default_factory=dict)
    prompt: str
    negative_prompt: str


DEFAULT_VALIDATION_CASES: tuple[LoRAValidationCase, ...] = (
    LoRAValidationCase(
        id="portrait",
        composition="upper body",
        camera="eye level",
        pose="standing",
        expression="smile",
        clothing="shirt",
        setting="bedroom",
        lighting="soft lighting",
        atmosphere="depth of field",
    ),
    LoRAValidationCase(
        id="full_body",
        composition="full body",
        camera="straight on",
        pose="standing",
        expression="closed mouth",
        clothing="jacket",
        setting="street",
        lighting="sunlight",
        atmosphere="wind",
    ),
    LoRAValidationCase(
        id="dynamic",
        composition="cowboy shot",
        camera="side view",
        pose="walking",
        expression="smile",
        clothing="hoodie",
        setting="park",
        lighting="soft lighting",
        atmosphere="bokeh",
    ),
)


class LoRAValidationProbe(Protocol):
    def provenance(self) -> Mapping[str, Any]: ...

    async def evaluate(
        self,
        profile: LoRAProfile,
        case: LoRAValidationCase,
        *,
        weight: float,
        seed: int,
        run_id: str,
    ) -> LoRAValidationSample: ...


def _reference_images(character_dirs: tuple[Path, ...]) -> tuple[Path, ...]:
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    paths: list[Path] = []
    for directory in character_dirs:
        root = directory.expanduser()
        if not root.exists():
            continue
        paths.extend(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in extensions
        )
    return tuple(sorted(dict.fromkeys(paths), key=lambda item: item.as_posix().casefold()))


class ProductionLoRAValidationProbe:
    def __init__(
        self,
        characters: CharacterRegistry,
        resolver: LoRAResolver,
        prompts: PromptCompiler,
        backend: GenerationBackend,
        evaluator: EvaluationEngine,
        config: LoRARegistryConfig,
        *,
        model_family: str,
    ) -> None:
        self._characters = characters
        self._resolver = resolver
        self._prompts = prompts
        self._backend = backend
        self._evaluator = evaluator
        self._config = config
        self._model_family = model_family

    def provenance(self) -> Mapping[str, Any]:
        return {
            "probe": "production_generation_evaluation",
            "model_family": self._model_family,
            "backend": dict(self._backend.provenance()),
            "cases": [case.id for case in DEFAULT_VALIDATION_CASES],
        }

    async def evaluate(
        self,
        profile: LoRAProfile,
        case: LoRAValidationCase,
        *,
        weight: float,
        seed: int,
        run_id: str,
    ) -> LoRAValidationSample:
        character_id = self._validation_character(profile)
        character = self._characters.require(character_id)
        scene = ScenePlan(
            ordinal=1,
            title=f"LoRA validation {profile.id}/{case.id}",
            purpose="reproducible LoRA validation",
            character_ids=(character_id,),
            visual=VisualSpecification(
                composition=case.composition,
                camera=case.camera,
                pose=case.pose,
                expression=case.expression,
                clothing=case.clothing,
                setting=case.setting,
                lighting=case.lighting,
                atmosphere=case.atmosphere,
            ),
        )
        plan = self._validation_plan(
            profile,
            character_id=character_id,
            clothing=case.clothing,
            weight=weight,
        )
        compiled = self._prompts.compile(scene, plan)
        attempt_id = (
            f"lora-val-{run_id[:12]}-{case.id}-"
            f"{str(weight).replace('.', '_')}"
        )
        output_prefix = (
            f"{self._config.validation_output_prefix}/"
            f"{profile.id}/{run_id}/{case.id}-{weight:.2f}"
        )
        generated = await self._backend.generate(
            GenerationRequest(
                attempt_id=attempt_id,
                scene_id=attempt_id,
                compiled=compiled,
                lora_plan=plan,
                seed=seed,
                output_prefix=output_prefix,
            ),
            on_submitted=lambda _prompt_id: None,
        )
        if not generated.output_paths:
            raise RuntimeError("LoRA validation generation returned no image")
        image_path = generated.output_paths[0]
        identity_refs = _reference_images(character.reference_image_dirs)
        result = await self._evaluator.evaluate(
            EvaluationContext(
                attempt_id=attempt_id,
                scene_id=attempt_id,
                image_path=image_path,
                character_ids=(character_id,),
                positive_prompt=compiled.positive_prompt,
                negative_prompt=compiled.negative_prompt,
                identity_reference_image_paths={
                    character_id: identity_refs,
                },
            )
        )
        identity = (
            result.scores.identity_reference
            if result.scores.identity_reference is not None
            else result.scores.identity
        )
        quality = fmean(
            (
                result.scores.face_quality,
                result.scores.technical_quality,
                result.scores.aesthetic,
                result.scores.integrity,
            )
        )
        return LoRAValidationSample(
            case_id=case.id,
            weight=weight,
            seed=seed,
            character_id=character_id,
            prompt_id=generated.prompt_id,
            image_path=image_path,
            identity_score=identity,
            quality_score=quality,
            aggregate_score=result.scores.aggregate,
            result_state=result.state.value,
            scores={
                key: float(value)
                for key, value in result.scores.model_dump().items()
                if isinstance(value, int | float)
            },
            prompt=compiled.positive_prompt,
            negative_prompt=compiled.negative_prompt,
        )

    def _validation_character(self, profile: LoRAProfile) -> str:
        candidates = (
            *profile.target_character_ids,
            *self._config.validation_character_ids,
        )
        for character_id in dict.fromkeys(candidates):
            character = self._characters.get(character_id)
            if character is not None and character.enabled:
                return character_id
        for character in self._characters.list(enabled_only=True):
            return character.id
        raise RuntimeError("no enabled character available for LoRA validation")

    def _validation_plan(
        self,
        profile: LoRAProfile,
        *,
        character_id: str,
        clothing: str,
        weight: float,
    ) -> LoRAPlan:
        candidate = LoRAPlanEntry(
            lora_id=profile.id,
            path=str(profile.path),
            weight=weight,
            character_ids=(character_id,),
            trigger_tags=profile.trigger_tags,
            layer=profile.lora_type,
        )
        if profile.lora_type == "character":
            return LoRAPlan(
                model_family=self._model_family,
                character_ids=(character_id,),
                entries=(candidate,),
            )

        try:
            base = self._resolver.resolve(
                (character_id,),
                model_family=self._model_family,
                exclude_lora_ids=(profile.id,),
                clothing=clothing,
            )
        except LoRAResolutionError as exc:
            raise RuntimeError(
                "cannot build base character stack for LoRA validation: "
                + str(exc)
            ) from exc
        layer_order = LoRAResolver.LAYER_ORDER
        entries = tuple(
            sorted(
                (*base.entries, candidate),
                key=lambda entry: (
                    layer_order.get(entry.layer, 99),
                    entry.lora_id,
                ),
            )
        )
        return base.model_copy(update={"entries": entries})


class LoRAValidationMatrixRunner:
    def __init__(
        self,
        registry: LoRARegistry,
        service: LoRAValidationService,
        runs: LoRAValidationRunRepository,
        probe: LoRAValidationProbe,
        config: LoRARegistryConfig,
        *,
        cases: tuple[LoRAValidationCase, ...] = DEFAULT_VALIDATION_CASES,
    ) -> None:
        self._registry = registry
        self._service = service
        self._runs = runs
        self._probe = probe
        self._config = config
        self._cases = cases

    async def run(self, lora_id: str) -> LoRAValidationRun:
        profile = self._registry.require(lora_id)
        if not profile.path.is_file():
            invalidated = self._registry.invalidate(
                lora_id,
                target=LoRAState.DISABLED,
                reason=f"validation asset missing: {profile.path}",
                missing=True,
            )
            return self._runs.finish(
                self._runs.start(
                    lora_id,
                    checksum=profile.checksum,
                    evidence={"asset_path": str(profile.path)},
                ).id,
                status="missing",
                evidence={"asset_path": str(profile.path)},
                error={"reason": f"missing asset: {invalidated.path}"},
            )

        matrix = {
            "weights": list(self._config.validation_weights),
            "cases": [case.model_dump(mode="json") for case in self._cases],
            "probe": dict(self._probe.provenance()),
            "samples": [],
        }
        run = self._runs.start(
            lora_id,
            checksum=profile.checksum,
            evidence=matrix,
        )
        try:
            samples: list[LoRAValidationSample] = []
            for case_index, case in enumerate(self._cases):
                for weight_index, weight in enumerate(
                    self._config.validation_weights
                ):
                    seed = (
                        self._config.validation_seed
                        + case_index * 1000
                        + weight_index
                    )
                    sample = await self._probe.evaluate(
                        profile,
                        case,
                        weight=weight,
                        seed=seed,
                        run_id=run.id,
                    )
                    samples.append(sample)
                    matrix["samples"] = [
                        item.model_dump(mode="json") for item in samples
                    ]
                    self._runs.update_evidence(run.id, matrix)

            current = self._registry.require(lora_id)
            if current.checksum != run.checksum:
                return self._runs.finish(
                    run.id,
                    status="stale",
                    evidence=matrix,
                    error={
                        "reason": "asset checksum changed during validation",
                        "started_checksum": run.checksum,
                        "current_checksum": current.checksum,
                    },
                )

            report = self._report(samples)
            self._service.record(
                lora_id,
                report,
                run_id=run.id,
            )
            try:
                promoted = self._service.promote(lora_id)
            except ValueError as exc:
                self._registry.transition_state(lora_id, LoRAState.FAILED)
                return self._runs.finish(
                    run.id,
                    status="rejected",
                    evidence=matrix,
                    report=report.model_dump(mode="json"),
                    error={"reason": str(exc)},
                )

            return self._runs.finish(
                run.id,
                status="promoted",
                evidence={
                    **matrix,
                    "promoted_profile": promoted.model_dump(mode="json"),
                },
                report=report.model_dump(mode="json"),
            )
        except Exception as exc:  # noqa: BLE001
            current = self._registry.get(lora_id)
            if (
                current is not None
                and current.checksum == run.checksum
                and current.state is not LoRAState.DISABLED
            ):
                self._registry.invalidate(
                    lora_id,
                    target=LoRAState.FAILED,
                    reason=f"automated validation failed: {exc}",
                )
            return self._runs.finish(
                run.id,
                status="failed",
                evidence=matrix,
                error={
                    "type": type(exc).__name__,
                    "reason": str(exc)[:2000],
                },
            )

    async def run_pending(self, *, limit: int) -> tuple[LoRAValidationRun, ...]:
        candidates = [
            profile
            for profile in self._registry.list(
                states=(LoRAState.DISCOVERED, LoRAState.PENDING)
            )
            if profile.path.is_file()
        ]
        candidates.sort(
            key=lambda profile: (
                profile.last_validation_at is not None,
                profile.id,
            )
        )
        results: list[LoRAValidationRun] = []
        for profile in candidates[:limit]:
            results.append(await self.run(profile.id))
        return tuple(results)

    def _report(
        self,
        samples: list[LoRAValidationSample],
    ) -> LoRAValidationReport:
        if not samples:
            raise ValueError("LoRA validation matrix produced no samples")

        weight_scores: dict[float, tuple[float, float, float]] = {}
        for weight in self._config.validation_weights:
            weighted = [sample for sample in samples if sample.weight == weight]
            if not weighted:
                continue
            weight_identity = fmean(sample.identity_score for sample in weighted)
            weight_quality = fmean(sample.quality_score for sample in weighted)
            weight_scores[weight] = (
                weight_identity,
                weight_quality,
                fmean(
                    (sample.identity_score + sample.quality_score) / 2
                    for sample in weighted
                ),
            )
        if not weight_scores:
            raise ValueError("LoRA validation matrix has no configured weight samples")

        best_weight = max(
            weight_scores,
            key=lambda weight: (weight_scores[weight][2], -abs(weight - 1.0)),
        )
        best_samples = [
            sample for sample in samples if sample.weight == best_weight
        ]
        identity = fmean(sample.identity_score for sample in best_samples)
        quality = fmean(sample.quality_score for sample in best_samples)

        case_scores: list[float] = []
        for case in self._cases:
            case_samples = [
                sample
                for sample in best_samples
                if sample.case_id == case.id
            ]
            if not case_samples:
                continue
            case_scores.append(
                fmean(
                    min(sample.identity_score, sample.quality_score)
                    for sample in case_samples
                )
            )
        flexibility = min(case_scores) if case_scores else 0.0

        passing = sorted(
            weight
            for weight, values in weight_scores.items()
            if (
                values[0] >= self._config.production_identity_threshold
                and values[1] >= self._config.production_quality_threshold
            )
        )
        return LoRAValidationReport(
            identity_score=identity,
            quality_score=quality,
            flexibility_score=flexibility,
            recommended_weight=best_weight,
            validated_min_weight=passing[0] if passing else best_weight,
            validated_max_weight=passing[-1] if passing else best_weight,
        )
