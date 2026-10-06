from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.config.models import LoRARegistryConfig
from artifex.domain import LoRAProfile, LoRAState
from artifex.loras.registry import LoRARegistry


class LoRAValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identity_score: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    flexibility_score: float = Field(ge=0, le=1)
    recommended_weight: float | None = Field(default=None, ge=-4, le=4)
    validated_min_weight: float | None = Field(default=None, ge=-4, le=4)
    validated_max_weight: float | None = Field(default=None, ge=-4, le=4)

    @model_validator(mode="after")
    def validate_weight_range(self) -> LoRAValidationReport:
        if (
            self.validated_min_weight is not None
            and self.validated_max_weight is not None
            and self.validated_min_weight > self.validated_max_weight
        ):
            raise ValueError("validated_min_weight must be <= validated_max_weight")
        return self


class LoRAValidationService:
    def __init__(
        self,
        registry: LoRARegistry,
        config: LoRARegistryConfig,
    ) -> None:
        self._registry = registry
        self._config = config

    def record(
        self,
        lora_id: str,
        report: LoRAValidationReport,
        *,
        run_id: str | None = None,
        validated_at: datetime | None = None,
    ) -> LoRAProfile:
        profile = self._registry.require(lora_id)
        if profile.state in {LoRAState.DISABLED, LoRAState.FAILED, LoRAState.DISCOVERED}:
            profile = self._registry.transition_state(lora_id, LoRAState.PENDING)
        if profile.state is LoRAState.PRODUCTION:
            profile = self._registry.transition_state(lora_id, LoRAState.VALIDATED)

        readiness = (
            report.identity_score * 0.50
            + report.quality_score * 0.30
            + report.flexibility_score * 0.20
        )
        payload = profile.model_dump(mode="python")
        payload.update(
            {
                "state": LoRAState.VALIDATED,
                "identity_score": report.identity_score,
                "quality_score": report.quality_score,
                "flexibility_score": report.flexibility_score,
                "readiness": readiness,
                "last_validation_at": validated_at or datetime.now(UTC),
                "last_validation_run_id": run_id,
                "missing_since": None,
            }
        )
        if report.recommended_weight is not None:
            payload["recommended_weight"] = report.recommended_weight
        if report.validated_min_weight is not None:
            payload["validated_min_weight"] = report.validated_min_weight
        if report.validated_max_weight is not None:
            payload["validated_max_weight"] = report.validated_max_weight

        validated = LoRAProfile.model_validate(payload)
        return self._registry.upsert(validated)

    def promote(self, lora_id: str) -> LoRAProfile:
        profile = self._registry.require(lora_id)
        if profile.state is not LoRAState.VALIDATED:
            raise ValueError("only validated LoRAs can be promoted to production")
        if profile.identity_score is None or profile.quality_score is None:
            raise ValueError("validation scores are missing")
        if profile.flexibility_score is None:
            raise ValueError("flexibility score is missing")
        if profile.identity_score < self._config.production_identity_threshold:
            raise ValueError("identity score is below production threshold")
        if profile.quality_score < self._config.production_quality_threshold:
            raise ValueError("quality score is below production threshold")
        if profile.flexibility_score < self._config.production_flexibility_threshold:
            raise ValueError("flexibility score is below production threshold")
        return self._registry.transition_state(lora_id, LoRAState.PRODUCTION)
