from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentConfig(StrictModel):
    autonomous: bool = True
    poll_interval_seconds: float = Field(default=10.0, gt=0)


class PlannerMixConfig(StrictModel):
    evergreen: float = Field(default=0.40, ge=0)
    trend: float = Field(default=0.30, ge=0)
    seasonal: float = Field(default=0.20, ge=0)
    exploration: float = Field(default=0.10, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> PlannerMixConfig:
        total = self.evergreen + self.trend + self.seasonal + self.exploration
        if abs(total - 1.0) > 1e-6:
            raise ValueError("planner mix weights must sum to 1.0")
        return self


class PlannerScoreWeightsConfig(StrictModel):
    trend: float = Field(default=0.15, ge=0)
    evergreen: float = Field(default=0.10, ge=0)
    character_fit: float = Field(default=0.15, ge=0)
    novelty: float = Field(default=0.15, ge=0)
    visual_strength: float = Field(default=0.15, ge=0)
    historical_performance: float = Field(default=0.10, ge=0)
    seasonality: float = Field(default=0.10, ge=0)
    series_potential: float = Field(default=0.05, ge=0)
    readiness: float = Field(default=0.05, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> PlannerScoreWeightsConfig:
        total = sum(
            (
                self.trend,
                self.evergreen,
                self.character_fit,
                self.novelty,
                self.visual_strength,
                self.historical_performance,
                self.seasonality,
                self.series_potential,
                self.readiness,
            )
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("planner positive score weights must sum to 1.0")
        return self


class PlannerConfig(StrictModel):
    candidate_count: int = Field(default=8, ge=1)
    mix: PlannerMixConfig = Field(default_factory=PlannerMixConfig)
    score_weights: PlannerScoreWeightsConfig = Field(
        default_factory=PlannerScoreWeightsConfig
    )
    similarity_penalty_weight: float = Field(default=0.35, ge=0)
    recent_character_penalty_weight: float = Field(default=0.15, ge=0)
    hard_similarity_threshold: float = Field(default=0.90, ge=0, le=1)


class ProductionConfig(StrictModel):
    retry_limit: int = Field(default=3, ge=0)
    idea_inventory_target: int = Field(default=30, ge=0)
    planned_inventory_target: int = Field(default=10, ge=0)
    completed_inventory_target: int | None = Field(default=7, ge=0)


class LlmConfig(StrictModel):
    backend: Literal["llama_cpp", "openai_compatible"] = "llama_cpp"
    base_url: str = "http://127.0.0.1:8080"
    model: str = "spark-x2.5-4b-heretic-jp"
    structured_output: Literal["json_schema", "json_object"] = "json_schema"
    temperature: float = Field(default=0.9, ge=0, le=2)
    timeout_seconds: float = Field(default=180.0, gt=0)
    request_attempts: int = Field(default=2, ge=1)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)
    structured_repair_attempts: int = Field(default=2, ge=0)
    api_key_env: str | None = None


class CharacterRegistryConfig(StrictModel):
    profile_dirs: tuple[Path, ...] = (Path("profiles/characters"),)
    minimum_readiness: float = Field(default=0.0, ge=0, le=1)


class LoRARegistryConfig(StrictModel):
    roots: tuple[Path, ...] = (Path("models/loras"),)
    extensions: tuple[str, ...] = (".safetensors",)
    metadata_header_max_mib: int = Field(default=16, ge=1)
    production_identity_threshold: float = Field(default=0.80, ge=0, le=1)
    production_quality_threshold: float = Field(default=0.70, ge=0, le=1)
    production_flexibility_threshold: float = Field(default=0.50, ge=0, le=1)
    maximum_character_loras_per_scene: int = Field(default=4, ge=1)


class ComfyUiConfig(StrictModel):
    base_url: str = "http://127.0.0.1:8188"


class DiscordConfig(StrictModel):
    enabled: bool = False
    token_env: str = "ARTIFEX_DISCORD_TOKEN"
    notify_completion: bool = True
    notify_review: bool = True
    notify_error: bool = True


class RightsConfig(StrictModel):
    enforce: bool = True


class StorageConfig(StrictModel):
    database_url: str = "sqlite:///data/artifex.sqlite3"
    packs_dir: Path = Path("data/packs")
    minimum_free_gib: float = Field(default=5.0, ge=0)


class ArtifexSettings(StrictModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    characters: CharacterRegistryConfig = Field(default_factory=CharacterRegistryConfig)
    loras: LoRARegistryConfig = Field(default_factory=LoRARegistryConfig)
    comfyui: ComfyUiConfig = Field(default_factory=ComfyUiConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    rights: RightsConfig = Field(default_factory=RightsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
