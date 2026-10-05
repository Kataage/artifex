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
    def validate_total(self) -> "PlannerMixConfig":
        total = self.evergreen + self.trend + self.seasonal + self.exploration
        if abs(total - 1.0) > 1e-6:
            raise ValueError("planner mix weights must sum to 1.0")
        return self


class PlannerConfig(StrictModel):
    candidate_count: int = Field(default=8, ge=1)
    mix: PlannerMixConfig = Field(default_factory=PlannerMixConfig)


class ProductionConfig(StrictModel):
    retry_limit: int = Field(default=3, ge=0)
    completed_inventory_target: int | None = Field(default=7, ge=0)


class LlmConfig(StrictModel):
    backend: Literal["llama_cpp", "openai_compatible"] = "llama_cpp"
    base_url: str = "http://127.0.0.1:8080"
    model: str = "spark-x2.5-4b-heretic-jp"


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
    comfyui: ComfyUiConfig = Field(default_factory=ComfyUiConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    rights: RightsConfig = Field(default_factory=RightsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
