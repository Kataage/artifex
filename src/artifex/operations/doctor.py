from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from artifex.characters import CharacterRegistry
from artifex.config.models import ArtifexSettings
from artifex.operations.health import HealthChecker, HealthReport


class DoctorCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ready: bool
    detail: str


class DoctorReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    health: HealthReport
    checks: tuple[DoctorCheck, ...]

    @property
    def ready(self) -> bool:
        return self.health.ready and all(check.ready for check in self.checks)


class DoctorService:
    def __init__(
        self,
        settings: ArtifexSettings,
        health: HealthChecker,
        characters: CharacterRegistry,
    ) -> None:
        self._settings = settings
        self._health = health
        self._characters = characters

    async def run(self) -> DoctorReport:
        health = await self._health.check_all(include_worker=False)
        checks = (
            DoctorCheck(
                name="production_checkpoint",
                ready=bool(self._settings.production.checkpoint),
                detail=(
                    f"Configured: {self._settings.production.checkpoint}"
                    if self._settings.production.checkpoint
                    else "production.checkpoint is not configured"
                ),
            ),
            DoctorCheck(
                name="comfy_output_dir",
                ready=self._settings.comfyui.output_dir is not None,
                detail=(
                    str(self._settings.comfyui.output_dir)
                    if self._settings.comfyui.output_dir is not None
                    else "comfyui.output_dir is not configured"
                ),
            ),
            DoctorCheck(
                name="vision_evaluator",
                ready=bool(
                    self._settings.evaluation.vision_base_url
                    and self._settings.evaluation.vision_model
                ),
                detail=(
                    f"{self._settings.evaluation.vision_base_url} / "
                    f"{self._settings.evaluation.vision_model}"
                    if (
                        self._settings.evaluation.vision_base_url
                        and self._settings.evaluation.vision_model
                    )
                    else "evaluation vision_base_url/model are not configured"
                ),
            ),
            DoctorCheck(
                name="native_research_provider",
                ready=(
                    not self._settings.research.required_for_ideation
                    or (
                        self._settings.research.enabled
                        and "ddgs" in self._settings.research.provider_order
                    )
                ),
                detail=(
                    "Windows-native DDGS provider is configured"
                    if "ddgs" in self._settings.research.provider_order
                    else "DDGS is missing from research.provider_order"
                ),
            ),
            DoctorCheck(
                name="character_catalog",
                ready=bool(self._characters.list(enabled_only=True)),
                detail=(
                    f"{len(self._characters.list(enabled_only=True))} enabled character(s)"
                ),
            ),
        )
        return DoctorReport(health=health, checks=checks)
