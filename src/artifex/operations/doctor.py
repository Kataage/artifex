from __future__ import annotations

from importlib.util import find_spec

from pydantic import BaseModel, ConfigDict

from artifex.characters import CharacterRegistry
from artifex.comfy import (
    ComfyUIClient,
    WorkflowPatchRequest,
    WorkflowTemplateRegistry,
)
from artifex.config.models import ArtifexSettings
from artifex.domain import LoRAPolicy, LoRAState
from artifex.evaluation import load_calibration_profile
from artifex.loras import LoRARegistry
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
        *,
        comfy: ComfyUIClient | None = None,
        loras: LoRARegistry | None = None,
    ) -> None:
        self._settings = settings
        self._health = health
        self._characters = characters
        self._comfy = comfy
        self._loras = loras

    async def run(self) -> DoctorReport:
        health = await self._health.check_all(include_worker=False)
        semantic_dependencies = (
            self._settings.evaluation.semantic_provider == "siglip2"
            and find_spec("torch") is not None
            and find_spec("transformers") is not None
        )
        semantic_calibrated = False
        semantic_detail: str
        calibration_path = self._settings.evaluation.semantic_calibration_path
        if self._settings.evaluation.semantic_provider != "siglip2":
            semantic_detail = "degraded local_fallback is not production-ready"
        elif not semantic_dependencies:
            semantic_detail = (
                "install production semantic dependencies with: uv sync --extra semantic"
            )
        elif not calibration_path.is_file():
            semantic_detail = (
                f"semantic calibration is missing: {calibration_path}; "
                "run artifex semantic calibrate"
            )
        else:
            try:
                profile = load_calibration_profile(calibration_path)
                semantic_calibrated = (
                    profile.validated
                    and profile.provider == "transformers_siglip2"
                    and profile.profile_id
                    == self._settings.evaluation.semantic_calibration_profile
                    and profile.model == self._settings.evaluation.semantic_model
                    and (
                        self._settings.evaluation.semantic_revision is None
                        or profile.revision
                        == self._settings.evaluation.semantic_revision
                    )
                )
                semantic_detail = (
                    f"{profile.model}@{profile.revision}; "
                    f"corpus={profile.corpus_id}; validated={profile.validated}"
                )
            except (OSError, ValueError) as exc:
                semantic_detail = f"invalid semantic calibration: {exc}"

        workflow_check = await self._workflow_dependency_check()
        lora_check = self._lora_asset_check()
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
                name="semantic_embeddings",
                ready=semantic_dependencies and semantic_calibrated,
                detail=semantic_detail,
            ),
            workflow_check,
            lora_check,
            DoctorCheck(
                name="vram_release_policy",
                ready=self._settings.comfyui.release_vram_after_attempt,
                detail=(
                    "ComfyUI models/cache are released after every generation attempt"
                    if self._settings.comfyui.release_vram_after_attempt
                    else (
                        "comfyui.release_vram_after_attempt=false; long-running "
                        "production can retain avoidable model/VRAM residency"
                    )
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

    def _lora_asset_check(self) -> DoctorCheck:
        if self._loras is None:
            return DoctorCheck(
                name="lora_assets",
                ready=False,
                detail="LoRA registry probe is not attached",
            )
        profiles = self._loras.list()
        counts = {
            state.value: sum(1 for profile in profiles if profile.state is state)
            for state in LoRAState
        }
        missing_required: list[str] = []
        for character in self._characters.list(enabled_only=True):
            if character.lora_policy is not LoRAPolicy.REQUIRED:
                continue
            if not self._loras.for_character(
                character.id,
                model_family=self._settings.production.model_family,
                production_only=True,
            ):
                missing_required.append(character.id)
        detail = (
            " ".join(
                f"{state}={counts[state]}"
                for state in (
                    "production",
                    "pending",
                    "discovered",
                    "failed",
                    "disabled",
                    "validated",
                )
            )
        )
        if missing_required:
            detail += "; required character LoRA missing: " + ", ".join(
                sorted(missing_required)
            )
        return DoctorCheck(
            name="lora_assets",
            ready=not missing_required,
            detail=detail,
        )

    async def _workflow_dependency_check(self) -> DoctorCheck:
        if self._comfy is None:
            return DoctorCheck(
                name="comfy_workflow_dependencies",
                ready=False,
                detail="ComfyUI dependency probe is not attached",
            )

        templates = WorkflowTemplateRegistry.with_packaged_templates()
        template_ids = (
            self._settings.comfyui.default_template,
            self._settings.production.repair_workflow_template,
        )
        checkpoint = self._settings.production.checkpoint or "__MISSING_CHECKPOINT__"
        request = WorkflowPatchRequest(
            positive_prompt="artifex doctor",
            negative_prompt="",
            checkpoint=checkpoint,
            refiner_checkpoint=self._settings.comfyui.refiner_checkpoint,
            vae=self._settings.comfyui.vae,
            upscale_model=self._settings.comfyui.upscale_model,
            seed=1,
            width=self._settings.production.width,
            height=self._settings.production.height,
            batch_size=self._settings.production.batch_size,
            output_prefix="ARTIFEX/doctor",
            base_steps=self._settings.comfyui.base_steps,
            base_cfg=self._settings.comfyui.base_cfg,
            base_sampler=self._settings.comfyui.base_sampler,
            base_scheduler=self._settings.comfyui.base_scheduler,
            base_denoise=self._settings.comfyui.base_denoise,
            refiner_steps=self._settings.comfyui.refiner_steps,
            refiner_cfg=self._settings.comfyui.refiner_cfg,
            refiner_sampler=self._settings.comfyui.refiner_sampler,
            refiner_scheduler=self._settings.comfyui.refiner_scheduler,
            refiner_denoise=self._settings.comfyui.refiner_denoise,
            upscale_steps=self._settings.comfyui.upscale_steps,
            upscale_cfg=self._settings.comfyui.upscale_cfg,
            upscale_sampler=self._settings.comfyui.upscale_sampler,
            upscale_scheduler=self._settings.comfyui.upscale_scheduler,
            upscale_denoise=self._settings.comfyui.upscale_denoise,
        )

        details: list[str] = []
        ready = True
        try:
            for template_id in dict.fromkeys(template_ids):
                template = templates.require(template_id)
                status = await self._comfy.validate_requirements(
                    template.requirements(request)
                )
                ready = ready and status.ready
                details.append(f"{template_id}: {status.detail}")
        except (KeyError, ValueError, RuntimeError) as exc:
            return DoctorCheck(
                name="comfy_workflow_dependencies",
                ready=False,
                detail=f"production workflow qualification failed: {exc}",
            )

        return DoctorCheck(
            name="comfy_workflow_dependencies",
            ready=ready,
            detail=" | ".join(details),
        )
