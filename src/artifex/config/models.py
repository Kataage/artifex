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
    infrastructure_retry_limit: int = Field(default=5, ge=0)
    retry_alternate_lora_limit: int = Field(default=1, ge=0)
    retry_adjust_lora_weight_limit: int = Field(default=2, ge=0)
    retry_revise_prompt_limit: int = Field(default=2, ge=0)
    retry_repair_workflow_limit: int = Field(default=1, ge=0)
    retry_vary_scene_limit: int = Field(default=2, ge=0)
    retry_change_seed_limit: int = Field(default=3, ge=0)
    retry_lora_weight_step: float = Field(default=0.10, gt=0, le=0.5)
    repair_workflow_template: str = "illust_main_repair_v1"
    idea_inventory_target: int = Field(default=30, ge=0)
    planned_inventory_target: int = Field(default=10, ge=0)
    completed_inventory_target: int | None = Field(default=7, ge=0)
    idea_replenish_batch: int = Field(default=3, ge=1)
    model_family: str = "ilxl"
    checkpoint: str | None = None
    width: int = Field(default=1024, ge=64, le=8192)
    height: int = Field(default=1536, ge=64, le=8192)
    batch_size: int = Field(default=1, ge=1, le=64)


class EditorialConfig(StrictModel):
    mode: Literal["continuous", "watermark"] = "watermark"
    inventory_low_watermark: int = Field(default=3, ge=0)
    inventory_high_watermark: int = Field(default=7, ge=1)
    inventory_expiry_hours: float | None = Field(default=168.0, gt=0)
    series_target_share: float = Field(default=0.35, ge=0, le=1)
    cadence_window_packs: int = Field(default=10, ge=1, le=100)
    max_consecutive_series: int = Field(default=2, ge=1, le=20)
    max_consecutive_standalone: int = Field(default=4, ge=1, le=20)
    diversity_window_packs: int = Field(default=12, ge=1, le=100)
    character_cooldown_packs: int = Field(default=2, ge=0, le=20)
    character_diversity_weight: float = Field(default=0.50, ge=0)
    branch_diversity_weight: float = Field(default=0.30, ge=0)
    group_diversity_weight: float = Field(default=0.20, ge=0)
    format_diversity_weight: float = Field(default=0.15, ge=0)
    theme_diversity_weight: float = Field(default=0.15, ge=0)
    series_candidate_count: int = Field(default=4, ge=1, le=16)

    @model_validator(mode="after")
    def validate_editorial(self) -> EditorialConfig:
        if self.inventory_low_watermark >= self.inventory_high_watermark:
            raise ValueError(
                "editorial inventory_low_watermark must be below high watermark"
            )
        total = (
            self.character_diversity_weight
            + self.branch_diversity_weight
            + self.group_diversity_weight
            + self.format_diversity_weight
            + self.theme_diversity_weight
        )
        if total <= 0:
            raise ValueError("editorial diversity weights must have positive total")
        return self


class LlmModelProfileConfig(StrictModel):
    source: Literal["huggingface", "url", "local"] = "huggingface"
    repository: str | None = None
    revision: str = "main"
    filename: str | None = None
    url: str | None = None
    path: Path | None = None
    sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_source(self) -> LlmModelProfileConfig:
        if self.source == "huggingface" and not (self.repository and self.filename):
            raise ValueError("huggingface LLM profile requires repository and filename")
        if self.source == "url" and not (self.url and self.filename):
            raise ValueError("url LLM profile requires url and filename")
        if self.source == "local" and self.path is None:
            raise ValueError("local LLM profile requires path")
        return self


class LlmBootstrapConfig(StrictModel):
    enabled: bool = True
    auto_download: bool = True
    profile: str = "spark-x2.5-4b-heretic-jp-q8_0"
    models_dir: Path = Path("models/llm")
    profiles: dict[str, LlmModelProfileConfig] = Field(
        default_factory=lambda: {
            "spark-x2.5-4b-heretic-jp-q8_0": LlmModelProfileConfig(
                source="huggingface",
                repository="soyaakinohara/Spark-X2.5-4B-Heretic-jp-gguf",
                revision="f01809e437fb3046ae9afd31d684556f9ae5dd46",
                filename="Spark-X2.5-4B-Heretic-jp-Q8_0.gguf",
            )
        }
    )

    def selected(self) -> LlmModelProfileConfig:
        try:
            return self.profiles[self.profile]
        except KeyError as exc:
            raise ValueError(f"unknown LLM bootstrap profile: {self.profile}") from exc

    def model_path(self) -> Path:
        profile = self.selected()
        if profile.source == "local":
            assert profile.path is not None
            return profile.path
        assert profile.filename is not None
        return self.models_dir / profile.filename


class LlmServerConfig(StrictModel):
    # Never launch binaries implicitly from third-party installs. Enable only
    # when the operator selects their local llama.cpp executable.
    enabled: bool = False
    executable: str = "llama-server"
    gpu_layers: int | Literal["auto", "all"] = "auto"
    device: str | None = None
    startup_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    poll_seconds: float = Field(default=15.0, gt=0, le=3600)
    restart_limit: int = Field(default=3, ge=0, le=20)
    restart_backoff_seconds: float = Field(default=5.0, ge=0, le=300)
    log_path: Path = Path("data/logs/llama-server.log")

    @model_validator(mode="after")
    def validate_executable(self) -> LlmServerConfig:
        if not self.executable.strip():
            raise ValueError("llm.server.executable must not be empty")
        if self.device is not None and not self.device.strip():
            raise ValueError("llm.server.device must not be empty")
        if isinstance(self.gpu_layers, int) and self.gpu_layers < 0:
            raise ValueError("llm.server.gpu_layers must be nonnegative, auto or all")
        return self


class LlmConfig(StrictModel):
    backend: Literal["llama_cpp", "openai_compatible"] = "llama_cpp"
    base_url: str = "http://127.0.0.1:8899"
    model: str = "spark-x2.5-4b-heretic-jp"
    bootstrap: LlmBootstrapConfig = Field(default_factory=LlmBootstrapConfig)
    server: LlmServerConfig = Field(default_factory=LlmServerConfig)
    structured_output: Literal["json_schema", "json_object"] = "json_schema"
    temperature: float = Field(default=0.9, ge=0, le=2)
    timeout_seconds: float = Field(default=180.0, gt=0)
    request_attempts: int = Field(default=2, ge=1)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)
    structured_repair_attempts: int = Field(default=2, ge=0)
    api_key_env: str | None = None
    context_window_tokens: int = Field(default=8192, ge=2048)
    reserved_output_tokens: int = Field(default=1536, ge=256)
    repair_headroom_tokens: int = Field(default=512, ge=0)
    enforce_token_budget: bool = True
    token_count_mode: Literal["llama_cpp", "estimate", "disabled"] = "llama_cpp"

    @model_validator(mode="after")
    def validate_context_budget(self) -> LlmConfig:
        reserved = self.reserved_output_tokens + self.repair_headroom_tokens
        if reserved >= self.context_window_tokens:
            raise ValueError(
                "LLM reserved output + repair headroom must be below context window"
            )
        return self

    @property
    def max_input_tokens(self) -> int:
        return (
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.repair_headroom_tokens
        )


class ContextConfig(StrictModel):
    version: str = "context-v1"
    max_characters: int = Field(default=18, ge=1, le=100)
    max_recent_concepts: int = Field(default=12, ge=0, le=100)
    max_long_term_concepts: int = Field(default=8, ge=0, le=100)
    max_research_items: int = Field(default=12, ge=0, le=100)
    max_trend_signals: int = Field(default=12, ge=0, le=100)
    character_tokens: int = Field(default=900, ge=100)
    research_tokens: int = Field(default=1400, ge=100)
    recent_history_tokens: int = Field(default=700, ge=0)
    long_term_tokens: int = Field(default=700, ge=0)
    trend_tokens: int = Field(default=500, ge=0)
    evergreen_tokens: int = Field(default=300, ge=0)
    operator_tokens: int = Field(default=200, ge=0)
    series_tokens: int = Field(default=1200, ge=200)
    series_recent_pack_ids: int = Field(default=3, ge=0, le=20)
    long_term_candidate_limit: int = Field(default=500, ge=1, le=10000)


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
    maximum_total_loras_per_scene: int = Field(default=8, ge=1)
    default_style_lora_ids: tuple[str, ...] = ()
    default_utility_lora_ids: tuple[str, ...] = ()
    rescan_interval_seconds: float = Field(default=300.0, gt=0)
    validation_enabled: bool = True
    validation_interval_seconds: float = Field(default=60.0, gt=0)
    validation_assets_per_cycle: int = Field(default=1, ge=1, le=20)
    validation_weights: tuple[float, ...] = (0.70, 0.85, 1.00)
    validation_character_ids: tuple[str, ...] = ()
    validation_seed: int = Field(default=730241, ge=0)
    validation_output_prefix: str = "ARTIFEX/validation"

    @model_validator(mode="after")
    def validate_lora_lifecycle(self) -> LoRARegistryConfig:
        if self.maximum_total_loras_per_scene < self.maximum_character_loras_per_scene:
            raise ValueError(
                "maximum_total_loras_per_scene must be >= "
                "maximum_character_loras_per_scene"
            )
        if not self.validation_weights:
            raise ValueError("validation_weights must not be empty")
        if any(weight < -4 or weight > 4 for weight in self.validation_weights):
            raise ValueError("validation_weights must be within [-4, 4]")
        return self


class PromptCompilerConfig(StrictModel):
    ilxl_profile: str = "ilxl-danbooru-v2"
    unresolved_policy: Literal["report", "error"] = "report"
    checkpoint_profile_overrides: dict[str, str] = Field(default_factory=dict)
    extra_lexicon_paths: tuple[Path, ...] = ()


class RenderNodeConfig(StrictModel):
    type: Literal["comfyui"] = "comfyui"
    enabled: bool = True
    base_url: str
    output_mode: Literal["filesystem", "api"] = "api"
    output_dir: Path | None = None
    download_dir: Path = Path("data/render-cache")
    attestation_url: str | None = None
    attestation_token_env: str | None = "ARTIFEX_RENDER_NODE_TOKEN"


class RenderNodesConfig(StrictModel):
    primary: str | None = None
    nodes: dict[str, RenderNodeConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_primary(self) -> RenderNodesConfig:
        if not self.nodes:
            if self.primary is not None:
                raise ValueError("render_nodes.primary requires at least one node")
            return self
        if self.primary is None:
            raise ValueError("render_nodes.primary is required when nodes are configured")
        node = self.nodes.get(self.primary)
        if node is None:
            raise ValueError(f"unknown primary render node: {self.primary}")
        if not node.enabled:
            raise ValueError(f"primary render node is disabled: {self.primary}")
        return self

    def primary_node(self) -> tuple[str, RenderNodeConfig] | None:
        if not self.nodes or self.primary is None:
            return None
        return self.primary, self.nodes[self.primary]


class ManagedComfyConfig(StrictModel):
    """Optional renderer-local ComfyUI child process, never a remote service."""

    enabled: bool = False
    executable: Path | None = None
    working_directory: Path | None = None
    arguments: tuple[str, ...] = ()
    log_path: Path = Path("data/logs/comfyui.log")
    startup_timeout_seconds: float = Field(default=180.0, gt=0, le=1800)
    poll_seconds: float = Field(default=15.0, gt=0, le=3600)
    restart_limit: int = Field(default=3, ge=0, le=20)
    restart_backoff_seconds: float = Field(default=10.0, ge=0, le=600)

    @model_validator(mode="after")
    def validate_launch(self) -> ManagedComfyConfig:
        if self.enabled and (
            self.executable is None or self.working_directory is None
        ):
            raise ValueError(
                "managed ComfyUI requires executable and working_directory"
            )
        if any(not item.strip() for item in self.arguments):
            raise ValueError("managed ComfyUI arguments must be nonempty")
        return self


class RendererGatewayConfig(StrictModel):
    """Optional authenticated API-only LAN gateway for owned local ComfyUI."""

    enabled: bool = False
    bind_host: str = "127.0.0.1"
    port: int = Field(default=8191, ge=1, le=65535)
    admission_path: Path = Path("data/render-gateway-fence.sqlite3")


class RenderAgentConfig(StrictModel):
    node_id: str = "main"
    bind_host: str = "127.0.0.1"
    port: int = Field(default=8190, ge=1, le=65535)
    attestation_cache_seconds: float = Field(default=60.0, ge=0, le=3600)
    token_env: str | None = "ARTIFEX_RENDER_NODE_TOKEN"
    require_token: bool = True
    comfyui_process: ManagedComfyConfig = Field(default_factory=ManagedComfyConfig)
    gateway: RendererGatewayConfig = Field(default_factory=RendererGatewayConfig)
    asset_paths: dict[str, Path] = Field(default_factory=dict)
    lora_roots: tuple[Path, ...] = ()
    extensions: tuple[str, ...] = (".safetensors",)
    metadata_header_max_mib: int = Field(default=16, ge=1)


class ComfyUiConfig(StrictModel):
    base_url: str = "http://127.0.0.1:8188"
    # Shared by ALL Artifex processes on PC-A; use the same path in CLI and daemon.
    submission_fence_path: Path = Path("data/comfy-submission-fence.sqlite3")
    # PC-A: optional authenticated renderer gateway, token is read from env.
    gateway_token_env: str | None = None
    output_mode: Literal["filesystem", "api"] = "filesystem"
    output_dir: Path | None = None
    download_dir: Path = Path("data/render-cache")
    render_node_id: str = "legacy"
    timeout_seconds: float = Field(default=30.0, gt=0)
    execution_timeout_seconds: float = Field(default=900.0, gt=0)
    poll_interval_seconds: float = Field(default=1.0, gt=0)
    request_attempts: int = Field(default=3, ge=1)
    reconnect_backoff_seconds: float = Field(default=1.0, ge=0)
    default_template: str = "illust_main_v1"
    refiner_checkpoint: str = "anime-refiner-beta1.1.safetensors"
    vae: str = "pppanimixVAE_ilxl.safetensors"
    upscale_model: str = "4xRealisticrescaler_100000G.pt"
    base_steps: int = Field(default=48, ge=1, le=200)
    base_cfg: float = Field(default=5.5, ge=0, le=30)
    base_sampler: str = "dpmpp_3m_sde_gpu"
    base_scheduler: str = "karras"
    base_denoise: float = Field(default=1.0, ge=0, le=1)
    refiner_steps: int = Field(default=24, ge=1, le=200)
    refiner_cfg: float = Field(default=5.0, ge=0, le=30)
    refiner_sampler: str = "dpmpp_3m_sde_gpu"
    refiner_scheduler: str = "karras"
    refiner_denoise: float = Field(default=0.10, ge=0, le=1)
    upscale_steps: int = Field(default=15, ge=1, le=200)
    upscale_cfg: float = Field(default=5.5, ge=0, le=30)
    upscale_sampler: str = "euler_ancestral"
    upscale_scheduler: str = "karras"
    upscale_denoise: float = Field(default=0.50, ge=0, le=1)
    release_vram_after_attempt: bool = True
    release_vram_on_error: bool = True


class EvaluationWeightsConfig(StrictModel):
    identity: float = Field(default=0.25, ge=0)
    alignment: float = Field(default=0.15, ge=0)
    face_quality: float = Field(default=0.15, ge=0)
    technical_quality: float = Field(default=0.15, ge=0)
    aesthetic: float = Field(default=0.10, ge=0)
    novelty: float = Field(default=0.08, ge=0)
    continuity: float = Field(default=0.07, ge=0)
    integrity: float = Field(default=0.05, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> EvaluationWeightsConfig:
        total = sum(
            (
                self.identity,
                self.alignment,
                self.face_quality,
                self.technical_quality,
                self.aesthetic,
                self.novelty,
                self.continuity,
                self.integrity,
            )
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("evaluation weights must sum to 1.0")
        return self


class EvaluationConfig(StrictModel):
    provider: Literal["openai_compatible_vision"] = "openai_compatible_vision"
    semantic_provider: Literal["siglip2", "local_fallback"] = "siglip2"
    semantic_model: str = "google/siglip2-base-patch16-224"
    semantic_revision: str | None = None
    semantic_device: Literal["cpu", "cuda", "auto"] = "cpu"
    semantic_cache_dir: Path | None = None
    semantic_local_files_only: bool = False
    allow_degraded_semantic: bool = False
    semantic_calibration_profile: str = "siglip2-hololive-ilxl-v1"
    semantic_calibration_path: Path = Path(
        "data/calibration/siglip2-hololive-ilxl-v1.json"
    )
    require_semantic_calibration: bool = True
    identity_reference_required: bool = True
    identity_reference_limit_per_character: int = Field(default=8, ge=1, le=64)
    identity_reference_hard_min: float = Field(default=0.30, ge=0, le=1)
    identity_reference_accept_min: float = Field(default=0.40, ge=0, le=1)
    duplicate_reference_limit: int = Field(default=128, ge=1, le=2000)
    novelty_reference_limit: int = Field(default=192, ge=1, le=4000)
    global_diversity_reference_limit: int = Field(default=64, ge=0, le=1000)
    historical_reference_scan_limit: int = Field(default=2000, ge=21, le=50000)
    vision_base_url: str | None = None
    vision_model: str | None = None
    vision_api_key_env: str | None = None
    vision_timeout_seconds: float = Field(default=120.0, gt=0)
    vision_request_attempts: int = Field(default=2, ge=1)
    vision_retry_backoff_seconds: float = Field(default=1.0, ge=0)
    weights: EvaluationWeightsConfig = Field(default_factory=EvaluationWeightsConfig)
    identity_hard_min: float = Field(default=0.60, ge=0, le=1)
    integrity_hard_min: float = Field(default=0.80, ge=0, le=1)
    similarity_hard_max: float = Field(default=0.92, ge=0, le=1)
    accepted_score_min: float = Field(default=0.75, ge=0, le=1)
    review_score_min: float = Field(default=0.55, ge=0, le=1)
    identity_accept_min: float = Field(default=0.75, ge=0, le=1)
    alignment_review_min: float = Field(default=0.60, ge=0, le=1)
    face_review_min: float = Field(default=0.60, ge=0, le=1)
    technical_review_min: float = Field(default=0.60, ge=0, le=1)
    continuity_review_min: float = Field(default=0.55, ge=0, le=1)


class SeasonalEventConfig(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=200)
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    relevance: float = Field(default=1.0, ge=0, le=1)
    lead_days: int = Field(default=30, ge=0, le=120)
    trail_days: int = Field(default=3, ge=0, le=30)


class TrendConfig(StrictModel):
    enabled: bool = True
    refresh_interval_seconds: float = Field(default=1800.0, gt=0)
    default_ttl_hours: float = Field(default=24.0, gt=0)
    freshness_half_life_hours: float = Field(default=8.0, gt=0)
    max_summary_signals: int = Field(default=20, ge=1)
    minimum_effective_strength: float = Field(default=0.05, ge=0, le=1)
    max_provider_results: int = Field(default=8, ge=1, le=30)
    max_external_topic_chars: int = Field(default=180, ge=40, le=300)
    current_web_enabled: bool = True
    tag_trends_enabled: bool = True
    current_queries: tuple[str, ...] = (
        "hololive fanart illustration trends",
        "hololive fanart",
        "vtuber anime illustration trends",
    )
    tag_queries: tuple[str, ...] = ("hololive%", "virtual_youtuber%")
    seasonal_enabled: bool = True
    max_seasonal_events: int = Field(default=12, ge=1, le=50)
    custom_seasonal_events: tuple[SeasonalEventConfig, ...] = ()


class ResearchConfig(StrictModel):
    enabled: bool = True
    required_for_ideation: bool = True
    adult_enabled: bool = False
    default_region: str = "jp-jp"
    default_safesearch: Literal["on", "moderate", "off"] = "moderate"
    adult_safesearch: Literal["on", "moderate", "off"] = "off"
    max_queries_per_cycle: int = Field(default=4, ge=1, le=20)
    max_results_per_query: int = Field(default=8, ge=1, le=50)
    max_brief_items: int = Field(default=20, ge=1, le=100)
    max_brief_chars: int = Field(default=12000, ge=1000, le=100000)
    max_snippet_chars: int = Field(default=1200, ge=100, le=10000)
    max_cycle_seconds: float = Field(default=45.0, gt=0, le=600)
    daily_request_budget: int = Field(default=200, ge=1, le=10000)
    max_extract_chars: int = Field(default=8000, ge=500, le=100000)
    cache_ttl_hours: float = Field(default=6.0, gt=0)
    current_cache_ttl_hours: float = Field(default=2.0, gt=0)
    evergreen_cache_ttl_hours: float = Field(default=72.0, gt=0)
    timeout_seconds: int = Field(default=10, ge=1, le=120)
    provider_order: tuple[str, ...] = ("ddgs", "searxng", "gelbooru")
    ddgs_backend: str = "auto"
    searxng_base_url: str | None = None
    gelbooru_enabled: bool = True
    gelbooru_base_url: str = "https://gelbooru.com"
    gelbooru_api_key_env: str | None = None
    gelbooru_user_id_env: str | None = None
    allow_private_network: bool = False
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()


class DiscordConfig(StrictModel):
    enabled: bool = False
    token_env: str = "ARTIFEX_DISCORD_TOKEN"
    guild_id: int | None = Field(default=None, ge=1)
    channel_id: int | None = Field(default=None, ge=1)
    allowed_user_ids: tuple[int, ...] = ()
    allowed_role_ids: tuple[int, ...] = ()
    command_prefix: str = "!artifex "
    notify_completion: bool = True
    notify_review: bool = True
    notify_error: bool = True
    notify_backend: bool = True
    notify_disk: bool = True
    daily_summary_hour_local: int = Field(default=9, ge=0, le=23)
    daily_summary_timezone: str = "Asia/Tokyo"

    @model_validator(mode="after")
    def validate_authorization(self) -> DiscordConfig:
        if self.enabled and not (self.allowed_user_ids or self.allowed_role_ids):
            raise ValueError(
                "Discord requires allowed_user_ids or allowed_role_ids when enabled"
            )
        return self


class PerformanceMetricWeightsConfig(StrictModel):
    views: float = Field(default=0.05, ge=0)
    engagement: float = Field(default=0.20, ge=0)
    conversion: float = Field(default=0.25, ge=0)
    subscribers: float = Field(default=0.15, ge=0)
    revenue: float = Field(default=0.20, ge=0)
    retention: float = Field(default=0.15, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> PerformanceMetricWeightsConfig:
        total = sum(
            (
                self.views,
                self.engagement,
                self.conversion,
                self.subscribers,
                self.revenue,
                self.retention,
            )
        )
        if total <= 0:
            raise ValueError("performance metric weights must have positive total")
        return self


class PatreonPerformanceConfig(StrictModel):
    enabled: bool = True
    half_life_days: float = Field(default=45.0, gt=0)
    prior_score: float = Field(default=0.5, ge=0, le=1)
    confidence_sample_scale: float = Field(default=250.0, gt=0)
    view_scale: float = Field(default=2500.0, gt=0)
    engagement_rate_scale: float = Field(default=0.10, gt=0)
    conversion_rate_scale: float = Field(default=0.05, gt=0)
    subscriber_scale: float = Field(default=25.0, gt=0)
    revenue_scale_cents: float = Field(default=10000.0, gt=0)
    minimum_confidence_for_editorial: float = Field(default=0.20, ge=0, le=1)
    editorial_effect_cap: float = Field(default=0.15, ge=0, le=0.5)
    metric_weights: PerformanceMetricWeightsConfig = Field(
        default_factory=PerformanceMetricWeightsConfig
    )


class PatreonConfig(StrictModel):
    enabled: bool = True
    api_enabled: bool = False
    api_base_url: str = "https://www.patreon.com"
    api_access_token_env: str = "PATREON_ACCESS_TOKEN"
    api_timeout_seconds: float = Field(default=30.0, gt=0)
    api_user_agent: str = "Artifex - Patreon Performance Sync"
    default_archetype: Literal[
        "public_preview_member_continuation",
        "sfw_complete_member_alternate",
        "public_only",
        "member_only",
    ] = "public_preview_member_continuation"
    generate_post_package: bool = True
    default_tags: tuple[str, ...] = ("illustration", "hololive")
    include_character_tags: bool = True
    performance: PatreonPerformanceConfig = Field(
        default_factory=PatreonPerformanceConfig
    )


class RightsConfig(StrictModel):
    enforce: bool = True
    default_profile: str = "unconfigured"
    platform_profile: str | None = None
    profile_dirs: tuple[Path, ...] = (Path("profiles/policies"),)


class StorageConfig(StrictModel):
    database_url: str = "sqlite:///data/artifex.sqlite3"
    packs_dir: Path = Path("data/packs")
    minimum_free_gib: float = Field(default=5.0, ge=0)


class OperationsConfig(StrictModel):
    health_interval_seconds: float = Field(default=30.0, gt=0)
    backend_failure_threshold: int = Field(default=3, ge=1)
    heartbeat_interval_seconds: float = Field(default=30.0, gt=0)
    heartbeat_stale_seconds: float = Field(default=120.0, gt=0)
    sustained_run_cycles: int = Field(default=5, ge=1)
    max_scenes_per_pack: int = Field(default=12, ge=1)
    max_attempts_per_scene: int = Field(default=4, ge=1)
    max_events_per_cycle: int = Field(default=50, ge=1)
    event_retention_rows: int = Field(default=10000, ge=100)


class QualificationConfig(StrictModel):
    evidence_dir: Path = Path("data/qualification")
    minimum_soak_hours: float = Field(default=8.0, gt=0)
    asset_paths: dict[str, Path] = Field(default_factory=dict)
    required_asset_labels: tuple[str, ...] = (
        "production_checkpoint",
        "refiner_checkpoint",
        "vae",
        "upscale_model",
        "llm_model",
        "semantic_model",
    )
    require_native_windows: bool = True
    require_uv: bool = True
    require_nvidia_gpu: bool = True
    require_lora_validation_evidence: bool = True
    require_stable_asset_hashes: bool = True
    require_stable_workflow_snapshot: bool = True


class ArtifexSettings(StrictModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)
    editorial: EditorialConfig = Field(default_factory=EditorialConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    characters: CharacterRegistryConfig = Field(default_factory=CharacterRegistryConfig)
    loras: LoRARegistryConfig = Field(default_factory=LoRARegistryConfig)
    prompts: PromptCompilerConfig = Field(default_factory=PromptCompilerConfig)
    render_nodes: RenderNodesConfig = Field(default_factory=RenderNodesConfig)
    render_agent: RenderAgentConfig = Field(default_factory=RenderAgentConfig)
    comfyui: ComfyUiConfig = Field(default_factory=ComfyUiConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    trends: TrendConfig = Field(default_factory=TrendConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    patreon: PatreonConfig = Field(default_factory=PatreonConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    rights: RightsConfig = Field(default_factory=RightsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    operations: OperationsConfig = Field(default_factory=OperationsConfig)
    qualification: QualificationConfig = Field(default_factory=QualificationConfig)

    @model_validator(mode="after")
    def validate_context_sections(self) -> ArtifexSettings:
        sections = (
            self.context.character_tokens
            + self.context.research_tokens
            + self.context.recent_history_tokens
            + self.context.long_term_tokens
            + self.context.trend_tokens
            + self.context.evergreen_tokens
            + self.context.operator_tokens
        )
        if sections >= self.llm.max_input_tokens:
            raise ValueError(
                "context section budgets must leave headroom for system/schema overhead"
            )

        primary = self.render_nodes.primary_node()
        if primary is not None:
            node_id, node = primary
            updates: dict[str, object] = {
                "base_url": node.base_url,
                "output_mode": node.output_mode,
                "download_dir": node.download_dir,
                "render_node_id": node_id,
            }
            if node.output_dir is not None:
                updates["output_dir"] = node.output_dir
            self.comfyui = self.comfyui.model_copy(update=updates)
        return self
