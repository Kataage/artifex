from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from artifex.archive import PackArchive
from artifex.characters import CharacterRegistry
from artifex.comfy import ComfyUIClient, WorkflowTemplateRegistry
from artifex.config.models import ArtifexSettings
from artifex.db import Database
from artifex.discord import (
    ArtifexRemoteOperations,
    AuthorizationPolicy,
    DailySummaryBuilder,
    DiscordCommandRouter,
)
from artifex.editorial import (
    EditorialRepository,
    EditorialService,
    PackInventoryRepository,
)
from artifex.evaluation import (
    AttemptSelector,
    EvaluationEngine,
    EvaluationRepository,
    GenerationAttemptRepository,
    LocalSimilarityEmbeddingProvider,
    OpenAICompatibleVisionEvaluationProvider,
    load_calibration_profile,
    SemanticEmbeddingRepository,
    SemanticIndex,
    SigLIP2EmbeddingProvider,
    SimilarityAwareSignalProvider,
    SimilarityService,
)
from artifex.llm import OpenAICompatibleClient, StructuredGenerator
from artifex.llm.provenance import LlmCallRepository
from artifex.loras import LoRADiscovery, LoRARegistry, LoRAResolver
from artifex.memory import ConceptMemoryRetriever, ContextMemoryManager
from artifex.operations import (
    HealthChecker,
    HealthSupervisor,
    MaintenanceGroup,
    SignalIngestionMaintenance,
)
from artifex.operations.doctor import DoctorService
from artifex.operations.recovery import RecoveryManager
from artifex.packs import PackPlanner, PackRepository
from artifex.planner import ConceptRepository, IdeaDirector, SeriesIdeaDirector
from artifex.planner.scoring import DefaultSignalProvider
from artifex.policy import (
    PolicyApplicationService,
    PolicyDecisionRepository,
    PolicyEngine,
    PolicyRegistry,
)
from artifex.production import (
    ComfyGenerationBackend,
    PlanningContextBuilder,
    ProductionCoordinator,
)
from artifex.prompts import ILXLDanbooruAdapter, PromptCompiler
from artifex.research import (
    DDGSResearchProvider,
    GelbooruMetadataProvider,
    ResearchDirector,
    ResearchRepository,
    ResearchRouter,
    ResearchService,
    SearXNGResearchProvider,
)
from artifex.research.provider import ResearchProvider
from artifex.retry import RetryPolicy
from artifex.review import ReviewQueueRepository
from artifex.runtime import RuntimeDaemon, RuntimeStore
from artifex.scheduler import Scheduler
from artifex.series import SeriesRepository
from artifex.telemetry import TelemetryRepository
from artifex.trends import (
    SeasonalCalendarProvider,
    SeasonalCollector,
    SeasonalProvider,
    SeasonalRepository,
    SignalHealthRepository,
    SignalIngestionService,
    TrendCollector,
    TrendPlannerContext,
    TrendProvider,
    TrendRepository,
    current_web_trend_provider,
    tag_trend_provider,
)


class DiscordRuntime(Protocol):
    def is_ready(self) -> bool: ...

    async def start(self, token: str) -> None: ...

    async def close(self) -> None: ...

    async def notify(self, notification: Any) -> bool: ...


@dataclass(slots=True)
class CoreServices:
    settings: ArtifexSettings
    database: Database
    runtime: RuntimeStore
    scheduler: Scheduler
    editorial: EditorialService
    characters: CharacterRegistry
    loras: LoRARegistry
    reviews: ReviewQueueRepository
    series: SeriesRepository
    policy_decisions: PolicyDecisionRepository
    telemetry: TelemetryRepository
    research: ResearchService
    trends: TrendRepository
    seasonal: SeasonalRepository
    signals: SignalIngestionService
    comfy: ComfyUIClient

    async def close(self) -> None:
        await self.research.aclose()
        await self.comfy.aclose()
        self.database.dispose()


def build_core(settings: ArtifexSettings) -> CoreServices:
    database = Database(settings.storage.database_url)
    database.migrate()

    characters = CharacterRegistry(database)
    characters.load_directories(settings.characters.profile_dirs)

    loras = LoRARegistry(database)
    LoRADiscovery(
        loras,
        characters,
        extensions=settings.loras.extensions,
        max_header_bytes=settings.loras.metadata_header_max_mib * 1024 * 1024,
    ).scan(settings.loras.roots)

    runtime = RuntimeStore(database)
    series = SeriesRepository(
        database,
        rolling_summary_max_chars=settings.context.series_tokens,
    )
    inventory = PackInventoryRepository(
        database,
        expiry_hours=settings.editorial.inventory_expiry_hours,
    )
    editorial = EditorialService(
        database,
        settings.editorial,
        series,
        inventory,
        EditorialRepository(database),
    )
    scheduler = Scheduler(
        database,
        runtime,
        settings.production,
        editorial=editorial,
    )
    telemetry = TelemetryRepository(database)
    research_providers: list[ResearchProvider] = [
        DDGSResearchProvider(settings.research)
    ]
    if settings.research.searxng_base_url:
        research_providers.append(SearXNGResearchProvider(settings.research))
    if settings.research.gelbooru_enabled:
        research_providers.append(GelbooruMetadataProvider(settings.research))
    research = ResearchService(
        settings.research,
        ResearchRepository(database),
        ResearchRouter(
            research_providers,
            provider_order=settings.research.provider_order,
        ),
    )
    trend_repository = TrendRepository(database, settings.trends)
    seasonal_repository = SeasonalRepository(database, settings.trends)
    trend_providers: list[TrendProvider] = []
    if settings.trends.current_web_enabled:
        trend_providers.append(
            current_web_trend_provider(
                research,
                settings.trends,
                region=settings.research.default_region,
                safesearch=settings.research.default_safesearch,
            )
        )
    if (
        settings.trends.tag_trends_enabled
        and settings.research.gelbooru_enabled
    ):
        trend_providers.append(
            tag_trend_provider(
                research,
                settings.trends,
                region=settings.research.default_region,
                safesearch=settings.research.default_safesearch,
            )
        )
    seasonal_providers: tuple[SeasonalProvider, ...] = (
        (SeasonalCalendarProvider(settings.trends),)
        if settings.trends.seasonal_enabled
        else ()
    )
    signals = SignalIngestionService(
        TrendCollector(trend_providers, trend_repository, settings.trends),
        SeasonalCollector(seasonal_providers, seasonal_repository),
        trend_repository,
        seasonal_repository,
        SignalHealthRepository(database),
        telemetry,
    )
    comfy = ComfyUIClient(settings.comfyui)

    return CoreServices(
        settings=settings,
        database=database,
        runtime=runtime,
        scheduler=scheduler,
        editorial=editorial,
        characters=characters,
        loras=loras,
        reviews=ReviewQueueRepository(database),
        series=series,
        policy_decisions=PolicyDecisionRepository(database),
        telemetry=telemetry,
        research=research,
        trends=trend_repository,
        seasonal=seasonal_repository,
        signals=signals,
        comfy=comfy,
    )


def build_doctor(core: CoreServices) -> DoctorService:
    checker = HealthChecker(
        core.settings,
        core.database,
        core.telemetry,
        comfy=core.comfy,
        research_probe=core.research.health,
    )
    return DoctorService(core.settings, checker, core.characters)


class ArtifexApplication:
    def __init__(
        self,
        core: CoreServices,
        daemon: RuntimeDaemon,
        llm: OpenAICompatibleClient,
        vision: OpenAICompatibleVisionEvaluationProvider,
        *,
        discord: DiscordRuntime | None = None,
        discord_token: str | None = None,
    ) -> None:
        self._core = core
        self._daemon = daemon
        self._llm = llm
        self._vision = vision
        self._discord = discord
        self._discord_token = discord_token

    async def run(self) -> None:
        try:
            if self._discord is None:
                await self._daemon.run_forever()
                return
            if self._discord_token is None:
                raise RuntimeError("Discord runtime is missing its token")
            async with asyncio.TaskGroup() as group:
                group.create_task(self._daemon.run_forever())
                group.create_task(self._discord.start(self._discord_token))
        finally:
            await self.close()

    async def close(self) -> None:
        self._daemon.request_stop()
        if self._discord is not None:
            await self._discord.close()
        await self._vision.aclose()
        await self._llm.aclose()
        await self._core.close()


def _apply_semantic_calibration(settings: ArtifexSettings) -> ArtifexSettings:
    evaluation = settings.evaluation
    if evaluation.semantic_provider != "siglip2":
        return settings

    path = evaluation.semantic_calibration_path
    if not path.is_file():
        if evaluation.require_semantic_calibration:
            raise ValueError(
                "production semantic calibration is missing: "
                f"{path}. Run 'artifex semantic calibrate' against a curated "
                "ILXL/Hololive regression corpus."
            )
        return settings

    profile = load_calibration_profile(path)
    if profile.profile_id != evaluation.semantic_calibration_profile:
        raise ValueError(
            "semantic calibration profile id mismatch: "
            f"{profile.profile_id} != {evaluation.semantic_calibration_profile}"
        )
    if profile.model != evaluation.semantic_model:
        raise ValueError(
            "semantic calibration model mismatch: "
            f"{profile.model} != {evaluation.semantic_model}"
        )
    if not profile.validated and evaluation.require_semantic_calibration:
        raise ValueError(
            f"semantic calibration profile is not validated: {path}"
        )
    if (
        evaluation.semantic_revision is not None
        and evaluation.semantic_revision != profile.revision
    ):
        raise ValueError(
            "semantic calibration revision mismatch: "
            f"{profile.revision} != {evaluation.semantic_revision}"
        )

    calibrated_evaluation = evaluation.model_copy(
        update={
            "semantic_revision": profile.revision,
            "identity_reference_hard_min": profile.identity_hard_min,
            "identity_reference_accept_min": profile.identity_accept_min,
            "similarity_hard_max": profile.similarity_hard_max,
        }
    )
    calibrated_planner = settings.planner.model_copy(
        update={
            "hard_similarity_threshold": profile.planner_hard_similarity_threshold,
        }
    )
    return settings.model_copy(
        update={
            "evaluation": calibrated_evaluation,
            "planner": calibrated_planner,
        }
    )


def build_application(settings: ArtifexSettings) -> ArtifexApplication:
    settings = _apply_semantic_calibration(settings)
    if not settings.production.checkpoint:
        raise ValueError("production.checkpoint must be configured")
    if settings.comfyui.output_dir is None:
        raise ValueError("comfyui.output_dir must be configured")
    if not settings.evaluation.vision_base_url or not settings.evaluation.vision_model:
        raise ValueError(
            "evaluation.vision_base_url and evaluation.vision_model must be configured"
        )

    core = build_core(settings)
    if not core.characters.list(enabled_only=True):
        core.database.dispose()
        raise ValueError("no enabled character profiles were loaded")

    semantic_embeddings: SigLIP2EmbeddingProvider | LocalSimilarityEmbeddingProvider
    if settings.evaluation.semantic_provider == "siglip2":
        semantic_embeddings = SigLIP2EmbeddingProvider(
            model=settings.evaluation.semantic_model,
            revision=settings.evaluation.semantic_revision,
            device=settings.evaluation.semantic_device,
            cache_dir=settings.evaluation.semantic_cache_dir,
            local_files_only=settings.evaluation.semantic_local_files_only,
        )
    else:
        if not settings.evaluation.allow_degraded_semantic:
            core.database.dispose()
            raise ValueError(
                "local_fallback semantic embeddings are degraded; "
                "set evaluation.allow_degraded_semantic=true only for diagnostics"
            )
        semantic_embeddings = LocalSimilarityEmbeddingProvider()

    semantic_index = SemanticIndex(
        SemanticEmbeddingRepository(core.database),
        semantic_embeddings,
    )
    similarity = SimilarityService(semantic_embeddings, semantic_index)

    llm = OpenAICompatibleClient(
        settings.llm,
        provenance=LlmCallRepository(core.database),
    )
    generator = StructuredGenerator(
        llm,
        repair_attempts=settings.llm.structured_repair_attempts,
    )
    concepts = ConceptRepository(core.database)
    trends = TrendPlannerContext(
        core.trends,
        settings.trends,
        seasonal=core.seasonal,
    )
    context_builder = PlanningContextBuilder(
        core.characters,
        core.loras,
        concepts,
        settings.characters,
        trends=trends,
    )
    idea_director = IdeaDirector(
        generator,
        concepts,
        settings.planner,
        signal_provider=SimilarityAwareSignalProvider(
            DefaultSignalProvider(),
            similarity,
        ),
        require_research=settings.research.required_for_ideation,
    )
    series_idea_director = SeriesIdeaDirector(
        generator,
        concepts,
        settings.planner,
        settings.editorial,
        signal_provider=SimilarityAwareSignalProvider(
            DefaultSignalProvider(),
            similarity,
        ),
        require_research=settings.research.required_for_ideation,
    )
    research_director = ResearchDirector(core.research, settings.research)
    context_memory = ContextMemoryManager(
        settings.context,
        ConceptMemoryRetriever(
            concepts,
            semantic_embeddings,
            index=semantic_index,
        ),
    )

    packs = PackRepository(core.database)
    pack_planner = PackPlanner(generator, packs, settings.context, settings.patreon)

    policy_registry = PolicyRegistry.with_packaged_defaults()
    policy_registry.load_directories(settings.rights.profile_dirs)
    policy_engine = PolicyEngine(
        policy_registry,
        core.characters,
        core.policy_decisions,
        settings.rights,
    )
    policy = PolicyApplicationService(core.database, policy_engine)

    lora_resolver = LoRAResolver(
        core.characters,
        core.loras,
        settings.characters,
        settings.loras,
    )
    prompts = PromptCompiler(core.characters, (ILXLDanbooruAdapter(),))
    templates = WorkflowTemplateRegistry.with_packaged_templates()
    backend = ComfyGenerationBackend(
        core.comfy,
        templates,
        settings.production,
        settings.comfyui,
    )

    vision = OpenAICompatibleVisionEvaluationProvider(settings.evaluation)
    evaluator = EvaluationEngine(vision, similarity, settings.evaluation)
    attempts = GenerationAttemptRepository(core.database)
    evaluations = EvaluationRepository(core.database)
    selector = AttemptSelector(core.database, core.runtime, evaluations)
    retry_policy = RetryPolicy(settings.production)
    recovery = RecoveryManager(
        core.database,
        core.runtime,
        attempts,
        core.comfy,
        core.telemetry,
    )
    archive = PackArchive(core.database, settings.storage.packs_dir, settings.patreon)

    coordinator = ProductionCoordinator(
        core.database,
        core.runtime,
        core.scheduler,
        idea_director,
        context_builder,
        concepts,
        pack_planner,
        packs,
        policy,
        lora_resolver,
        prompts,
        backend,
        evaluator,
        attempts,
        evaluations,
        selector,
        retry_policy,
        core.reviews,
        recovery,
        archive,
        core.characters,
        core.series,
        core.telemetry,
        settings.production,
        settings.operations,
        research=research_director,
        context_memory=context_memory,
        series_idea_director=series_idea_director,
        editorial=core.editorial,
    )

    remote = ArtifexRemoteOperations(
        core.database,
        core.runtime,
        core.scheduler,
        core.reviews,
        core.characters,
        core.series,
        core.policy_decisions,
        signals=core.signals,
    )
    router = DiscordCommandRouter(AuthorizationPolicy(settings.discord), remote)

    discord_runtime: DiscordRuntime | None = None
    discord_token: str | None = None
    if settings.discord.enabled:
        from artifex.discord.bot import ArtifexDiscordClient, token_from_environment

        discord_runtime = ArtifexDiscordClient(
            settings.discord,
            router,
            DailySummaryBuilder(core.database),
        )
        discord_token = token_from_environment(settings.discord)

    checker = HealthChecker(
        settings,
        core.database,
        core.telemetry,
        comfy=core.comfy,
        discord_connected=(
            discord_runtime.is_ready if discord_runtime is not None else None
        ),
        research_probe=core.research.health,
    )
    supervisor = HealthSupervisor(
        checker,
        core.runtime,
        core.telemetry,
        settings.operations,
        notifications=discord_runtime,
    )
    signal_maintenance = SignalIngestionMaintenance(
        core.signals,
        core.telemetry,
        interval_seconds=settings.trends.refresh_interval_seconds,
    )
    daemon = RuntimeDaemon(
        core.runtime,
        core.scheduler,
        coordinator,
        settings.agent,
        maintenance=MaintenanceGroup((supervisor, signal_maintenance)),
    )
    return ArtifexApplication(
        core,
        daemon,
        llm,
        vision,
        discord=discord_runtime,
        discord_token=discord_token,
    )
