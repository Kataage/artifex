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
from artifex.evaluation import (
    AttemptSelector,
    EvaluationEngine,
    EvaluationRepository,
    GenerationAttemptRepository,
    LocalSimilarityEmbeddingProvider,
    OpenAICompatibleVisionEvaluationProvider,
    SimilarityAwareSignalProvider,
    SimilarityService,
)
from artifex.llm import OpenAICompatibleClient, StructuredGenerator
from artifex.loras import LoRADiscovery, LoRARegistry, LoRAResolver
from artifex.operations import HealthChecker, HealthSupervisor
from artifex.operations.doctor import DoctorService
from artifex.operations.recovery import RecoveryManager
from artifex.packs import PackPlanner, PackRepository
from artifex.planner import ConceptRepository, IdeaDirector
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
from artifex.retry import RetryPolicy
from artifex.review import ReviewQueueRepository
from artifex.runtime import RuntimeDaemon, RuntimeStore
from artifex.scheduler import Scheduler
from artifex.series import SeriesRepository
from artifex.telemetry import TelemetryRepository
from artifex.trends import TrendPlannerContext, TrendRepository


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
    characters: CharacterRegistry
    loras: LoRARegistry
    reviews: ReviewQueueRepository
    series: SeriesRepository
    policy_decisions: PolicyDecisionRepository
    telemetry: TelemetryRepository
    research: ResearchService
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
    scheduler = Scheduler(database, runtime, settings.production)
    telemetry = TelemetryRepository(database)
    research_providers = [DDGSResearchProvider(settings.research)]
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
    comfy = ComfyUIClient(settings.comfyui)

    return CoreServices(
        settings=settings,
        database=database,
        runtime=runtime,
        scheduler=scheduler,
        characters=characters,
        loras=loras,
        reviews=ReviewQueueRepository(database),
        series=SeriesRepository(database),
        policy_decisions=PolicyDecisionRepository(database),
        telemetry=telemetry,
        research=research,
        comfy=comfy,
    )


def build_doctor(core: CoreServices) -> DoctorService:
    checker = HealthChecker(
        core.settings,
        core.database,
        core.telemetry,
        comfy=core.comfy,
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


def build_application(settings: ArtifexSettings) -> ArtifexApplication:
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

    local_embeddings = LocalSimilarityEmbeddingProvider()
    similarity = SimilarityService(local_embeddings)

    llm = OpenAICompatibleClient(settings.llm)
    generator = StructuredGenerator(
        llm,
        repair_attempts=settings.llm.structured_repair_attempts,
    )
    concepts = ConceptRepository(core.database)
    trends = TrendPlannerContext(
        TrendRepository(core.database, settings.trends),
        settings.trends,
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
    research_director = ResearchDirector(core.research, settings.research)

    packs = PackRepository(core.database)
    pack_planner = PackPlanner(generator, packs)

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
    archive = PackArchive(core.database, settings.storage.packs_dir)

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
    )

    remote = ArtifexRemoteOperations(
        core.database,
        core.runtime,
        core.scheduler,
        core.reviews,
        core.characters,
        core.series,
        core.policy_decisions,
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
    )
    supervisor = HealthSupervisor(
        checker,
        core.runtime,
        core.telemetry,
        settings.operations,
        notifications=discord_runtime,
    )
    daemon = RuntimeDaemon(
        core.runtime,
        core.scheduler,
        coordinator,
        settings.agent,
        maintenance=supervisor,
    )
    return ArtifexApplication(
        core,
        daemon,
        llm,
        vision,
        discord=discord_runtime,
        discord_token=discord_token,
    )
