from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import func, select

from artifex.archive import PackArchive
from artifex.characters import CharacterRegistry
from artifex.config.models import (
    AgentConfig,
    CharacterRegistryConfig,
    LoRARegistryConfig,
    OperationsConfig,
    ProductionConfig,
)
from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow, ReviewQueueRow, SceneRow
from artifex.domain import (
    AgentState,
    CharacterProfile,
    LoRAPolicy,
    LoRAProfile,
    LoRAState,
    PackState,
)
from artifex.evaluation import (
    AttemptSelector,
    EvaluationEngine,
    EvaluationRepository,
    GenerationAttemptRepository,
    RawEvaluationSignals,
)
from artifex.loras import LoRARegistry, LoRAResolver
from artifex.operations.recovery import RecoveryManager
from artifex.packs import (
    ContentPackPlan,
    PackRepository,
    ScenePlan,
    VisualSpecification,
)
from artifex.policy import (
    PolicyApplicationService,
    PolicyDecisionRepository,
    PolicyEngine,
    PolicyOutcome,
    PolicyProfile,
    PolicyRegistry,
)
from artifex.production import GeneratedBatch, GenerationRequest, ProductionCoordinator
from artifex.prompts import ILXLDanbooruAdapter, PromptCompiler
from artifex.retry import RetryPolicy
from artifex.review import ReviewQueueRepository, ReviewState
from artifex.runtime import RuntimeDaemon, RuntimeStore
from artifex.scheduler import Scheduler, SchedulerAction
from artifex.series import SeriesRepository
from artifex.telemetry import TelemetryRepository


class AcceptedProvider:
    async def evaluate(self, context):
        del context
        return RawEvaluationSignals(
            identity=0.98,
            alignment=0.98,
            face_quality=0.98,
            technical_quality=0.98,
            aesthetic=0.98,
            continuity=0.98,
            integrity=0.99,
        )


class NoSimilarity:
    async def max_image_similarity(self, path, references):
        del path, references
        return 0.0


class SequenceProvider:
    def __init__(self, signals: list[RawEvaluationSignals]) -> None:
        self.signals = signals
        self.calls = 0

    async def evaluate(self, context):
        del context
        signal = self.signals[min(self.calls, len(self.signals) - 1)]
        self.calls += 1
        return signal


class SequenceSimilarity:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.calls = 0

    async def max_image_similarity(self, path, references):
        del path, references
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


class FakeBackend:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0
        self.requests: list[GenerationRequest] = []

    def provenance(self):
        return {
            "backend": "fake-comfy",
            "workflow_template": "ilxl_base_v1",
            "workflow_version": 1,
            "checkpoint": "qualification.safetensors",
            "width": 512,
            "height": 768,
            "batch_size": 1,
        }

    async def generate(self, request: GenerationRequest, *, on_submitted):
        self.calls += 1
        self.requests.append(request)
        prompt_id = f"prompt-{self.calls}"
        on_submitted(prompt_id)
        path = self.root / f"{request.scene_id}-{request.attempt_id}.png"
        Image.new(
            "RGB",
            (32, 32),
            (self.calls * 23 % 255, self.calls * 47 % 255, self.calls * 71 % 255),
        ).save(path)
        return GeneratedBatch(
            prompt_id=prompt_id,
            output_paths=(path,),
            outputs=(
                {
                    "node_id": "7",
                    "filename": path.name,
                    "subfolder": "",
                    "output_type": "output",
                },
            ),
        )


class FakeRecoveryComfy:
    async def get_history(self, prompt_id):
        del prompt_id

    async def queue_snapshot(self):
        return {"queue_running": [], "queue_pending": []}


def _visual() -> VisualSpecification:
    return VisualSpecification(
        composition="full body",
        camera="eye level",
        pose="standing",
        expression="smile",
        clothing="character outfit",
        setting="bright room",
        lighting="soft light",
        atmosphere="clean",
    )


def _plan(format_name: str, characters: tuple[str, ...]) -> ContentPackPlan:
    return ContentPackPlan(
        format=format_name,
        character_ids=characters,
        title=f"{format_name} qualification",
        logline="Qualification Pack.",
        continuity_bible=("consistent character identity",),
        scenes=(
            ScenePlan(
                ordinal=1,
                title="Scene 1",
                purpose="qualification",
                character_ids=characters,
                visual=_visual(),
                publication_tier="public",
            ),
        ),
    )


def _allow_policy() -> PolicyProfile:
    return PolicyProfile(
        id="allow",
        version="1",
        use_rules={
            "private": PolicyOutcome.ALLOW,
            "public_free": PolicyOutcome.ALLOW,
            "paid_membership": PolicyOutcome.ALLOW,
            "commercial": PolicyOutcome.ALLOW,
        },
        tier_rules={
            "public": PolicyOutcome.ALLOW,
            "member": PolicyOutcome.ALLOW,
            "private_review": PolicyOutcome.ALLOW,
            "blocked": PolicyOutcome.BLOCK,
        },
    )


def _qualification(
    tmp_path: Path,
    *,
    provider=None,
    similarity=None,
):
    database = Database(f"sqlite:///{(tmp_path / 'qualification.sqlite3').as_posix()}")
    database.migrate()
    runtime = RuntimeStore(database)
    runtime.set_agent_state(AgentState.STARTING, expected=AgentState.STOPPED)
    runtime.set_agent_state(AgentState.RUNNING, expected=AgentState.STARTING)

    production = ProductionConfig(
        idea_inventory_target=0,
        planned_inventory_target=0,
        completed_inventory_target=None,
        checkpoint="qualification.safetensors",
        width=512,
        height=768,
        batch_size=1,
    )
    operations = OperationsConfig(
        max_attempts_per_scene=4,
        event_retention_rows=1000,
    )
    scheduler = Scheduler(database, runtime, production)
    characters = CharacterRegistry(database)
    loras = LoRARegistry(database)

    for index in range(1, 4):
        character_id = f"char-{index}"
        lora_id = f"lora-{index}"
        characters.upsert(
            CharacterProfile(
                id=character_id,
                display_name=f"Character {index}",
                namespace="qualification",
                canonical_tags=(character_id,),
                model_families=("ilxl",),
                lora_policy=LoRAPolicy.REQUIRED,
                preferred_lora_ids=(lora_id,),
                readiness=1.0,
                policy_profile="allow",
            )
        )
        loras.upsert(
            LoRAProfile(
                id=lora_id,
                path=tmp_path / f"{lora_id}.safetensors",
                state=LoRAState.PRODUCTION,
                target_character_ids=(character_id,),
                model_families=("ilxl",),
                trigger_tags=(f"trigger_{index}",),
                recommended_weight=0.8,
                identity_score=0.95,
                quality_score=0.95,
                flexibility_score=0.95,
                readiness=0.95,
            )
        )
        if index == 1:
            loras.upsert(
                LoRAProfile(
                    id="lora-1-alt",
                    path=tmp_path / "lora-1-alt.safetensors",
                    state=LoRAState.PRODUCTION,
                    target_character_ids=(character_id,),
                    model_families=("ilxl",),
                    trigger_tags=("trigger_1_alt",),
                    recommended_weight=0.7,
                    validated_min_weight=0.6,
                    validated_max_weight=0.9,
                    identity_score=0.90,
                    quality_score=0.92,
                    flexibility_score=0.95,
                    readiness=0.90,
                )
            )

    policy_registry = PolicyRegistry()
    policy_registry.register(_allow_policy())
    policy_decisions = PolicyDecisionRepository(database)
    policy = PolicyApplicationService(
        database,
        PolicyEngine(
            policy_registry,
            characters,
            policy_decisions,
            __import__("artifex.config.models", fromlist=["RightsConfig"]).RightsConfig(
                default_profile="allow"
            ),
        ),
    )

    packs = PackRepository(database)
    attempts = GenerationAttemptRepository(database)
    evaluations = EvaluationRepository(database)
    telemetry = TelemetryRepository(database)
    reviews = ReviewQueueRepository(database)
    backend = FakeBackend(tmp_path)
    evaluator = EvaluationEngine(
        provider or AcceptedProvider(),  # type: ignore[arg-type]
        similarity or NoSimilarity(),  # type: ignore[arg-type]
        __import__("artifex.config.models", fromlist=["EvaluationConfig"]).EvaluationConfig(
            identity_reference_required=False
        ),
    )
    selector = AttemptSelector(database, runtime, evaluations)
    recovery = RecoveryManager(
        database,
        runtime,
        attempts,
        FakeRecoveryComfy(),  # type: ignore[arg-type]
        telemetry,
    )
    coordinator = ProductionCoordinator(
        database,
        runtime,
        scheduler,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        packs,
        policy,
        LoRAResolver(
            characters,
            loras,
            CharacterRegistryConfig(),
            LoRARegistryConfig(),
        ),
        PromptCompiler(characters, (ILXLDanbooruAdapter(),)),
        backend,
        evaluator,
        attempts,
        evaluations,
        selector,
        RetryPolicy(production),
        reviews,
        recovery,
        PackArchive(database, tmp_path / "packs"),
        characters,
        SeriesRepository(database),
        telemetry,
        production,
        operations,
        seed_factory=iter(range(1000, 2000)).__next__,
    )
    return database, runtime, scheduler, packs, reviews, coordinator, backend


@pytest.mark.asyncio
async def test_single_duo_and_group_run_end_to_end(tmp_path: Path) -> None:
    database, _, _, packs, reviews, coordinator, backend = _qualification(tmp_path)

    records = (
        packs.create_planned_pack(_plan("single_feature", ("char-1",)), concept_id=None),
        packs.create_planned_pack(_plan("duo", ("char-1", "char-2")), concept_id=None),
        packs.create_planned_pack(
            _plan("group", ("char-1", "char-2", "char-3")),
            concept_id=None,
        ),
    )

    for record in records:
        await coordinator.run_pack(record.pack_id)
        finalized = packs.require(record.pack_id)
        assert finalized.state is PackState.FINALIZED
        with database.session() as session:
            row = session.get(PackRow, record.pack_id)
            assert row is not None
            assert row.checkpoint_json["archive_complete"] is True
            assert Path(row.payload_json["archive"]["manifest_path"]).exists()
            scene = session.scalar(
                select(SceneRow).where(SceneRow.pack_id == record.pack_id)
            )
            assert scene is not None
            assert scene.payload_json["policy_phase"] == "post_generation"
            assert scene.payload_json["content_rating"] == "general"

    assert backend.calls == 3
    assert reviews.list_open() == ()
    database.dispose()


@pytest.mark.asyncio
async def test_daemon_sustained_run_processes_next_pack_without_operator(
    tmp_path: Path,
) -> None:
    database, runtime, scheduler, packs, _, coordinator, _ = _qualification(tmp_path)

    formats = (
        ("single_feature", ("char-1",)),
        ("duo", ("char-1", "char-2")),
        ("group", ("char-1", "char-2", "char-3")),
        ("single_feature", ("char-2",)),
        ("duo", ("char-2", "char-3")),
    )
    records = [
        packs.create_planned_pack(_plan(format_name, characters), concept_id=None)
        for format_name, characters in formats
    ]
    daemon = RuntimeDaemon(
        runtime,
        scheduler,
        coordinator,
        AgentConfig(poll_interval_seconds=0.001),
    )

    for _ in records:
        decision = await daemon.run_once()
        assert decision.action is SchedulerAction.RUN_PACK

    with database.session() as session:
        final_count = session.scalar(
            select(func.count())
            .select_from(PackRow)
            .where(PackRow.state == PackState.FINALIZED.value)
        )
        nonterminal_count = session.scalar(
            select(func.count())
            .select_from(PackRow)
            .where(
                PackRow.state.not_in(
                    (
                        PackState.FINALIZED.value,
                        PackState.FAILED.value,
                        PackState.BLOCKED.value,
                    )
                )
            )
        )
        attempt_count = session.scalar(
            select(func.count()).select_from(GenerationAttemptRow)
        )
        open_reviews = session.scalar(
            select(func.count())
            .select_from(ReviewQueueRow)
            .where(ReviewQueueRow.state == ReviewState.OPEN.value)
        )
        scene_count = session.scalar(select(func.count()).select_from(SceneRow))

    assert final_count == len(records)
    assert nonterminal_count == 0
    assert attempt_count == scene_count == len(records)
    assert open_reviews == 0
    database.dispose()



@pytest.mark.asyncio
async def test_identity_failure_executes_lora_weight_and_prompt_remediation(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider(
        [
            RawEvaluationSignals(
                identity=0.20,
                alignment=0.98,
                face_quality=0.98,
                technical_quality=0.98,
                aesthetic=0.98,
                continuity=0.98,
                integrity=0.99,
            ),
            RawEvaluationSignals(
                identity=0.98,
                alignment=0.98,
                face_quality=0.98,
                technical_quality=0.98,
                aesthetic=0.98,
                continuity=0.98,
                integrity=0.99,
            ),
        ]
    )
    database, _, _, packs, reviews, coordinator, backend = _qualification(
        tmp_path,
        provider=provider,
    )
    record = packs.create_planned_pack(
        _plan("single_feature", ("char-1",)),
        concept_id=None,
    )

    await coordinator.run_pack(record.pack_id)

    assert packs.require(record.pack_id).state is PackState.FINALIZED
    assert backend.calls == 2
    first, second = backend.requests
    assert first.lora_plan.entries[0].lora_id == "lora-1"
    assert second.lora_plan.entries[0].lora_id == "lora-1-alt"
    assert second.lora_plan.entries[0].weight > 0.7
    assert second.compiled.positive_prompt != first.compiled.positive_prompt
    with database.session() as session:
        scene = session.scalar(
            select(SceneRow).where(SceneRow.pack_id == record.pack_id)
        )
        assert scene is not None
        history = scene.payload_json["retry_history"]
        assert len(history) == 1
        assert history[0]["applied_actions"] == [
            "alternate_lora",
            "adjust_lora_weight",
            "revise_prompt",
        ]
        assert history[0]["before_digest"] != history[0]["after_digest"]
        attempts = session.scalars(
            select(GenerationAttemptRow)
            .where(GenerationAttemptRow.scene_id == scene.id)
            .order_by(GenerationAttemptRow.ordinal.asc())
        ).all()
        assert attempts[1].parent_attempt_id == attempts[0].id
        retry_provenance = attempts[1].provenance_json["retry_execution"]
        assert retry_provenance["before_digest"] == history[0]["before_digest"]
        assert retry_provenance["after_digest"] == history[0]["after_digest"]
    assert reviews.list_open() == ()
    database.dispose()


@pytest.mark.asyncio
async def test_technical_failure_routes_retry_through_repair_workflow(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider(
        [
            RawEvaluationSignals(
                identity=0.98,
                alignment=0.98,
                face_quality=0.98,
                technical_quality=0.20,
                aesthetic=0.98,
                continuity=0.98,
                integrity=0.99,
            ),
            RawEvaluationSignals(
                identity=0.98,
                alignment=0.98,
                face_quality=0.98,
                technical_quality=0.98,
                aesthetic=0.98,
                continuity=0.98,
                integrity=0.99,
            ),
        ]
    )
    database, _, _, packs, _, coordinator, backend = _qualification(
        tmp_path,
        provider=provider,
    )
    record = packs.create_planned_pack(
        _plan("single_feature", ("char-1",)),
        concept_id=None,
    )

    await coordinator.run_pack(record.pack_id)

    assert backend.calls == 2
    first, second = backend.requests
    assert first.seed != second.seed
    assert first.workflow_template_id is None
    assert second.workflow_template_id == "illust_main_repair_v1"
    with database.session() as session:
        attempts = session.scalars(
            select(GenerationAttemptRow)
            .order_by(GenerationAttemptRow.ordinal.asc())
        ).all()
        assert attempts[-1].provenance_json["workflow_template"] == "illust_main_repair_v1"
    database.dispose()


@pytest.mark.asyncio
async def test_similarity_failure_varies_scene_semantics_before_retry(
    tmp_path: Path,
) -> None:
    database, _, _, packs, _, coordinator, backend = _qualification(
        tmp_path,
        similarity=SequenceSimilarity([0.99, 0.0]),
    )
    record = packs.create_planned_pack(
        _plan("single_feature", ("char-1",)),
        concept_id=None,
    )

    await coordinator.run_pack(record.pack_id)

    assert backend.calls == 2
    first, second = backend.requests
    assert first.seed != second.seed
    assert first.compiled.positive_prompt != second.compiled.positive_prompt
    with database.session() as session:
        scene = session.scalar(
            select(SceneRow).where(SceneRow.pack_id == record.pack_id)
        )
        assert scene is not None
        history = scene.payload_json["retry_history"]
        assert history[0]["applied_actions"] == ["vary_scene", "change_seed"]
        before_scene = history[0]["before_inputs"]["scene"]
        after_scene = history[0]["after_inputs"]["scene"]
        assert before_scene["visual"]["clothing"] == after_scene["visual"]["clothing"]
        assert before_scene["visual"]["setting"] == after_scene["visual"]["setting"]
        assert before_scene["visual"]["camera"] != after_scene["visual"]["camera"]
    database.dispose()
