from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

from sqlalchemy import select

from artifex.archive import PackArchive
from artifex.characters import CharacterRegistry
from artifex.comfy import ComfyUIError
from artifex.config.models import OperationsConfig, ProductionConfig
from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow, SceneRow
from artifex.domain import PackState, PublicationTier, ResultState, SceneState
from artifex.editorial import EditorialService, SeriesPlanKind
from artifex.evaluation import (
    AttemptSelector,
    EvaluationContext,
    EvaluationEngine,
    EvaluationRepository,
    EvaluationResult,
    GenerationAttemptRepository,
)
from artifex.loras import LoRAPlan, LoRAResolutionError, LoRAResolver
from artifex.memory import ContextMemoryManager
from artifex.operations.recovery import RecoveryManager, RecoveryState
from artifex.packs import PackPlanner, PackRepository, ScenePlan
from artifex.planner import ConceptRepository, IdeaDirector, SeriesIdeaDirector
from artifex.policy import PolicyApplicationService, PolicyOutcome, UseClass
from artifex.production.backend import GenerationBackend, GenerationRequest
from artifex.production.context import PlanningContextBuilder
from artifex.prompts import CompiledPrompt, PromptCompiler
from artifex.research import ResearchDirector
from artifex.retry import (
    RetryAction,
    RetryActionExecutor,
    RetryExecutionRecord,
    RetryPolicy,
    RetryProductionInputs,
)
from artifex.review import ReviewQueueRepository
from artifex.runtime import RuntimeStore
from artifex.scheduler import Scheduler
from artifex.series import SeriesRepository
from artifex.telemetry import EventSeverity, TelemetryRepository


def _seed() -> int:
    return secrets.randbits(63)


class ProductionCoordinator:
    """Persisted RuntimeHandler that connects planning through Pack finalization."""

    def __init__(
        self,
        database: Database,
        runtime: RuntimeStore,
        scheduler: Scheduler,
        idea_director: IdeaDirector,
        context_builder: PlanningContextBuilder,
        concepts: ConceptRepository,
        pack_planner: PackPlanner,
        packs: PackRepository,
        policy: PolicyApplicationService,
        loras: LoRAResolver,
        prompts: PromptCompiler,
        backend: GenerationBackend,
        evaluator: EvaluationEngine,
        attempts: GenerationAttemptRepository,
        evaluations: EvaluationRepository,
        selector: AttemptSelector,
        retries: RetryPolicy,
        reviews: ReviewQueueRepository,
        recovery: RecoveryManager,
        archive: PackArchive,
        characters: CharacterRegistry,
        series: SeriesRepository,
        telemetry: TelemetryRepository,
        production: ProductionConfig,
        operations: OperationsConfig,
        *,
        research: ResearchDirector | None = None,
        context_memory: ContextMemoryManager | None = None,
        series_idea_director: SeriesIdeaDirector | None = None,
        editorial: EditorialService | None = None,
        seed_factory: Callable[[], int] = _seed,
    ) -> None:
        if production.batch_size != 1:
            raise ValueError(
                "Initial Complete production coordinator requires batch_size=1 "
                "so each GenerationAttempt maps to one evaluated image"
            )
        self._database = database
        self._runtime = runtime
        self._scheduler = scheduler
        self._idea_director = idea_director
        self._context_builder = context_builder
        self._concepts = concepts
        self._pack_planner = pack_planner
        self._packs = packs
        self._policy = policy
        self._loras = loras
        self._prompts = prompts
        self._backend = backend
        self._evaluator = evaluator
        self._attempts = attempts
        self._evaluations = evaluations
        self._selector = selector
        self._retries = retries
        self._retry_executor = RetryActionExecutor(loras, prompts, production)
        self._reviews = reviews
        self._recovery = recovery
        self._archive = archive
        self._characters = characters
        self._series = series
        self._telemetry = telemetry
        self._production = production
        self._operations = operations
        self._research = research
        self._context_memory = context_memory
        self._series_idea_director = series_idea_director
        self._editorial = editorial
        self._seed_factory = seed_factory

    async def replenish_ideas(self) -> None:
        inventory = self._scheduler.inventory()
        missing = max(
            0,
            self._production.idea_inventory_target - inventory.ideas,
        )
        count = min(missing, self._production.idea_replenish_batch)
        if count == 0:
            return
        context = self._context_builder.build()
        if self._editorial is not None:
            context = self._editorial.apply_diversity(context)
        if self._context_memory is not None:
            context = self._context_memory.pre_research(context)
        if self._research is not None:
            context = await self._research.prepare(context)
            brief = context.research_brief
            self._telemetry.record(
                "research.ideation_ready",
                EventSeverity.INFO,
                {
                    "brief_id": brief.id if brief is not None else None,
                    "evidence_count": len(brief.evidence_ids) if brief is not None else 0,
                    "degraded": brief.degraded if brief is not None else True,
                },
            )
        if self._context_memory is not None:
            context = await self._context_memory.finalize(context)
            self._telemetry.record(
                "context.planning_ready",
                EventSeverity.INFO,
                {
                    "characters": len(context.characters),
                    "recent_concepts": len(context.recent_concepts),
                    "long_term_concepts": len(context.long_term_concepts),
                    "research_items": (
                        len(context.research_brief.items)
                        if context.research_brief is not None
                        else 0
                    ),
                    "trend_signals": len(context.trend_signals),
                },
            )
        selected = await self._idea_director.replenish(context, count=count)
        self._telemetry.record(
            "planner.inventory_replenished",
            EventSeverity.INFO,
            {"count": len(selected), "concept_ids": [item.concept_id for item in selected]},
        )

    async def plan_pack(
        self,
        concept_id: str | None = None,
        editorial_decision_id: str | None = None,
    ) -> None:
        try:
            concept = (
                self._concepts.idea(concept_id)
                if concept_id is not None
                else self._concepts.next_idea()
            )
            if concept is None:
                if self._editorial is not None:
                    self._editorial.fail_decision(
                        editorial_decision_id,
                        error="selected standalone concept is unavailable",
                    )
                return
            record = await self._pack_planner.plan_and_persist(concept)
            self._concepts.set_status(concept.concept_id, "planned")
            if self._editorial is not None:
                self._editorial.complete_decision(
                    editorial_decision_id,
                    pack_id=record.pack_id,
                    concept_id=concept.concept_id,
                )
            self._telemetry.record(
                "pack.planned",
                EventSeverity.INFO,
                {
                    "pack_id": record.pack_id,
                    "concept_id": concept.concept_id,
                    "format": record.plan.format.value,
                    "scene_count": len(record.plan.scenes),
                    "editorial_decision_id": editorial_decision_id,
                    "editorial_lane": "standalone",
                },
            )
        except Exception as exc:
            if self._editorial is not None:
                self._editorial.fail_decision(
                    editorial_decision_id,
                    error=str(exc),
                )
            raise

    async def plan_series_pack(
        self,
        series_id: str,
        editorial_decision_id: str | None = None,
        series_plan_kind: str | None = None,
    ) -> None:
        if self._series_idea_director is None:
            raise RuntimeError("Series Idea Director is not configured")
        try:
            series = self._series.require(series_id)
            raw_context = self._context_builder.build()
            by_id = {item.id: item for item in raw_context.characters}
            missing = [
                character_id
                for character_id in series.character_ids
                if character_id not in by_id
            ]
            if missing:
                raise RuntimeError(
                    "Series contains characters that are not production-ready: "
                    + ", ".join(missing)
                )
            context = raw_context.model_copy(
                update={
                    "characters": tuple(
                        by_id[character_id]
                        for character_id in series.character_ids
                    ),
                    "operator_notes": (
                        f"Series id={series.id}",
                        f"Series title={series.title}",
                        f"Next episode={series.current_episode + 1}",
                        f"Plan kind={series_plan_kind or 'continue'}",
                    ),
                }
            )
            if self._context_memory is not None:
                context = self._context_memory.pre_research(context)
            if self._research is not None:
                topic_parts = [
                    f"Series {series.title}",
                    *series.bible[:3],
                ]
                if series.rolling_summary:
                    topic_parts.append(series.rolling_summary[-400:])
                context = await self._research.prepare(
                    context,
                    topic_hint=" | ".join(topic_parts),
                )
            if self._context_memory is not None:
                context = await self._context_memory.finalize(context)

            kind = (
                SeriesPlanKind(series_plan_kind)
                if series_plan_kind is not None
                else (
                    SeriesPlanKind.START
                    if series.current_episode == 0
                    else SeriesPlanKind.CONTINUE
                )
            )
            concept = await self._series_idea_director.create_concept(
                context,
                series,
                plan_kind=kind,
            )
            record = await self._pack_planner.plan_and_persist(
                concept,
                series=series,
            )
            self._concepts.set_status(concept.concept_id, "planned")
            if self._editorial is not None:
                self._editorial.complete_decision(
                    editorial_decision_id,
                    pack_id=record.pack_id,
                    concept_id=concept.concept_id,
                )
            self._telemetry.record(
                "pack.series_planned",
                EventSeverity.INFO,
                {
                    "pack_id": record.pack_id,
                    "concept_id": concept.concept_id,
                    "series_id": series.id,
                    "episode_number": series.current_episode + 1,
                    "plan_kind": kind.value,
                    "editorial_decision_id": editorial_decision_id,
                },
            )
        except Exception as exc:
            if self._editorial is not None:
                self._editorial.fail_decision(
                    editorial_decision_id,
                    error=str(exc),
                )
            raise

    async def run_pack(self, pack_id: str) -> None:
        pack = self._packs.require(pack_id)
        if pack.state is PackState.PLANNED:
            self._runtime.transition_pack(pack_id, PackState.POLICY_CHECK)
            self._telemetry.record(
                "pack.transition",
                EventSeverity.INFO,
                {"pack_id": pack_id, "state": PackState.POLICY_CHECK.value},
            )

        blocked = await self._prepare_scenes(pack_id)
        if blocked:
            self._runtime.transition_pack(pack_id, PackState.BLOCKED)
            self._telemetry.record(
                "pack.blocked",
                EventSeverity.ERROR,
                {"pack_id": pack_id, "reason": "scene prerequisites/policy blocked"},
            )
            return

        current = self._pack_state(pack_id)
        if current is PackState.POLICY_CHECK:
            self._runtime.transition_pack(pack_id, PackState.GENERATING)
        await self._continue_pack(pack_id)

    async def recover_pack(self, pack_id: str) -> None:
        result = await self._recovery.recover_pack(pack_id)
        if result.state in {
            RecoveryState.WAITING,
            RecoveryState.BACKEND_UNAVAILABLE,
        }:
            return
        await self._continue_pack(pack_id)

    async def _prepare_scenes(self, pack_id: str) -> bool:
        record = self._packs.require(pack_id)
        rows = self._scene_rows(pack_id)
        plans = {plan.ordinal: plan for plan in record.plan.scenes}
        blocked = False

        for row in rows:
            state = SceneState(row.state)
            if state is not SceneState.PLANNED:
                if state is SceneState.BLOCKED:
                    blocked = True
                continue
            plan = plans[row.ordinal]
            decision = self._policy.evaluate_scene(
                row.id,
                use_class=self._use_class(plan.publication_tier),
                content_rating=plan.planned_content_rating,
                requested_tier=plan.publication_tier,
                phase="preflight",
            )
            if decision.outcome is PolicyOutcome.BLOCK:
                self._runtime.transition_scene(
                    row.id,
                    SceneState.BLOCKED,
                    payload_patch={
                        "policy_decision_id": decision.decision_id,
                        "policy_blocked": True,
                    },
                )
                self._ensure_review(
                    "scene",
                    row.id,
                    "policy blocked generation/publication",
                    {"policy_decision_id": decision.decision_id},
                )
                blocked = True
                continue

            try:
                lora_plan = self._loras.resolve(
                    plan.character_ids,
                    model_family=self._production.model_family,
                )
                compiled = self._prompts.compile(plan, lora_plan)
            except (LoRAResolutionError, KeyError, ValueError, RuntimeError) as exc:
                self._runtime.transition_scene(
                    row.id,
                    SceneState.BLOCKED,
                    payload_patch={"prerequisite_error": str(exc)},
                )
                self._ensure_review(
                    "scene",
                    row.id,
                    f"generation prerequisite blocked: {exc}",
                    {},
                )
                blocked = True
                continue

            patch: dict[str, Any] = {
                "compiled_prompt": compiled.model_dump(mode="json"),
                "lora_plan": lora_plan.model_dump(mode="json"),
                "policy_decision_id": decision.decision_id,
            }
            if decision.outcome is PolicyOutcome.REVIEW:
                patch["policy_review_decision_id"] = decision.decision_id
            self._runtime.transition_scene(row.id, SceneState.READY, payload_patch=patch)

        return blocked

    async def _continue_pack(self, pack_id: str) -> None:
        for row in self._scene_rows(pack_id):
            scene_state = SceneState(row.state)
            if scene_state is SceneState.RETRY:
                self._runtime.transition_scene(row.id, SceneState.READY)
                scene_state = SceneState.READY
            if scene_state is SceneState.READY:
                await self._generate_and_evaluate(row.id)
            elif scene_state is SceneState.EVALUATING:
                await self._evaluate_recovered(row.id)

        pack_state = self._pack_state(pack_id)
        if pack_state is PackState.GENERATING:
            self._runtime.transition_pack(pack_id, PackState.EVALUATING)
            pack_state = PackState.EVALUATING
        if pack_state is not PackState.EVALUATING:
            return

        scene_states = [SceneState(row.state) for row in self._scene_rows(pack_id)]
        if any(item is SceneState.BLOCKED for item in scene_states):
            self._runtime.transition_pack(pack_id, PackState.BLOCKED)
            return
        if any(item is SceneState.FAILED for item in scene_states):
            self._runtime.transition_pack(pack_id, PackState.FAILED)
            return
        if any(item is SceneState.REVIEW for item in scene_states):
            self._runtime.transition_pack(pack_id, PackState.REVIEW)
            return
        if any(item is SceneState.REJECTED for item in scene_states):
            self._runtime.transition_pack(pack_id, PackState.FAILED)
            return
        if not scene_states or any(
            item is not SceneState.ACCEPTED for item in scene_states
        ):
            return

        archive = self._archive.finalize(pack_id)
        record = self._packs.require(pack_id)
        if record.series_id is not None and record.plan.episode_number is not None:
            updates: dict[str, Any] = {}
            hooks_added: list[str] = []
            hooks_resolved: list[str] = []
            for scene in record.plan.scenes:
                updates.update(scene.series_state_updates)
                hooks_added.extend(scene.unresolved_hooks_added)
                hooks_resolved.extend(scene.unresolved_hooks_resolved)
            self._series.record_completed_pack(
                record.series_id,
                pack_id=pack_id,
                episode_number=record.plan.episode_number,
                continuity_updates=updates,
                hooks_added=hooks_added,
                hooks_resolved=hooks_resolved,
            )

        self._characters.record_use(record.plan.character_ids)
        if record.concept_id is not None:
            self._concepts.set_status(record.concept_id, "finalized")
        self._runtime.transition_pack(
            pack_id,
            PackState.FINALIZED,
            checkpoint_patch={
                "archive_complete": True,
                "archive_manifest_sha256": archive.manifest_sha256,
            },
        )
        if self._editorial is not None:
            self._editorial.mark_finalized_pack(
                pack_id,
                metadata={
                    "series_id": record.series_id,
                    "concept_id": record.concept_id,
                    "format": record.plan.format.value,
                },
            )
        self._telemetry.record(
            "pack.finalized",
            EventSeverity.INFO,
            {
                "pack_id": pack_id,
                "manifest": str(archive.manifest_path),
                "manifest_sha256": archive.manifest_sha256,
            },
        )

    async def _generate_and_evaluate(self, scene_id: str) -> None:
        while True:
            row = self._scene_row(scene_id)
            state = SceneState(row.state)
            if state is not SceneState.READY:
                return
            attempts_used = len(self._attempts.list_for_scene(scene_id))
            if attempts_used >= self._operations.max_attempts_per_scene:
                self._runtime.transition_scene(scene_id, SceneState.FAILED)
                self._ensure_review(
                    "scene",
                    scene_id,
                    "maximum generation attempts exceeded",
                    {},
                )
                return

            retry_inputs = self._retry_inputs(row)
            compiled = retry_inputs.compiled
            lora_plan = retry_inputs.lora_plan
            seed = self._seed_factory()
            raw_retry_actions = row.payload_json.get("retry_actions", ())
            retry_actions = (
                tuple(
                    value
                    for value in raw_retry_actions
                    if isinstance(value, str)
                )
                if isinstance(raw_retry_actions, list | tuple)
                else ()
            )
            backend_provenance = dict(self._backend.provenance())
            raw_history = row.payload_json.get("retry_history", ())
            latest_retry = (
                raw_history[-1]
                if isinstance(raw_history, list) and raw_history
                else None
            )
            effective_workflow = (
                retry_inputs.workflow_template_id
                or backend_provenance.get("workflow_template")
            )
            provenance = {
                **backend_provenance,
                "workflow_template": effective_workflow,
                "compiled_prompt": compiled.model_dump(mode="json"),
                "lora_plan": lora_plan.model_dump(mode="json"),
                "retry_actions": list(retry_actions),
                "retry_seed_revision": retry_inputs.seed_revision,
                "retry_execution": latest_retry,
            }
            previous = self._attempts.latest_for_scene(scene_id)
            attempt = self._attempts.create(
                scene_id=scene_id,
                prompt=compiled.positive_prompt,
                negative_prompt=compiled.negative_prompt,
                seed=seed,
                provenance=provenance,
                parent_attempt_id=previous.id if previous is not None else None,
            )
            self._runtime.transition_scene(scene_id, SceneState.GENERATING)
            request = GenerationRequest(
                attempt_id=attempt.id,
                scene_id=scene_id,
                compiled=compiled,
                lora_plan=lora_plan,
                seed=seed,
                output_prefix=f"ARTIFEX/{self._scene_pack_id(scene_id)}/{scene_id}",
                workflow_template_id=retry_inputs.workflow_template_id,
            )

            infrastructure_retries = 0
            on_submitted = partial(self._record_submission, attempt.id)

            while True:
                try:
                    batch = await self._backend.generate(
                        request,
                        on_submitted=on_submitted,
                    )
                    break
                except ComfyUIError as exc:
                    decision = self._retries.for_infrastructure(
                        exc,
                        infrastructure_retries_used=infrastructure_retries,
                    )
                    self._telemetry.record(
                        "generation.infrastructure_error",
                        EventSeverity.WARNING,
                        {
                            "scene_id": scene_id,
                            "attempt_id": attempt.id,
                            "kind": exc.kind.value,
                            "retry": decision.should_retry,
                            "reason": decision.reason,
                        },
                    )
                    if not decision.should_retry:
                        self._attempts.record_backend_status(
                            attempt.id,
                            status="failed",
                            error={"kind": exc.kind.value, "detail": str(exc)},
                        )
                        self._runtime.transition_scene(scene_id, SceneState.REVIEW)
                        self._ensure_review(
                            "scene",
                            scene_id,
                            decision.reason,
                            {"attempt_id": attempt.id},
                        )
                        return
                    infrastructure_retries += 1
                except RuntimeError as exc:
                    self._attempts.record_backend_status(
                        attempt.id,
                        status="failed",
                        error={"kind": "configuration", "detail": str(exc)},
                    )
                    self._runtime.transition_scene(scene_id, SceneState.REVIEW)
                    self._ensure_review(
                        "scene",
                        scene_id,
                        f"generation backend configuration error: {exc}",
                        {"attempt_id": attempt.id},
                    )
                    return

            self._attempts.patch_provenance(
                attempt.id,
                {
                    "comfy_prompt_id": batch.prompt_id,
                    "output_paths": [str(path) for path in batch.output_paths],
                    "backend_outputs": list(batch.outputs),
                },
            )
            self._attempts.record_backend_status(attempt.id, status="completed")
            self._runtime.transition_scene(scene_id, SceneState.EVALUATING)

            result = await self._evaluate_attempt(
                scene_id,
                attempt.id,
                batch.output_paths[0],
            )
            if result.state is ResultState.ACCEPTED:
                self._handle_accepted_result(scene_id, attempt.id, result)
                return

            retry_decision = self._retries.for_evaluation(
                result,
                production_retries_used=attempt.ordinal - 1,
            )
            if (
                retry_decision.should_retry
                and attempt.ordinal < self._operations.max_attempts_per_scene
                and self._apply_retry_decision(
                    scene_id,
                    result,
                    retry_decision.actions,
                    retry_decision.next_retry_number or attempt.ordinal,
                )
            ):
                self._runtime.transition_scene(scene_id, SceneState.READY)
                continue

            selection = self._selector.select(scene_id)
            if selection.state is ResultState.REJECTED:
                self._runtime.transition_scene(scene_id, SceneState.REVIEW)
            self._ensure_review(
                "scene",
                scene_id,
                retry_decision.reason,
                {"attempt_id": attempt.id},
            )
            return

    async def _evaluate_recovered(self, scene_id: str) -> None:
        attempt = self._attempts.latest_for_scene(scene_id)
        if attempt is None or attempt.backend_status != "completed":
            return
        paths = self._output_paths(attempt)
        if not paths:
            self._runtime.transition_scene(scene_id, SceneState.REVIEW)
            self._ensure_review(
                "scene",
                scene_id,
                "recovered attempt has no persisted output path",
                {"attempt_id": attempt.id},
            )
            return
        result = await self._evaluate_attempt(scene_id, attempt.id, paths[0])
        if result.state is ResultState.ACCEPTED:
            self._handle_accepted_result(scene_id, attempt.id, result)
            return

        retry_decision = self._retries.for_evaluation(
            result,
            production_retries_used=attempt.ordinal - 1,
        )
        if (
            retry_decision.should_retry
            and attempt.ordinal < self._operations.max_attempts_per_scene
            and self._apply_retry_decision(
                scene_id,
                result,
                retry_decision.actions,
                retry_decision.next_retry_number or attempt.ordinal,
            )
        ):
            self._runtime.transition_scene(scene_id, SceneState.READY)
            await self._generate_and_evaluate(scene_id)
            return
        selection = self._selector.select(scene_id)
        if selection.state is ResultState.REJECTED:
            self._runtime.transition_scene(scene_id, SceneState.REVIEW)
        self._ensure_review(
            "scene",
            scene_id,
            retry_decision.reason,
            {"attempt_id": attempt.id},
        )

    async def _evaluate_attempt(
        self,
        scene_id: str,
        attempt_id: str,
        image_path: Path,
    ) -> EvaluationResult:
        row = self._scene_row(scene_id)
        plan = self._scene_plan(row)
        attempt = self._attempts.require(attempt_id)
        result = await self._evaluator.evaluate(
            EvaluationContext(
                attempt_id=attempt_id,
                scene_id=scene_id,
                image_path=image_path,
                character_ids=plan.character_ids,
                positive_prompt=attempt.prompt,
                negative_prompt=attempt.negative_prompt,
                reference_image_paths=self._reference_paths(scene_id),
                adjacent_image_paths=self._adjacent_paths(row.pack_id, row.ordinal),
            )
        )
        self._evaluations.record(result)
        self._telemetry.record(
            "generation.evaluated",
            EventSeverity.INFO,
            {
                "scene_id": scene_id,
                "attempt_id": attempt_id,
                "state": result.state.value,
                "aggregate": result.scores.aggregate,
                "reasons": list(result.reasons),
            },
        )
        return result

    def _record_submission(self, attempt_id: str, prompt_id: str) -> None:
        current = self._attempts.require(attempt_id)
        raw_ids = current.provenance_json.get("comfy_prompt_ids", ())
        ids = [
            value for value in raw_ids
            if isinstance(value, str)
        ] if isinstance(raw_ids, list | tuple) else []
        if prompt_id not in ids:
            ids.append(prompt_id)
        self._attempts.patch_provenance(
            attempt_id,
            {
                "comfy_prompt_id": prompt_id,
                "comfy_prompt_ids": ids,
            },
        )
        self._attempts.record_backend_status(attempt_id, status="running")

    def _ensure_review(
        self,
        subject_type: str,
        subject_id: str,
        reason: str,
        payload: Mapping[str, object],
    ) -> None:
        if self._reviews.find_open(
            subject_type=subject_type,
            subject_id=subject_id,
        ) is None:
            self._reviews.enqueue(
                subject_type=subject_type,
                subject_id=subject_id,
                reason=reason,
                payload=payload,
            )

    def _retry_inputs(self, row: SceneRow) -> RetryProductionInputs:
        raw_compiled = row.payload_json.get("compiled_prompt")
        raw_lora = row.payload_json.get("lora_plan")
        if not isinstance(raw_compiled, dict) or not isinstance(raw_lora, dict):
            raise TypeError(f"scene {row.id} is missing compiled production inputs")
        raw_scene = row.payload_json.get("retry_scene_plan")
        if not isinstance(raw_scene, dict):
            raw_scene = row.payload_json.get("plan")
        if not isinstance(raw_scene, dict):
            raise TypeError(f"scene {row.id} is missing persisted plan")
        raw_workflow = row.payload_json.get("workflow_template_id")
        workflow = raw_workflow if isinstance(raw_workflow, str) else None
        raw_seed_revision = row.payload_json.get("retry_seed_revision", 0)
        seed_revision = (
            raw_seed_revision
            if isinstance(raw_seed_revision, int) and raw_seed_revision >= 0
            else 0
        )
        return RetryProductionInputs(
            scene=ScenePlan.model_validate(raw_scene),
            compiled=CompiledPrompt.model_validate(raw_compiled),
            lora_plan=LoRAPlan.model_validate(raw_lora),
            workflow_template_id=workflow,
            seed_revision=seed_revision,
        )

    def _retry_history(self, row: SceneRow) -> tuple[RetryExecutionRecord, ...]:
        raw = row.payload_json.get("retry_history", ())
        if not isinstance(raw, list | tuple):
            return ()
        records: list[RetryExecutionRecord] = []
        for item in raw:
            if isinstance(item, dict):
                records.append(RetryExecutionRecord.model_validate(item))
        return tuple(records)

    def _apply_retry_decision(
        self,
        scene_id: str,
        result: EvaluationResult,
        actions: tuple[RetryAction, ...],
        retry_number: int,
    ) -> bool:
        row = self._scene_row(scene_id)
        history = self._retry_history(row)
        execution = self._retry_executor.execute(
            inputs=self._retry_inputs(row),
            actions=actions,
            reasons=result.reasons,
            retry_number=retry_number,
            history=history,
        )
        record = execution.record
        updated_history = (
            *history,
            record,
        )
        if not execution.can_retry:
            self._runtime.transition_scene(
                scene_id,
                SceneState.REVIEW,
                payload_patch={
                    "retry_reason": "retry actions exhausted, ineffective, or cyclic",
                    "retry_actions": [
                        action.value for action in record.requested_actions
                    ],
                    "retry_history": [
                        item.model_dump(mode="json")
                        for item in updated_history
                    ],
                    "retry_cycle_detected": record.cycle_detected,
                },
            )
            return False

        after = record.after_inputs
        self._runtime.transition_scene(
            scene_id,
            SceneState.RETRY,
            payload_patch={
                "retry_reason": "reason-aware retry actions applied",
                "retry_actions": [
                    action.value for action in record.applied_actions
                ],
                "retry_history": [
                    item.model_dump(mode="json")
                    for item in updated_history
                ],
                "retry_scene_plan": after.scene.model_dump(mode="json"),
                "compiled_prompt": after.compiled.model_dump(mode="json"),
                "lora_plan": after.lora_plan.model_dump(mode="json"),
                "workflow_template_id": after.workflow_template_id,
                "retry_seed_revision": after.seed_revision,
            },
        )
        return True


    def _scene_plan(self, row: SceneRow) -> ScenePlan:
        raw = row.payload_json.get("plan")
        if not isinstance(raw, dict):
            raise TypeError(f"scene {row.id} is missing persisted plan")
        return ScenePlan.model_validate(raw)

    def _scene_rows(self, pack_id: str) -> tuple[SceneRow, ...]:
        with self._database.session() as session:
            rows = session.scalars(
                select(SceneRow)
                .where(SceneRow.pack_id == pack_id)
                .order_by(SceneRow.ordinal.asc(), SceneRow.id.asc())
            ).all()
            for row in rows:
                session.expunge(row)
            return tuple(rows)

    def _scene_row(self, scene_id: str) -> SceneRow:
        with self._database.session() as session:
            row = session.get(SceneRow, scene_id)
            if row is None:
                raise KeyError(f"unknown scene: {scene_id}")
            session.expunge(row)
            return row

    def _pack_state(self, pack_id: str) -> PackState:
        with self._database.session() as session:
            row = session.get(PackRow, pack_id)
            if row is None:
                raise KeyError(f"unknown pack: {pack_id}")
            return PackState(row.state)

    def _scene_pack_id(self, scene_id: str) -> str:
        return self._scene_row(scene_id).pack_id

    def _reference_paths(self, scene_id: str) -> tuple[Path, ...]:
        with self._database.session() as session:
            selected_ids = session.scalars(
                select(SceneRow.selected_attempt_id)
                .where(
                    SceneRow.selected_attempt_id.is_not(None),
                    SceneRow.id != scene_id,
                )
                .order_by(SceneRow.pack_id.desc(), SceneRow.ordinal.desc())
                .limit(20)
            ).all()
            attempts = session.scalars(
                select(GenerationAttemptRow).where(
                    GenerationAttemptRow.id.in_(selected_ids)
                )
            ).all()
        paths: list[Path] = []
        for attempt in attempts:
            paths.extend(self._output_paths(attempt))
        return tuple(path for path in paths if path.exists())[:20]

    def _adjacent_paths(self, pack_id: str, ordinal: int) -> tuple[Path, ...]:
        with self._database.session() as session:
            scene = session.scalar(
                select(SceneRow)
                .where(
                    SceneRow.pack_id == pack_id,
                    SceneRow.ordinal < ordinal,
                    SceneRow.selected_attempt_id.is_not(None),
                )
                .order_by(SceneRow.ordinal.desc())
                .limit(1)
            )
            if scene is None or scene.selected_attempt_id is None:
                return ()
            attempt = session.get(GenerationAttemptRow, scene.selected_attempt_id)
            if attempt is None:
                return ()
            session.expunge(attempt)
        return tuple(path for path in self._output_paths(attempt) if path.exists())

    @staticmethod
    def _output_paths(attempt: GenerationAttemptRow) -> tuple[Path, ...]:
        raw = attempt.provenance_json.get("output_paths", ())
        if not isinstance(raw, list | tuple):
            return ()
        return tuple(
            Path(value)
            for value in raw
            if isinstance(value, str) and value
        )

    def _handle_accepted_result(
        self,
        scene_id: str,
        attempt_id: str,
        result: EvaluationResult,
    ) -> None:
        row = self._scene_row(scene_id)
        plan = self._scene_plan(row)
        decision = self._policy.evaluate_scene(
            scene_id,
            use_class=self._use_class(plan.publication_tier),
            content_labels=result.content_labels,
            content_rating=result.content_rating,
            requested_tier=plan.publication_tier,
            phase="post_generation",
        )

        if decision.outcome is PolicyOutcome.BLOCK:
            self._runtime.transition_scene(
                scene_id,
                SceneState.BLOCKED,
                payload_patch={
                    "post_generation_policy_decision_id": decision.decision_id,
                    "post_generation_policy_blocked": True,
                    "post_generation_content_rating": result.content_rating.value,
                    "post_generation_content_labels": list(result.content_labels),
                },
            )
            self._ensure_review(
                "scene",
                scene_id,
                "post-generation policy hard-blocked publication; retry or reject",
                {
                    "attempt_id": attempt_id,
                    "policy_decision_id": decision.decision_id,
                    "content_rating": result.content_rating.value,
                    "content_labels": list(result.content_labels),
                },
            )
            return

        review_decision_id: str | None = None
        if decision.outcome is PolicyOutcome.REVIEW:
            review_decision_id = decision.decision_id
            self._runtime.transition_scene(
                scene_id,
                SceneState.EVALUATING,
                payload_patch={
                    "post_generation_policy_review_decision_id": decision.decision_id,
                },
            )
        else:
            prior = row.payload_json.get("policy_review_decision_id")
            if isinstance(prior, str) and prior:
                review_decision_id = prior

        if review_decision_id is not None:
            self._runtime.select_scene_attempt(
                scene_id,
                attempt_id,
                SceneState.REVIEW,
            )
            self._ensure_review(
                "scene",
                scene_id,
                "policy requires operator review after output classification",
                {
                    "attempt_id": attempt_id,
                    "policy_decision_id": review_decision_id,
                    "content_rating": result.content_rating.value,
                    "content_labels": list(result.content_labels),
                },
            )
            return

        self._selector.select(scene_id)

    @staticmethod
    def _use_class(tier: PublicationTier) -> UseClass:
        if tier is PublicationTier.PUBLIC:
            return UseClass.PUBLIC_FREE
        if tier is PublicationTier.MEMBER:
            return UseClass.PAID_MEMBERSHIP
        return UseClass.PRIVATE
