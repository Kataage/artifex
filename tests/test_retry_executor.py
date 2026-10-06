from __future__ import annotations

from pathlib import Path

from artifex.characters import CharacterRegistry
from artifex.config.models import (
    CharacterRegistryConfig,
    LoRARegistryConfig,
    ProductionConfig,
)
from artifex.db import Database
from artifex.domain import CharacterProfile, LoRAPolicy, LoRAProfile, LoRAState
from artifex.loras import LoRARegistry, LoRAResolver
from artifex.packs import ScenePlan, VisualSpecification
from artifex.prompts import ILXLDanbooruAdapter, PromptCompiler
from artifex.retry import (
    RetryAction,
    RetryActionExecutor,
    RetryExecutionRecord,
    RetryPolicy,
    RetryProductionInputs,
)


def _setup(tmp_path: Path):
    database = Database(f"sqlite:///{(tmp_path / 'retry.sqlite3').as_posix()}")
    database.migrate()
    characters = CharacterRegistry(database)
    loras = LoRARegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="Character A",
            namespace="test",
            canonical_tags=("char_a",),
            model_families=("ilxl",),
            lora_policy=LoRAPolicy.REQUIRED,
            preferred_lora_ids=("lora-a", "lora-b"),
            readiness=1.0,
        )
    )
    for lora_id, weight, readiness in (
        ("lora-a", 0.8, 0.95),
        ("lora-b", 0.75, 0.90),
    ):
        loras.upsert(
            LoRAProfile(
                id=lora_id,
                path=tmp_path / f"{lora_id}.safetensors",
                state=LoRAState.PRODUCTION,
                target_character_ids=("char-a",),
                model_families=("ilxl",),
                trigger_tags=(f"{lora_id}_trigger",),
                recommended_weight=weight,
                validated_min_weight=0.6,
                validated_max_weight=0.9,
                identity_score=readiness,
                quality_score=readiness,
                flexibility_score=0.9,
                readiness=readiness,
            )
        )
    resolver = LoRAResolver(
        characters,
        loras,
        CharacterRegistryConfig(),
        LoRARegistryConfig(),
    )
    compiler = PromptCompiler(characters, (ILXLDanbooruAdapter(),))
    scene = ScenePlan(
        ordinal=1,
        title="Retry scene",
        purpose="exercise retry remediation",
        character_ids=("char-a",),
        role="feature",
        visual=VisualSpecification(
            composition="full body",
            camera="eye level",
            pose="standing",
            expression="smile",
            clothing="white dress",
            setting="open window",
            lighting="sunset light",
            atmosphere="airy",
        ),
        publication_tier="public",
    )
    plan = resolver.resolve(("char-a",), model_family="ilxl")
    compiled = compiler.compile(scene, plan)
    inputs = RetryProductionInputs(
        scene=scene,
        compiled=compiled,
        lora_plan=plan,
    )
    executor = RetryActionExecutor(
        resolver,
        compiler,
        ProductionConfig(
            retry_limit=8,
            retry_alternate_lora_limit=2,
            retry_adjust_lora_weight_limit=4,
            retry_revise_prompt_limit=4,
            retry_repair_workflow_limit=2,
            retry_vary_scene_limit=8,
            retry_change_seed_limit=8,
        ),
    )
    return database, executor, inputs


def test_alternate_lora_materially_changes_selected_asset(tmp_path: Path) -> None:
    database, executor, inputs = _setup(tmp_path)

    result = executor.execute(
        inputs=inputs,
        actions=(RetryAction.ALTERNATE_LORA,),
        reasons=("identity_hard_min",),
        retry_number=1,
    )

    assert result.can_retry is True
    assert result.record.applied_actions == (RetryAction.ALTERNATE_LORA,)
    assert result.record.before_inputs.lora_plan.entries[0].lora_id == "lora-a"
    assert result.record.after_inputs.lora_plan.entries[0].lora_id == "lora-b"
    assert result.record.before_digest != result.record.after_digest
    database.dispose()


def test_adjust_lora_weight_stays_inside_validated_bounds(tmp_path: Path) -> None:
    database, executor, inputs = _setup(tmp_path)

    first = executor.execute(
        inputs=inputs,
        actions=(RetryAction.ADJUST_LORA_WEIGHT,),
        reasons=("identity_accept_min",),
        retry_number=1,
    )
    second = executor.execute(
        inputs=first.record.after_inputs,
        actions=(RetryAction.ADJUST_LORA_WEIGHT,),
        reasons=("identity_accept_min",),
        retry_number=2,
        history=(first.record,),
    )

    assert first.record.after_inputs.lora_plan.entries[0].weight == 0.9
    assert second.record.applied_actions == ()
    assert second.can_retry is False
    database.dispose()


def test_revise_prompt_changes_structured_scene_and_compiled_prompt(
    tmp_path: Path,
) -> None:
    database, executor, inputs = _setup(tmp_path)

    result = executor.execute(
        inputs=inputs,
        actions=(RetryAction.REVISE_PROMPT,),
        reasons=("alignment_review_min",),
        retry_number=1,
    )

    assert result.can_retry is True
    assert (
        result.record.after_inputs.scene.visual.positive_constraints
        != inputs.scene.visual.positive_constraints
    )
    assert (
        result.record.after_inputs.compiled.positive_prompt
        != inputs.compiled.positive_prompt
    )
    database.dispose()


def test_repair_workflow_routes_to_dedicated_template(tmp_path: Path) -> None:
    database, executor, inputs = _setup(tmp_path)

    result = executor.execute(
        inputs=inputs,
        actions=(RetryAction.REPAIR_WORKFLOW,),
        reasons=("technical_quality_review_min",),
        retry_number=1,
    )

    assert result.can_retry is True
    assert result.record.after_inputs.workflow_template_id == "illust_main_repair_v1"
    database.dispose()


def test_vary_scene_changes_semantics_but_preserves_identity_continuity(
    tmp_path: Path,
) -> None:
    database, executor, inputs = _setup(tmp_path)

    result = executor.execute(
        inputs=inputs,
        actions=(RetryAction.VARY_SCENE,),
        reasons=("image_similarity_hard_max",),
        retry_number=1,
    )

    after = result.record.after_inputs.scene
    assert result.can_retry is True
    assert after.visual.composition != inputs.scene.visual.composition
    assert after.visual.camera != inputs.scene.visual.camera
    assert after.character_ids == inputs.scene.character_ids
    assert after.visual.clothing == inputs.scene.visual.clothing
    assert after.visual.setting == inputs.scene.visual.setting
    database.dispose()


def test_change_seed_advances_seed_revision(tmp_path: Path) -> None:
    database, executor, inputs = _setup(tmp_path)

    result = executor.execute(
        inputs=inputs,
        actions=(RetryAction.CHANGE_SEED,),
        reasons=("aggregate_review_min",),
        retry_number=1,
    )

    assert result.can_retry is True
    assert result.record.after_inputs.seed_revision == 1
    assert result.record.before_digest != result.record.after_digest
    database.dispose()


def test_action_budget_and_cycle_detection_bound_retries(tmp_path: Path) -> None:
    database, executor, inputs = _setup(tmp_path)

    first = executor.execute(
        inputs=inputs,
        actions=(RetryAction.VARY_SCENE,),
        reasons=("image_similarity_hard_max",),
        retry_number=1,
    )
    second = executor.execute(
        inputs=first.record.after_inputs,
        actions=(RetryAction.VARY_SCENE,),
        reasons=("image_similarity_hard_max",),
        retry_number=2,
        history=(first.record,),
    )
    third = executor.execute(
        inputs=second.record.after_inputs,
        actions=(RetryAction.VARY_SCENE,),
        reasons=("image_similarity_hard_max",),
        retry_number=3,
        history=(first.record, second.record),
    )

    assert first.can_retry is True
    assert second.can_retry is True
    assert third.can_retry is False
    assert third.record.cycle_detected is True
    database.dispose()


def test_retry_policy_maps_failure_reasons_to_executable_actions() -> None:
    policy = RetryPolicy(ProductionConfig(retry_limit=5))

    identity = policy._actions_for_reasons(("identity_hard_min",))
    technical = policy._actions_for_reasons(("technical_quality_review_min",))
    similarity = policy._actions_for_reasons(("image_similarity_hard_max",))

    assert identity == (
        RetryAction.ALTERNATE_LORA,
        RetryAction.ADJUST_LORA_WEIGHT,
        RetryAction.REVISE_PROMPT,
    )
    assert technical == (
        RetryAction.CHANGE_SEED,
        RetryAction.REPAIR_WORKFLOW,
    )
    assert similarity == (
        RetryAction.VARY_SCENE,
        RetryAction.CHANGE_SEED,
    )


def test_retry_record_serializes_complete_before_after_provenance(
    tmp_path: Path,
) -> None:
    database, executor, inputs = _setup(tmp_path)
    result = executor.execute(
        inputs=inputs,
        actions=(RetryAction.REVISE_PROMPT, RetryAction.CHANGE_SEED),
        reasons=("alignment_review_min",),
        retry_number=1,
    )

    payload = result.record.model_dump(mode="json")
    restored = RetryExecutionRecord.model_validate(payload)

    assert restored.before_inputs.compiled == inputs.compiled
    assert restored.after_inputs.compiled != inputs.compiled
    assert restored.after_inputs.seed_revision == 1
    assert restored.before_digest != restored.after_digest
    database.dispose()
