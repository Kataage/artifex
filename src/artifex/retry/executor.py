from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ProductionConfig
from artifex.loras import LoRAPlan, LoRAResolutionError, LoRAResolver
from artifex.packs import ScenePlan
from artifex.prompts import CompiledPrompt, PromptCompiler
from artifex.retry.models import RetryAction


class RetryExecutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetryProductionInputs(RetryExecutionModel):
    scene: ScenePlan
    compiled: CompiledPrompt
    lora_plan: LoRAPlan
    workflow_template_id: str | None = None
    seed_revision: int = Field(default=0, ge=0)


class RetryExecutionRecord(RetryExecutionModel):
    retry_number: int = Field(ge=1)
    requested_actions: tuple[RetryAction, ...]
    applied_actions: tuple[RetryAction, ...]
    skipped_actions: tuple[RetryAction, ...]
    reasons: tuple[str, ...]
    before_digest: str
    after_digest: str
    before_inputs: RetryProductionInputs
    after_inputs: RetryProductionInputs
    cycle_detected: bool = False


class RetryExecutionResult(RetryExecutionModel):
    can_retry: bool
    record: RetryExecutionRecord


class RetryActionExecutor:
    def __init__(
        self,
        loras: LoRAResolver,
        prompts: PromptCompiler,
        config: ProductionConfig,
    ) -> None:
        self._loras = loras
        self._prompts = prompts
        self._config = config

    def execute(
        self,
        *,
        inputs: RetryProductionInputs,
        actions: tuple[RetryAction, ...],
        reasons: tuple[str, ...],
        retry_number: int,
        history: tuple[RetryExecutionRecord, ...] = (),
    ) -> RetryExecutionResult:
        current = inputs
        applied: list[RetryAction] = []
        skipped: list[RetryAction] = []

        for action in actions:
            if action in {
                RetryAction.OPERATOR_REVIEW,
                RetryAction.BLOCK_MISSING_ASSET,
                RetryAction.INFRASTRUCTURE_RETRY,
            }:
                skipped.append(action)
                continue
            if self._used(history, action) >= self._limit(action):
                skipped.append(action)
                continue

            updated = self._apply(
                action,
                current,
                reasons=reasons,
                retry_number=retry_number,
            )
            if updated == current:
                skipped.append(action)
                continue
            current = updated
            applied.append(action)

        before_digest = _digest(inputs)
        after_digest = _digest(current)
        prior_digests = {item.after_digest for item in history}
        cycle_detected = after_digest in prior_digests
        can_retry = bool(applied) and not cycle_detected and after_digest != before_digest

        return RetryExecutionResult(
            can_retry=can_retry,
            record=RetryExecutionRecord(
                retry_number=retry_number,
                requested_actions=actions,
                applied_actions=tuple(applied),
                skipped_actions=tuple(skipped),
                reasons=reasons,
                before_digest=before_digest,
                after_digest=after_digest,
                before_inputs=inputs,
                after_inputs=current,
                cycle_detected=cycle_detected,
            ),
        )

    def _apply(
        self,
        action: RetryAction,
        inputs: RetryProductionInputs,
        *,
        reasons: tuple[str, ...],
        retry_number: int,
    ) -> RetryProductionInputs:
        if action is RetryAction.ALTERNATE_LORA:
            excluded = tuple(entry.lora_id for entry in inputs.lora_plan.entries)
            try:
                alternate = self._loras.resolve(
                    inputs.scene.character_ids,
                    model_family=inputs.lora_plan.model_family,
                    exclude_lora_ids=excluded,
                )
            except LoRAResolutionError:
                return inputs
            if alternate == inputs.lora_plan:
                return inputs
            return inputs.model_copy(
                update={
                    "lora_plan": alternate,
                    "compiled": self._prompts.compile(inputs.scene, alternate),
                }
            )

        if action is RetryAction.ADJUST_LORA_WEIGHT:
            adjusted = self._loras.adjust_weights(
                inputs.lora_plan,
                delta=self._config.retry_lora_weight_step,
            )
            if adjusted == inputs.lora_plan:
                return inputs
            return inputs.model_copy(
                update={
                    "lora_plan": adjusted,
                    "compiled": self._prompts.compile(inputs.scene, adjusted),
                }
            )

        if action is RetryAction.REVISE_PROMPT:
            scene = _revise_scene(inputs.scene, reasons, retry_number)
            if scene == inputs.scene:
                return inputs
            return inputs.model_copy(
                update={
                    "scene": scene,
                    "compiled": self._prompts.compile(scene, inputs.lora_plan),
                }
            )

        if action is RetryAction.REPAIR_WORKFLOW:
            target = self._config.repair_workflow_template
            if inputs.workflow_template_id == target:
                return inputs
            return inputs.model_copy(update={"workflow_template_id": target})

        if action is RetryAction.VARY_SCENE:
            scene = _vary_scene(inputs.scene, retry_number)
            if scene == inputs.scene:
                return inputs
            return inputs.model_copy(
                update={
                    "scene": scene,
                    "compiled": self._prompts.compile(scene, inputs.lora_plan),
                }
            )

        if action is RetryAction.CHANGE_SEED:
            return inputs.model_copy(
                update={"seed_revision": inputs.seed_revision + 1}
            )

        return inputs

    def _limit(self, action: RetryAction) -> int:
        return {
            RetryAction.ALTERNATE_LORA: self._config.retry_alternate_lora_limit,
            RetryAction.ADJUST_LORA_WEIGHT: (
                self._config.retry_adjust_lora_weight_limit
            ),
            RetryAction.REVISE_PROMPT: self._config.retry_revise_prompt_limit,
            RetryAction.REPAIR_WORKFLOW: (
                self._config.retry_repair_workflow_limit
            ),
            RetryAction.VARY_SCENE: self._config.retry_vary_scene_limit,
            RetryAction.CHANGE_SEED: self._config.retry_change_seed_limit,
        }.get(action, 0)

    @staticmethod
    def _used(
        history: tuple[RetryExecutionRecord, ...],
        action: RetryAction,
    ) -> int:
        return sum(
            1
            for record in history
            if action in record.applied_actions
        )


def _digest(inputs: RetryProductionInputs) -> str:
    payload = json.dumps(
        inputs.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _revise_scene(
    scene: ScenePlan,
    reasons: tuple[str, ...],
    retry_number: int,
) -> ScenePlan:
    positives = list(scene.visual.positive_constraints)
    negatives = list(scene.visual.negative_constraints)

    reason_text = " ".join(reasons)
    if "identity_" in reason_text:
        positive_choices = (
            "canonical character identity",
            "facial identity consistency",
        )
        negative_choices = (
            "wrong character, identity drift",
            "face mismatch, character mismatch",
        )
    elif "alignment_" in reason_text:
        positive_choices = (
            "match requested pose and composition",
            "strict prompt adherence",
        )
        negative_choices = (
            "pose mismatch, composition mismatch",
            "prompt deviation",
        )
    elif "continuity_" in reason_text:
        positive_choices = (
            "same outfit and character design",
            "continuity with prior scene",
        )
        negative_choices = (
            "outfit drift, design drift",
            "continuity break",
        )
    else:
        positive_choices = ("strict prompt adherence", "clean subject definition")
        negative_choices = ("prompt deviation", "ambiguous subject")

    index = (retry_number - 1) % len(positive_choices)
    positives.append(positive_choices[index])
    negatives.append(negative_choices[index])
    visual = scene.visual.model_copy(
        update={
            "positive_constraints": tuple(dict.fromkeys(positives)),
            "negative_constraints": tuple(dict.fromkeys(negatives)),
        }
    )
    return scene.model_copy(update={"visual": visual})


def _vary_scene(scene: ScenePlan, retry_number: int) -> ScenePlan:
    variants: tuple[dict[str, str], ...] = (
        {
            "composition": "cowboy shot",
            "camera": "three-quarter view",
            "expression": "soft smile",
        },
        {
            "composition": "upper body",
            "camera": "slightly from above",
            "expression": "gentle expression",
        },
    )
    variant = variants[(retry_number - 1) % len(variants)]
    visual = scene.visual.model_copy(update=variant)
    return scene.model_copy(update={"visual": visual})
