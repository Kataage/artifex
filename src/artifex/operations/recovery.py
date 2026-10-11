from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from artifex.comfy import ComfyExecutionResult, ComfyUIError
from artifex.db import Database
from artifex.db.models import PackRow, SceneRow
from artifex.domain import PackState, SceneState
from artifex.evaluation.repository import GenerationAttemptRepository
from artifex.review import ReviewQueueRepository
from artifex.runtime import RuntimeStore
from artifex.telemetry import EventSeverity, TelemetryRepository


class RecoveryState(StrEnum):
    NOOP = "noop"
    RECOVERED = "recovered"
    WAITING = "waiting"
    RESUBMIT = "resubmit"  # Legacy result; uncertain GPU jobs must not resubmit.
    REVIEW = "review"
    BACKEND_UNAVAILABLE = "backend_unavailable"


class RecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pack_id: str
    state: RecoveryState
    recovered_scenes: tuple[str, ...] = ()
    waiting_scenes: tuple[str, ...] = ()
    resubmit_scenes: tuple[str, ...] = ()
    review_scenes: tuple[str, ...] = ()


class RecoveryComfyClient(Protocol):
    async def get_history(self, prompt_id: str) -> ComfyExecutionResult | None: ...

    async def queue_snapshot(self) -> dict[str, Any]: ...


class RecoveryManager:
    def __init__(
        self,
        database: Database,
        runtime: RuntimeStore,
        attempts: GenerationAttemptRepository,
        comfy: RecoveryComfyClient,
        telemetry: TelemetryRepository,
    ) -> None:
        self._database = database
        self._runtime = runtime
        self._attempts = attempts
        self._comfy = comfy
        self._telemetry = telemetry
        self._reviews = ReviewQueueRepository(database)

    async def recover_pack(self, pack_id: str) -> RecoveryResult:
        with self._database.session() as session:
            pack = session.get(PackRow, pack_id)
            if pack is None:
                raise KeyError(f"unknown pack: {pack_id}")
            pack_state = PackState(pack.state)
            if pack_state not in {PackState.GENERATING, PackState.EVALUATING}:
                return RecoveryResult(pack_id=pack_id, state=RecoveryState.NOOP)

            scene_rows = session.query(SceneRow).filter(SceneRow.pack_id == pack_id).all()
            scene_states = {row.id: SceneState(row.state) for row in scene_rows}

        recovered: list[str] = []
        waiting: list[str] = []
        resubmit: list[str] = []
        review: list[str] = []

        for scene_id, scene_state in sorted(scene_states.items()):
            if scene_state is not SceneState.GENERATING:
                continue

            attempt = self._attempts.latest_for_scene(scene_id)
            if attempt is None:
                # A missing receipt is not evidence that /prompt was never sent.
                # Never create a replacement GPU job from an ambiguous crash.
                self._mark_scene_for_review(
                    scene_id,
                    reason="generating scene has no persisted attempt; "
                    "original GPU submission cannot be ruled out",
                    attempt_id=None,
                )
                review.append(scene_id)
                continue

            prompt_id = self._prompt_id(attempt.provenance_json)
            if prompt_id is None:
                self._attempts.record_backend_status(
                    attempt.id,
                    status="uncertain",
                    error={"reason": "ComfyUI submission outcome unknown: no prompt ID"},
                )
                self._mark_scene_for_review(
                    scene_id,
                    reason="ComfyUI submission outcome unknown after restart: "
                    "persisted attempt has no prompt ID",
                    attempt_id=attempt.id,
                )
                review.append(scene_id)
                continue

            try:
                history = await self._comfy.get_history(prompt_id)
            except ComfyUIError as exc:
                self._telemetry.record(
                    "recovery.backend_unavailable",
                    EventSeverity.WARNING,
                    {
                        "pack_id": pack_id,
                        "scene_id": scene_id,
                        "attempt_id": attempt.id,
                        "prompt_id": prompt_id,
                        "kind": exc.kind.value,
                        "retryable": exc.retryable,
                    },
                )
                return RecoveryResult(
                    pack_id=pack_id,
                    state=RecoveryState.BACKEND_UNAVAILABLE,
                    recovered_scenes=tuple(recovered),
                    waiting_scenes=tuple(waiting),
                    resubmit_scenes=tuple(resubmit),
                    review_scenes=tuple(review),
                )

            if history is not None and history.completed:
                self._record_completed(scene_id, attempt.id, history)
                recovered.append(scene_id)
                continue

            try:
                queue = await self._comfy.queue_snapshot()
            except ComfyUIError as exc:
                self._telemetry.record(
                    "recovery.backend_unavailable",
                    EventSeverity.WARNING,
                    {
                        "pack_id": pack_id,
                        "scene_id": scene_id,
                        "attempt_id": attempt.id,
                        "prompt_id": prompt_id,
                        "kind": exc.kind.value,
                        "retryable": exc.retryable,
                    },
                )
                return RecoveryResult(
                    pack_id=pack_id,
                    state=RecoveryState.BACKEND_UNAVAILABLE,
                    recovered_scenes=tuple(recovered),
                    waiting_scenes=tuple(waiting),
                    resubmit_scenes=tuple(resubmit),
                    review_scenes=tuple(review),
                )

            if self._queue_contains_prompt(queue, prompt_id):
                self._attempts.record_backend_status(attempt.id, status="running")
                waiting.append(scene_id)
                continue

            # History can be evicted and queues can be emptied even after
            # /prompt succeeded. Absence cannot authorize a duplicate render.
            self._attempts.record_backend_status(
                attempt.id,
                status="uncertain",
                error={
                    "reason": "prompt absent from ComfyUI history and queue",
                    "prompt_id": prompt_id,
                },
            )
            self._mark_scene_for_review(
                scene_id,
                reason="previously accepted ComfyUI prompt is missing from "
                "history and queue; reconcile before any new GPU work",
                attempt_id=attempt.id,
            )
            review.append(scene_id)

        if waiting:
            state = RecoveryState.WAITING
        elif review:
            state = RecoveryState.REVIEW
        elif resubmit:
            state = RecoveryState.RESUBMIT
        elif recovered:
            state = RecoveryState.RECOVERED
        else:
            state = RecoveryState.NOOP

        self._telemetry.record(
            "recovery.pack_checked",
            EventSeverity.INFO,
            {
                "pack_id": pack_id,
                "state": state.value,
                "recovered": recovered,
                "waiting": waiting,
                "resubmit": resubmit,
                "review": review,
            },
        )
        return RecoveryResult(
            pack_id=pack_id,
            state=state,
            recovered_scenes=tuple(recovered),
            waiting_scenes=tuple(waiting),
            resubmit_scenes=tuple(resubmit),
            review_scenes=tuple(review),
        )

    def _record_completed(
        self,
        scene_id: str,
        attempt_id: str,
        history: ComfyExecutionResult,
    ) -> None:
        outputs = [item.model_dump(mode="json") for item in history.outputs]
        self._attempts.patch_provenance(
            attempt_id,
            {
                "comfy_prompt_id": history.prompt_id,
                "comfy_outputs": outputs,
                "recovered_from_history": True,
            },
        )
        self._attempts.record_backend_status(attempt_id, status="completed")
        self._runtime.transition_scene(
            scene_id,
            SceneState.EVALUATING,
            payload_patch={"recovered_attempt_id": attempt_id},
        )

    def _mark_scene_for_review(
        self,
        scene_id: str,
        *,
        reason: str,
        attempt_id: str | None,
    ) -> None:
        # Persist the review first. If the controller crashes before the
        # scene transition, the next recovery will find the existing item.
        if self._reviews.find_open(subject_type="scene", subject_id=scene_id) is None:
            self._reviews.enqueue(
                subject_type="scene",
                subject_id=scene_id,
                reason=reason,
                payload={"attempt_id": attempt_id},
            )
        self._runtime.transition_scene(
            scene_id,
            SceneState.REVIEW,
            payload_patch={
                "recovery_reason": reason,
                "recovery_attempt_id": attempt_id,
            },
        )
        self._telemetry.record(
            "recovery.scene_submission_uncertain",
            EventSeverity.WARNING,
            {"scene_id": scene_id, "attempt_id": attempt_id, "reason": reason},
        )

    @staticmethod
    def _prompt_id(provenance: Mapping[str, Any]) -> str | None:
        value = provenance.get("comfy_prompt_id")
        return value if isinstance(value, str) and value else None

    @classmethod
    def _queue_contains_prompt(cls, value: object, prompt_id: str) -> bool:
        if isinstance(value, str):
            return value == prompt_id
        if isinstance(value, Mapping):
            return any(
                cls._queue_contains_prompt(child, prompt_id)
                for child in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(cls._queue_contains_prompt(child, prompt_id) for child in value)
        return False
