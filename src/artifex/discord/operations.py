from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from artifex.characters import CharacterRegistry
from artifex.db import Database
from artifex.db.models import ConceptRow, PackRow, SceneRow
from artifex.discord.models import CommandName, CommandRequest, CommandResponse
from artifex.domain import AgentState, PackState, SceneState
from artifex.policy import PolicyDecisionRepository
from artifex.review import ReviewItem, ReviewQueueRepository, ReviewState
from artifex.runtime import RuntimeStore
from artifex.scheduler import Scheduler
from artifex.series import SeriesRepository


class ArtifexRemoteOperations:
    def __init__(
        self,
        database: Database,
        runtime: RuntimeStore,
        scheduler: Scheduler,
        reviews: ReviewQueueRepository,
        characters: CharacterRegistry,
        series: SeriesRepository,
        policy_decisions: PolicyDecisionRepository,
    ) -> None:
        self._database = database
        self._runtime = runtime
        self._scheduler = scheduler
        self._reviews = reviews
        self._characters = characters
        self._series = series
        self._policy_decisions = policy_decisions

    def execute(self, request: CommandRequest) -> CommandResponse:
        match request.name:
            case CommandName.STATUS:
                return self._status()
            case CommandName.PAUSE:
                return self._pause()
            case CommandName.RESUME:
                return self._resume()
            case CommandName.CURRENT:
                return self._current()
            case CommandName.QUEUE:
                return self._queue()
            case CommandName.RECENT:
                return self._recent()
            case CommandName.APPROVE:
                return self._approve(self._one_arg(request, "review id"))
            case CommandName.REJECT:
                return self._reject(self._one_arg(request, "review id"))
            case CommandName.RETRY:
                return self._retry(self._one_arg(request, "review or scene id"))
            case CommandName.SKIP:
                return self._skip(self._one_arg(request, "review or subject id"))
            case CommandName.ALTERNATE:
                return self._alternate(self._one_arg(request, "review id"))
            case CommandName.NEXT:
                return self._next_review()
            case CommandName.CHARACTER:
                return self._character(self._one_arg(request, "character id"))
            case CommandName.SERIES:
                return self._series_info(self._one_arg(request, "series id"))
            case CommandName.DETAILS:
                return self._details(self._one_arg(request, "review id"))

    @staticmethod
    def _one_arg(request: CommandRequest, label: str) -> str:
        if len(request.args) != 1 or not request.args[0].strip():
            raise ValueError(f"{request.name.value} requires exactly one {label}")
        return request.args[0].strip()

    def _status(self) -> CommandResponse:
        inventory = self._scheduler.inventory()
        reviews = self._reviews.list_open(limit=100)
        state = self._runtime.get_agent_state()
        message = (
            f"Artifex: {state.value} | ideas={inventory.ideas} "
            f"planned={inventory.planned} completed={inventory.completed_available} "
            f"reviews={len(reviews)}"
        )
        return CommandResponse(
            ok=True,
            message=message,
            data={
                "agent_state": state.value,
                "ideas": inventory.ideas,
                "planned": inventory.planned,
                "completed_available": inventory.completed_available,
                "open_reviews": len(reviews),
            },
        )

    def _pause(self) -> CommandResponse:
        current = self._runtime.get_agent_state()
        if current is AgentState.PAUSED:
            return CommandResponse(ok=True, message="Artifex is already paused.")
        self._runtime.set_agent_state(
            AgentState.PAUSED,
            expected=current,
            reason="discord_operator_pause",
        )
        return CommandResponse(ok=True, message="Artifex paused.")

    def _resume(self) -> CommandResponse:
        current = self._runtime.get_agent_state()
        if current is AgentState.RUNNING:
            return CommandResponse(ok=True, message="Artifex is already running.")
        self._runtime.set_agent_state(
            AgentState.RUNNING,
            expected=current,
            reason="discord_operator_resume",
        )
        return CommandResponse(ok=True, message="Artifex resumed.")

    def _current(self) -> CommandResponse:
        active_states = (
            PackState.POLICY_CHECK.value,
            PackState.GENERATING.value,
            PackState.EVALUATING.value,
            PackState.REVIEW.value,
        )
        with self._database.session() as session:
            pack = session.scalar(
                select(PackRow)
                .where(PackRow.state.in_(active_states))
                .order_by(PackRow.updated_at.asc(), PackRow.id.asc())
                .limit(1)
            )
            if pack is None:
                return CommandResponse(ok=True, message="No active Pack.")
            scene_rows = session.scalars(
                select(SceneRow)
                .where(SceneRow.pack_id == pack.id)
                .order_by(SceneRow.ordinal.asc())
            ).all()
            scenes = [
                {
                    "id": scene.id,
                    "ordinal": scene.ordinal,
                    "state": scene.state,
                    "tier": scene.publication_tier,
                }
                for scene in scene_rows
            ]
            message = (
                f"Current Pack {pack.id}: {pack.state} "
                f"({len(scene_rows)} scene(s), format={pack.format_type})"
            )
            return CommandResponse(
                ok=True,
                message=message,
                data={
                    "pack_id": pack.id,
                    "state": pack.state,
                    "format": pack.format_type,
                    "scenes": scenes,
                },
            )

    def _queue(self) -> CommandResponse:
        terminal = {
            PackState.FINALIZED.value,
            PackState.FAILED.value,
            PackState.BLOCKED.value,
        }
        with self._database.session() as session:
            rows = session.scalars(
                select(PackRow)
                .where(PackRow.state.not_in(terminal))
                .order_by(PackRow.created_at.asc(), PackRow.id.asc())
                .limit(20)
            ).all()
            items = [
                {"id": row.id, "state": row.state, "format": row.format_type}
                for row in rows
            ]
        if not items:
            return CommandResponse(ok=True, message="Pack queue is empty.", data={"packs": []})
        message = "Queue: " + ", ".join(
            f"{item['id']}({item['state']})" for item in items
        )
        return CommandResponse(ok=True, message=message, data={"packs": items})

    def _recent(self) -> CommandResponse:
        terminal = {
            PackState.FINALIZED.value,
            PackState.FAILED.value,
            PackState.BLOCKED.value,
        }
        with self._database.session() as session:
            rows = session.scalars(
                select(PackRow)
                .where(PackRow.state.in_(terminal))
                .order_by(PackRow.updated_at.desc(), PackRow.id.desc())
                .limit(10)
            ).all()
            items = [
                {
                    "id": row.id,
                    "state": row.state,
                    "format": row.format_type,
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in rows
            ]
        if not items:
            return CommandResponse(ok=True, message="No recent completed Pack.")
        message = "Recent: " + ", ".join(
            f"{item['id']}({item['state']})" for item in items
        )
        return CommandResponse(ok=True, message=message, data={"packs": items})

    def _approve(self, review_id: str) -> CommandResponse:
        item = self._reviews.require(review_id)
        self._require_open(item)
        self._resolve_policy(item, approved=True)
        self._accept_subject(item)
        resolved = self._reviews.resolve(review_id, ReviewState.APPROVED)
        return CommandResponse(
            ok=True,
            message=f"Approved review {review_id} ({resolved.subject_type}:{resolved.subject_id}).",
        )

    def _reject(self, review_id: str) -> CommandResponse:
        item = self._reviews.require(review_id)
        self._require_open(item)
        self._resolve_policy(item, approved=False)
        self._reject_subject(item)
        resolved = self._reviews.resolve(review_id, ReviewState.REJECTED)
        return CommandResponse(
            ok=True,
            message=f"Rejected review {review_id} ({resolved.subject_type}:{resolved.subject_id}).",
        )

    def _retry(self, target: str) -> CommandResponse:
        item = self._reviews.get(target)
        subject_type = item.subject_type if item is not None else "scene"
        subject_id = item.subject_id if item is not None else target

        if subject_type == "scene":
            self._retry_scene(subject_id)
        elif subject_type == "pack":
            self._retry_pack(subject_id)
        else:
            raise ValueError(f"retry unsupported for subject type: {subject_type}")

        if item is not None:
            self._reviews.resolve(
                item.id,
                ReviewState.RETRY,
                payload_patch={"operator_retry_at": datetime.now(UTC).isoformat()},
            )
        return CommandResponse(ok=True, message=f"Retry requested for {subject_type} {subject_id}.")

    def _skip(self, target: str) -> CommandResponse:
        item = self._reviews.get(target)
        subject_type = item.subject_type if item is not None else "scene"
        subject_id = item.subject_id if item is not None else target

        if subject_type == "scene":
            self._skip_scene(subject_id)
        elif subject_type == "pack":
            self._skip_pack(subject_id)
        else:
            raise ValueError(f"skip unsupported for subject type: {subject_type}")

        if item is not None:
            self._reviews.resolve(item.id, ReviewState.SKIPPED)
        return CommandResponse(ok=True, message=f"Skipped {subject_type} {subject_id}.")

    def _alternate(self, review_id: str) -> CommandResponse:
        item = self._reviews.require(review_id)
        self._require_open(item)

        if item.subject_type == "pack":
            pack_id = item.subject_id
        elif item.subject_type == "scene":
            with self._database.session() as session:
                scene = session.get(SceneRow, item.subject_id)
                if scene is None:
                    raise KeyError(f"unknown scene: {item.subject_id}")
                pack_id = scene.pack_id
        else:
            raise ValueError(
                f"alternate idea unsupported for subject type: {item.subject_type}"
            )

        self._skip_pack(pack_id)
        self._reviews.resolve(
            review_id,
            ReviewState.SKIPPED,
            payload_patch={"alternate_idea_requested": True},
        )
        return CommandResponse(
            ok=True,
            message=f"Alternate idea requested; Pack {pack_id} retired.",
        )

    def _next_review(self) -> CommandResponse:
        item = self._reviews.next_open()
        if item is None:
            return CommandResponse(ok=True, message="No pending review.", data={})
        return CommandResponse(
            ok=True,
            message=(
                f"Next review {item.id}: {item.subject_type}:{item.subject_id} — "
                f"{item.reason}"
            ),
            data=item.model_dump(mode="json"),
        )

    def _details(self, review_id: str) -> CommandResponse:
        item = self._reviews.require(review_id)
        return CommandResponse(
            ok=True,
            message=(
                f"Review {item.id} [{item.state.value}] "
                f"{item.subject_type}:{item.subject_id} — {item.reason}"
            ),
            data=item.model_dump(mode="json"),
        )

    def _character(self, character_id: str) -> CommandResponse:
        profile = self._characters.require(character_id)
        return CommandResponse(
            ok=True,
            message=(
                f"{profile.display_name} ({profile.id}) enabled={profile.enabled} "
                f"readiness={profile.readiness:.2f} LoRA={profile.lora_policy.value}"
            ),
            data=profile.model_dump(mode="json"),
        )

    def _series_info(self, series_id: str) -> CommandResponse:
        profile = self._series.require(series_id)
        return CommandResponse(
            ok=True,
            message=(
                f"Series {profile.title} ({profile.id}) status={profile.status.value} "
                f"episode={profile.current_episode} hooks={len(profile.unresolved_hooks)}"
            ),
            data=profile.model_dump(mode="json"),
        )

    def _resolve_policy(self, item: ReviewItem, *, approved: bool) -> None:
        raw = item.payload.get("policy_decision_id")
        if raw is None:
            return
        if not isinstance(raw, str) or not raw:
            raise TypeError("review policy_decision_id must be a non-empty string")
        self._policy_decisions.resolve_review(raw, approved=approved)

    def _accept_subject(self, item: ReviewItem) -> None:
        if item.subject_type == "scene":
            with self._database.session() as session:
                scene = session.get(SceneRow, item.subject_id)
                if scene is None:
                    raise KeyError(f"unknown scene: {item.subject_id}")
                state = SceneState(scene.state)
                attempt_id = self._attempt_id(item, scene.selected_attempt_id)
            if state is SceneState.REVIEW:
                self._runtime.select_scene_attempt(
                    item.subject_id,
                    attempt_id,
                    SceneState.ACCEPTED,
                )
        elif item.subject_type == "pack":
            with self._database.session() as session:
                pack = session.get(PackRow, item.subject_id)
                if pack is None:
                    raise KeyError(f"unknown pack: {item.subject_id}")
                state = PackState(pack.state)
            if state is PackState.REVIEW:
                self._runtime.transition_pack(item.subject_id, PackState.FINALIZED)

    def _reject_subject(self, item: ReviewItem) -> None:
        if item.subject_type == "scene":
            with self._database.session() as session:
                scene = session.get(SceneRow, item.subject_id)
                if scene is None:
                    raise KeyError(f"unknown scene: {item.subject_id}")
                state = SceneState(scene.state)
                attempt_id = self._attempt_id(item, scene.selected_attempt_id)
            if state in {SceneState.REVIEW, SceneState.EVALUATING}:
                self._runtime.select_scene_attempt(
                    item.subject_id,
                    attempt_id,
                    SceneState.REJECTED,
                )
        elif item.subject_type == "pack":
            with self._database.session() as session:
                pack = session.get(PackRow, item.subject_id)
                if pack is None:
                    raise KeyError(f"unknown pack: {item.subject_id}")
                state = PackState(pack.state)
            if state is PackState.REVIEW:
                self._runtime.transition_pack(item.subject_id, PackState.FAILED)

    @staticmethod
    def _attempt_id(item: ReviewItem, fallback: str | None) -> str | None:
        raw = item.payload.get("attempt_id")
        if raw is None:
            return fallback
        if not isinstance(raw, str) or not raw:
            raise TypeError("review attempt_id must be a non-empty string")
        return raw

    def _retry_scene(self, scene_id: str) -> None:
        with self._database.session() as session:
            row = session.get(SceneRow, scene_id)
            if row is None:
                raise KeyError(f"unknown scene: {scene_id}")
            state = SceneState(row.state)

        if state is SceneState.READY:
            return
        if state is SceneState.REVIEW:
            self._runtime.transition_scene(
                scene_id,
                SceneState.READY,
                payload_patch={"operator_retry": True},
            )
            return
        if state is SceneState.REJECTED:
            self._runtime.transition_scene(scene_id, SceneState.REVIEW)
            self._runtime.transition_scene(
                scene_id,
                SceneState.READY,
                payload_patch={"operator_retry": True},
            )
            return
        if state in {SceneState.FAILED, SceneState.GENERATING, SceneState.EVALUATING}:
            self._runtime.transition_scene(scene_id, SceneState.RETRY)
            self._runtime.transition_scene(
                scene_id,
                SceneState.READY,
                payload_patch={"operator_retry": True},
            )
            return
        if state is SceneState.RETRY:
            self._runtime.transition_scene(
                scene_id,
                SceneState.READY,
                payload_patch={"operator_retry": True},
            )
            return
        raise ValueError(f"scene cannot be retried from {state.value}")

    def _retry_pack(self, pack_id: str) -> None:
        with self._database.session() as session:
            row = session.get(PackRow, pack_id)
            if row is None:
                raise KeyError(f"unknown pack: {pack_id}")
            state = PackState(row.state)

        if state in {PackState.FAILED, PackState.BLOCKED}:
            self._runtime.transition_pack(
                pack_id,
                PackState.PLANNED,
                checkpoint_patch={"operator_retry": True},
            )
        elif state is PackState.REVIEW:
            self._runtime.transition_pack(
                pack_id,
                PackState.GENERATING,
                checkpoint_patch={"operator_retry": True},
            )
        elif state in {PackState.PLANNED, PackState.POLICY_CHECK}:
            return
        else:
            raise ValueError(f"pack cannot be retried from {state.value}")

    def _skip_scene(self, scene_id: str) -> None:
        with self._database.session() as session:
            row = session.get(SceneRow, scene_id)
            if row is None:
                raise KeyError(f"unknown scene: {scene_id}")
            state = SceneState(row.state)
            attempt_id = row.selected_attempt_id

        if state in {SceneState.REVIEW, SceneState.EVALUATING}:
            self._runtime.select_scene_attempt(
                scene_id,
                attempt_id,
                SceneState.REJECTED,
            )
        elif state in {
            SceneState.PLANNED,
            SceneState.READY,
            SceneState.GENERATING,
            SceneState.RETRY,
        }:
            self._runtime.transition_scene(scene_id, SceneState.FAILED)
        elif state not in {SceneState.REJECTED, SceneState.FAILED}:
            raise ValueError(f"scene cannot be skipped from {state.value}")

    def _skip_pack(self, pack_id: str) -> None:
        with self._database.session() as session:
            row = session.get(PackRow, pack_id)
            if row is None:
                raise KeyError(f"unknown pack: {pack_id}")
            state = PackState(row.state)
            concept_id = row.concept_id

        if state in {
            PackState.IDEA,
            PackState.PLANNED,
            PackState.POLICY_CHECK,
            PackState.GENERATING,
            PackState.EVALUATING,
            PackState.REVIEW,
        }:
            self._runtime.transition_pack(pack_id, PackState.FAILED)
        elif state not in {PackState.FAILED, PackState.BLOCKED}:
            raise ValueError(f"pack cannot be skipped from {state.value}")

        if concept_id is not None:
            with self._database.session() as session:
                concept = session.get(ConceptRow, concept_id)
                if concept is not None:
                    concept.status = "rejected_by_operator"

    @staticmethod
    def _require_open(item: ReviewItem) -> None:
        if item.state is not ReviewState.OPEN:
            raise ValueError(f"review item is already resolved: {item.id}")
