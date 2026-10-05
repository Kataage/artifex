from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow, SceneRow, SettingRow
from artifex.domain import AgentState, PackState, SceneState
from artifex.runtime.transitions import (
    validate_agent_transition,
    validate_pack_transition,
    validate_scene_transition,
)

_AGENT_STATE_KEY = "runtime.agent"


class RuntimeStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    def get_agent_state(self) -> AgentState:
        with self._database.session() as session:
            row = session.get(SettingRow, _AGENT_STATE_KEY)
            if row is None:
                return AgentState.STOPPED
            raw = row.value_json
            if isinstance(raw, dict):
                value = raw.get("state", AgentState.STOPPED.value)
            else:
                value = raw
            return AgentState(str(value))

    def get_agent_reason(self) -> str | None:
        with self._database.session() as session:
            row = session.get(SettingRow, _AGENT_STATE_KEY)
            if row is None or not isinstance(row.value_json, dict):
                return None
            raw = row.value_json.get("reason")
            return str(raw) if raw is not None else None

    def set_agent_state(
        self,
        target: AgentState,
        *,
        expected: AgentState | None = None,
        reason: str | None = None,
    ) -> AgentState:
        now = datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(SettingRow, _AGENT_STATE_KEY)
            current = self._read_agent_row(row)

            if expected is not None and current is not expected:
                raise RuntimeError(
                    f"agent state changed concurrently: expected {expected.value}, got {current.value}"
                )
            validate_agent_transition(current, target)

            payload: dict[str, Any] = {
                "state": target.value,
                "reason": reason,
                "updated_at": now.isoformat(),
            }
            if row is None:
                row = SettingRow(
                    key=_AGENT_STATE_KEY,
                    value_json=payload,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.value_json = payload
                row.updated_at = now
        return target

    def reconcile_process_start(self) -> AgentState:
        """Convert stale process-owned states after an unclean process exit.

        Paused and blocked are operator/health intent and remain intact.
        Running/degraded/starting/stopping are process-owned and become STARTING
        so the new process can recover persisted work before entering RUNNING.
        """

        now = datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(SettingRow, _AGENT_STATE_KEY)
            current = self._read_agent_row(row)
            if current in {AgentState.PAUSED, AgentState.BLOCKED}:
                return current
            if current is AgentState.STOPPED:
                target = AgentState.STARTING
            else:
                target = AgentState.STARTING

            payload: dict[str, Any] = {
                "state": target.value,
                "reason": "process_start_reconciliation",
                "updated_at": now.isoformat(),
            }
            if row is None:
                session.add(
                    SettingRow(
                        key=_AGENT_STATE_KEY,
                        value_json=payload,
                        updated_at=now,
                    )
                )
            else:
                row.value_json = payload
                row.updated_at = now
            return target

    def transition_pack(
        self,
        pack_id: str,
        target: PackState,
        *,
        checkpoint_patch: dict[str, Any] | None = None,
    ) -> PackState:
        now = datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(PackRow, pack_id)
            if row is None:
                raise KeyError(f"unknown pack: {pack_id}")
            current = PackState(row.state)
            validate_pack_transition(current, target)
            row.state = target.value
            row.updated_at = now
            if checkpoint_patch:
                row.checkpoint_json = {**row.checkpoint_json, **checkpoint_patch}
        return target

    def transition_scene(
        self,
        scene_id: str,
        target: SceneState,
        *,
        payload_patch: dict[str, Any] | None = None,
    ) -> SceneState:
        with self._database.session() as session:
            row = session.get(SceneRow, scene_id)
            if row is None:
                raise KeyError(f"unknown scene: {scene_id}")
            current = SceneState(row.state)
            validate_scene_transition(current, target)
            row.state = target.value
            if payload_patch:
                row.payload_json = {**row.payload_json, **payload_patch}
        return target

    def select_scene_attempt(
        self,
        scene_id: str,
        attempt_id: str | None,
        target: SceneState,
    ) -> SceneState:
        if target not in {
            SceneState.ACCEPTED,
            SceneState.REVIEW,
            SceneState.REJECTED,
        }:
            raise ValueError("scene selection target must be accepted, review, or rejected")

        with self._database.session() as session:
            scene = session.get(SceneRow, scene_id)
            if scene is None:
                raise KeyError(f"unknown scene: {scene_id}")
            if attempt_id is not None:
                attempt = session.get(GenerationAttemptRow, attempt_id)
                if attempt is None:
                    raise KeyError(f"unknown generation attempt: {attempt_id}")
                if attempt.scene_id != scene_id:
                    raise ValueError("selected attempt belongs to a different scene")

            current = SceneState(scene.state)
            validate_scene_transition(current, target)
            scene.state = target.value
            scene.selected_attempt_id = attempt_id
        return target

    def recoverable_pack_ids(self) -> tuple[str, ...]:
        recoverable = {
            PackState.GENERATING.value,
            PackState.EVALUATING.value,
        }
        with self._database.session() as session:
            ids = session.scalars(
                select(PackRow.id)
                .where(PackRow.state.in_(recoverable))
                .order_by(PackRow.updated_at.asc(), PackRow.id.asc())
            ).all()
        return tuple(ids)

    @staticmethod
    def _read_agent_row(row: SettingRow | None) -> AgentState:
        if row is None:
            return AgentState.STOPPED
        raw = row.value_json
        if isinstance(raw, dict):
            raw = raw.get("state", AgentState.STOPPED.value)
        return AgentState(str(raw))
