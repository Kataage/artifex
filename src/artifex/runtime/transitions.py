from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TypeVar

from artifex.domain import AgentState, PackState, SceneState

StateT = TypeVar("StateT", bound=StrEnum)


class InvalidTransition(ValueError):
    def __init__(self, entity: str, current: StrEnum, target: StrEnum) -> None:
        super().__init__(f"invalid {entity} transition: {current.value} -> {target.value}")
        self.entity = entity
        self.current = current
        self.target = target


AGENT_TRANSITIONS: Mapping[AgentState, frozenset[AgentState]] = {
    AgentState.STOPPED: frozenset({AgentState.STARTING}),
    AgentState.STARTING: frozenset(
        {
            AgentState.RUNNING,
            AgentState.PAUSED,
            AgentState.DEGRADED,
            AgentState.BLOCKED,
            AgentState.STOPPING,
            AgentState.STOPPED,
        }
    ),
    AgentState.RUNNING: frozenset(
        {
            AgentState.PAUSED,
            AgentState.DEGRADED,
            AgentState.BLOCKED,
            AgentState.STOPPING,
        }
    ),
    AgentState.PAUSED: frozenset(
        {AgentState.RUNNING, AgentState.BLOCKED, AgentState.STOPPING}
    ),
    AgentState.DEGRADED: frozenset(
        {
            AgentState.RUNNING,
            AgentState.PAUSED,
            AgentState.BLOCKED,
            AgentState.STOPPING,
        }
    ),
    AgentState.BLOCKED: frozenset(
        {AgentState.RUNNING, AgentState.PAUSED, AgentState.STOPPING}
    ),
    AgentState.STOPPING: frozenset({AgentState.STOPPED}),
}

PACK_TRANSITIONS: Mapping[PackState, frozenset[PackState]] = {
    PackState.IDEA: frozenset({PackState.PLANNED, PackState.BLOCKED, PackState.FAILED}),
    PackState.PLANNED: frozenset(
        {PackState.POLICY_CHECK, PackState.BLOCKED, PackState.FAILED}
    ),
    PackState.POLICY_CHECK: frozenset(
        {
            PackState.GENERATING,
            PackState.REVIEW,
            PackState.BLOCKED,
            PackState.FAILED,
        }
    ),
    PackState.GENERATING: frozenset(
        {
            PackState.EVALUATING,
            PackState.REVIEW,
            PackState.BLOCKED,
            PackState.FAILED,
        }
    ),
    PackState.EVALUATING: frozenset(
        {
            PackState.GENERATING,
            PackState.REVIEW,
            PackState.FINALIZED,
            PackState.BLOCKED,
            PackState.FAILED,
        }
    ),
    PackState.REVIEW: frozenset(
        {
            PackState.GENERATING,
            PackState.EVALUATING,
            PackState.FINALIZED,
            PackState.BLOCKED,
            PackState.FAILED,
        }
    ),
    PackState.FINALIZED: frozenset(),
    PackState.FAILED: frozenset({PackState.PLANNED, PackState.REVIEW}),
    PackState.BLOCKED: frozenset({PackState.PLANNED, PackState.REVIEW}),
}

SCENE_TRANSITIONS: Mapping[SceneState, frozenset[SceneState]] = {
    SceneState.PLANNED: frozenset(
        {SceneState.READY, SceneState.BLOCKED, SceneState.FAILED}
    ),
    SceneState.READY: frozenset(
        {SceneState.GENERATING, SceneState.BLOCKED, SceneState.FAILED}
    ),
    SceneState.GENERATING: frozenset(
        {
            SceneState.EVALUATING,
            SceneState.RETRY,
            SceneState.REVIEW,
            SceneState.BLOCKED,
            SceneState.FAILED,
        }
    ),
    SceneState.EVALUATING: frozenset(
        {
            SceneState.ACCEPTED,
            SceneState.REVIEW,
            SceneState.RETRY,
            SceneState.REJECTED,
            SceneState.BLOCKED,
            SceneState.FAILED,
        }
    ),
    SceneState.RETRY: frozenset(
        {SceneState.READY, SceneState.GENERATING, SceneState.FAILED}
    ),
    SceneState.REVIEW: frozenset(
        {
            SceneState.READY,
            SceneState.ACCEPTED,
            SceneState.REJECTED,
            SceneState.BLOCKED,
            SceneState.FAILED,
        }
    ),
    SceneState.ACCEPTED: frozenset(),
    SceneState.REJECTED: frozenset({SceneState.REVIEW}),
    SceneState.FAILED: frozenset({SceneState.RETRY, SceneState.REVIEW}),
    SceneState.BLOCKED: frozenset({SceneState.PLANNED, SceneState.REVIEW}),
}


def _validate(
    entity: str,
    current: StateT,
    target: StateT,
    transitions: Mapping[StateT, frozenset[StateT]],
) -> None:
    if current == target:
        return
    if target not in transitions[current]:
        raise InvalidTransition(entity, current, target)


def validate_agent_transition(current: AgentState, target: AgentState) -> None:
    _validate("agent", current, target, AGENT_TRANSITIONS)


def validate_pack_transition(current: PackState, target: PackState) -> None:
    _validate("pack", current, target, PACK_TRANSITIONS)


def validate_scene_transition(current: SceneState, target: SceneState) -> None:
    _validate("scene", current, target, SCENE_TRANSITIONS)
