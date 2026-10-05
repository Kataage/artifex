from __future__ import annotations

import pytest

from artifex.domain import AgentState, PackState, SceneState
from artifex.runtime.transitions import (
    InvalidTransition,
    validate_agent_transition,
    validate_pack_transition,
    validate_scene_transition,
)


def test_agent_transition_rejects_skipping_startup() -> None:
    validate_agent_transition(AgentState.STOPPED, AgentState.STARTING)
    with pytest.raises(InvalidTransition):
        validate_agent_transition(AgentState.STOPPED, AgentState.RUNNING)


def test_pack_finalized_is_terminal() -> None:
    with pytest.raises(InvalidTransition):
        validate_pack_transition(PackState.FINALIZED, PackState.PLANNED)


def test_scene_retry_can_return_to_ready() -> None:
    validate_scene_transition(SceneState.RETRY, SceneState.READY)
