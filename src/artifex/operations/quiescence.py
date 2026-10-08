from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from artifex.comfy import ComfyUIClient
from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow
from artifex.domain import AgentState, PackState
from artifex.runtime import RuntimeStore

QuiescenceState = Literal[
    "not_paused", "draining", "queue_unverifiable", "observed_idle", "observed_quiescent"
]


class QuiescenceSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: AgentState
    active_packs: int = Field(ge=0)
    unfinished_attempts: int = Field(ge=0)
    queue_running: int | None = None
    queue_pending: int | None = None
    error: str | None = None

    @property
    def idle(self) -> bool:
        return (
            self.state is AgentState.PAUSED
            and self.active_packs == 0
            and self.unfinished_attempts == 0
            and self.queue_running == 0
            and self.queue_pending == 0
            and self.error is None
        )


class ControllerQuiescenceReport(BaseModel):
    """Observed drain, not a safe authorization to kill any ComfyUI process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: QuiescenceState
    ready: bool
    paused_by_command: bool
    samples: int = Field(ge=1)
    consecutive_idle_samples: int = Field(ge=0)
    last: QuiescenceSample
    next_actions: tuple[str, ...]
    restart_authorized: bool = False
    production_qualified: bool = False


def _active_work(database: Database) -> tuple[int, int]:
    """Include prepared/generating Packs and submissions not yet terminal.

    A stale persisted in-flight Pack is intentionally blocking; recovering or
    reviewing it must be explicit rather than blindly assuming GPU idleness.
    """
    with database.session() as session:
        packs = session.scalar(
            select(func.count()).select_from(PackRow).where(
                PackRow.state.in_(
                    (PackState.POLICY_CHECK.value,
                     PackState.GENERATING.value,
                     PackState.EVALUATING.value)
                )
            )
        )
        attempts = session.scalar(
            select(func.count()).select_from(GenerationAttemptRow).where(
                GenerationAttemptRow.backend_status.in_(
                    ("created", "queued", "submitted", "running", "pending")
                )
            )
        )
    return int(packs or 0), int(attempts or 0)


def _queue_counts(raw: object) -> tuple[int | None, int | None]:
    if not isinstance(raw, dict):
        return None, None
    running = raw.get("queue_running")
    pending = raw.get("queue_pending")
    if not isinstance(running, list) or not isinstance(pending, list):
        return None, None
    return len(running), len(pending)


async def quiesce_controller(
    database: Database,
    runtime: RuntimeStore,
    comfy: ComfyUIClient,
    *,
    apply: bool = False,
    wait_seconds: float = 0,
    poll_seconds: float = 5,
    required_idle_samples: int = 2,
    work_fn: Callable[[Database], tuple[int, int]] = _active_work,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ControllerQuiescenceReport:
    """Pause scheduler (opt-in), then verify *observed* controller drain.

    The runtime pause never interrupts a submitted GPU job. Queue snapshots
    are not atomic with new submissions by external clients; therefore even
    sustained empty samples NEVER authorize a renderer restart.
    """
    if not 0 <= wait_seconds <= 600:
        raise ValueError("wait_seconds must be within [0, 600]")
    if not 0 < poll_seconds <= 60:
        raise ValueError("poll_seconds must be within (0, 60]")
    if not 1 <= required_idle_samples <= 10:
        raise ValueError("required_idle_samples must be within [1, 10]")
    paused_by_command = False
    if apply:
        current = runtime.get_agent_state()
        if current is AgentState.RUNNING:
            runtime.set_agent_state(
                AgentState.PAUSED, expected=AgentState.RUNNING,
                reason="controller_quiescence",
            )
            paused_by_command = True
        elif current is not AgentState.PAUSED:
            raise ValueError(
                f"Cannot quiesce agent in {current.value} state; "
                "only running or already paused is supported"
            )

    loop = asyncio.get_running_loop()
    start = loop.time()
    streak = 0
    sample_count = 0
    last: QuiescenceSample | None = None
    while True:
        sample_count += 1
        state = runtime.get_agent_state()
        active_packs, unfinished_attempts = await asyncio.to_thread(work_fn, database)
        running: int | None = None
        pending: int | None = None
        error: str | None = None
        try:
            running, pending = _queue_counts(await comfy.queue_snapshot())
        except (httpx.HTTPError, OSError, ValueError, TypeError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        last = QuiescenceSample(
            state=state, active_packs=active_packs,
            unfinished_attempts=unfinished_attempts,
            queue_running=running, queue_pending=pending, error=error,
        )
        streak = streak + 1 if last.idle else 0
        if streak >= required_idle_samples:
            break
        remaining = wait_seconds - (loop.time() - start)
        if remaining <= 0:
            break
        await sleep_fn(min(poll_seconds, remaining))

    assert last is not None
    if last.state is not AgentState.PAUSED:
        status: QuiescenceState = "not_paused"
        actions = ("pause_artifex_controller_then_repeat_drain",)
    elif last.error or last.queue_running is None or last.queue_pending is None:
        status = "queue_unverifiable"
        actions = ("check_comfyui_queue_connectivity_and_schema",)
    elif last.active_packs or last.unfinished_attempts or last.queue_running or last.queue_pending:
        status = "draining"
        actions = (
            "let_current_artifex_packs_and_external_gpu_jobs_complete",
            "investigate_persisted_stuck_generation_before_any_restart",
        )
    elif streak < required_idle_samples:
        status = "observed_idle"
        actions = ("repeat_drain_to_confirm_consecutive_idle_observations",)
    else:
        status = "observed_quiescent"
        actions = (
            "external_comfyui_submitters_remain_unfenced",
            "do_not_restart_until_exclusive_renderer_admission_is_enforced",
        )
    return ControllerQuiescenceReport(
        status=status,
        ready=status == "observed_quiescent",
        paused_by_command=paused_by_command,
        samples=sample_count,
        consecutive_idle_samples=streak,
        last=last,
        next_actions=actions,
    )
