from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.comfy import ComfyUIClient
from artifex.comfy.admission import ComfySubmissionFence
from artifex.db import Database
from artifex.domain import AgentState
from artifex.operations.quiescence import ControllerQuiescenceReport, quiesce_controller
from artifex.runtime import RuntimeStore


class SubmissionFenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["preview", "seal", "release"]
    state: Literal["open", "sealed", "blocked_on_drain", "released"]
    submission_blocked: bool
    admission_db_path: str
    drain: ControllerQuiescenceReport | None = None
    next_actions: tuple[str, ...]
    # Atomicity is PC-A Artifex-local. A direct network /prompt still bypasses.
    external_comfyui_clients_fenced: bool = False
    restart_authorized: bool = False
    production_qualified: bool = False


async def manage_submission_fence(
    database: Database,
    runtime: RuntimeStore,
    comfy: ComfyUIClient,
    fence: ComfySubmissionFence,
    *,
    apply: bool = False,
    release: bool = False,
    wait_seconds: float = 90,
    poll_seconds: float = 5,
) -> SubmissionFenceReport:
    """Preview, seal after drain, or explicitly release PC-A Artifex admissions.

    A sealed fence survives a controller crash/restart because its state is
    persisted in a dedicated SQLite DB and checked under the same write lock
    as the HTTP /prompt. Cannot authorize an uncontrolled ComfyUI restart.
    """
    if not 0 <= wait_seconds <= 600:
        raise ValueError("wait_seconds must be between 0 and 600")
    if not 0 < poll_seconds <= 60:
        raise ValueError("poll_seconds must be within (0, 60]")
    if release and not apply:
        raise ValueError("--release requires --apply")

    blocked = await asyncio.to_thread(fence.status)
    if not apply:
        return SubmissionFenceReport(
            mode="preview", state="sealed" if blocked else "open",
            submission_blocked=blocked, admission_db_path=str(fence.path),
            next_actions=(
                "review_fence_and_controller_pause_before_renderer_maintenance",
            ),
        )
    if release:
        if runtime.get_agent_state() is not AgentState.PAUSED:
            raise ValueError(
                "Releasing admission requires PAUSED controller; "
                "verify renderer recovery before resuming scheduling"
            )
        await fence.release()
        return SubmissionFenceReport(
            mode="release", state="released", submission_blocked=False,
            admission_db_path=str(fence.path),
            next_actions=(
                "verify_renderer_runtime_and_two_pc_deployment",
                "resume_controller_only_when_healthy",
            ),
        )

    if blocked:
        return SubmissionFenceReport(
            mode="seal", state="sealed", submission_blocked=True,
            admission_db_path=str(fence.path),
            next_actions=("admission_already_sealed; no_comfyui_restart_authorized",),
        )

    first = await quiesce_controller(
        database, runtime, comfy,
        apply=True, wait_seconds=wait_seconds, poll_seconds=poll_seconds,
        required_idle_samples=2,
    )
    if not first.ready:
        return SubmissionFenceReport(
            mode="seal", state="blocked_on_drain", submission_blocked=False,
            admission_db_path=str(fence.path), drain=first,
            next_actions=first.next_actions,
        )

    # The state/queue/DB recheck occurs UNDER the same SQLite transaction
    # guarding every Artifex POST. No compliant request can race this check.
    final_drain: ControllerQuiescenceReport = first

    async def validate() -> bool:
        nonlocal final_drain
        final_drain = await quiesce_controller(
            database, runtime, comfy,
            apply=False, wait_seconds=3, poll_seconds=0.25,
            required_idle_samples=2,
        )
        return final_drain.ready

    sealed = await fence.seal(validate)
    return SubmissionFenceReport(
        mode="seal",
        state="sealed" if sealed else "blocked_on_drain",
        submission_blocked=sealed, admission_db_path=str(fence.path),
        drain=final_drain,
        next_actions=(
            (
                "artifex_submissions_now_atomically_sealed",
                "external_direct_comfyui_clients_still_unfenced",
                "never_restart_renderer_without_external_admission_control",
            ) if sealed else final_drain.next_actions
        ),
    )
