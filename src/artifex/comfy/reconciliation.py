from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from artifex.comfy import ComfyUIClient
from artifex.comfy.workflow_audit import WorkflowAudit, audit_workflows
from artifex.config.models import ArtifexSettings

QueueState = Literal["idle", "busy", "unknown"]
ReconcileState = Literal[
    "ready", "busy", "pending_runtime_refresh", "unverifiable_queue", "unreachable",
]


class RendererReconcileSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checked_at: datetime
    queue_state: QueueState
    queued: int | None = None
    running: int | None = None
    workflow_ready: bool
    missing_node_types: tuple[str, ...] = ()
    missing_model_choices: tuple[str, ...] = ()
    unverifiable_model_choices: tuple[str, ...] = ()
    error: str | None = None


class RendererReconcileReport(BaseModel):
    """An observed runtime state, never proof of a completed production render."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ReconcileState
    ready: bool
    samples: int = Field(ge=1)
    last: RendererReconcileSample
    next_actions: tuple[str, ...]
    restart_performed: bool = False
    production_qualified: bool = False


def _queue_counts(body: object) -> tuple[QueueState, int | None, int | None]:
    """An absent/ambiguous ComfyUI /queue field must never count as idle."""
    if not isinstance(body, dict):
        return "unknown", None, None
    running = body.get("queue_running")
    pending = body.get("queue_pending")
    if not isinstance(running, list) or not isinstance(pending, list):
        return "unknown", None, None
    return (
        "idle" if not running and not pending else "busy",
        len(pending),
        len(running),
    )


def _sample(
    audit: WorkflowAudit | None,
    queue: object,
    error: str | None,
) -> RendererReconcileSample:
    state, pending, running = _queue_counts(queue)
    return RendererReconcileSample(
        checked_at=datetime.now(UTC),
        queue_state=state,
        queued=pending,
        running=running,
        workflow_ready=audit.ready if audit is not None else False,
        missing_node_types=tuple(sorted({
            kind for entry in audit.entries for kind in entry.missing_node_types
        })) if audit is not None else (),
        missing_model_choices=tuple(sorted({
            f"{asset.label}:{asset.requested}"
            for entry in audit.entries for asset in entry.missing_assets
        })) if audit is not None else (),
        unverifiable_model_choices=tuple(sorted({
            f"{asset.label}:{asset.requested}"
            for entry in audit.entries for asset in entry.unverifiable_assets
        })) if audit is not None else (),
        error=error,
    )


def _decision(
    sample: RendererReconcileSample,
    *,
    require_idle: bool,
) -> tuple[ReconcileState, tuple[str, ...]]:
    if sample.error is not None:
        return "unreachable", (
            "check_local_comfyui_service_and_configured_url",
            "repeat_read_only_runtime_reconciliation",
        )
    if sample.queue_state == "unknown":
        return "unverifiable_queue", (
            "inspect_comfyui_queue_api_schema_or_permissions",
            "do_not_restart_or_assume_idle",
        )
    if sample.queue_state == "busy" and require_idle:
        return "busy", (
            "wait_for_comfyui_running_and_pending_jobs_to_finish",
            "do_not_restart_or_interrupt_gpu_jobs",
        )
    if sample.workflow_ready:
        return "ready", ("run_two_pc_deployment_verify",)
    actions: list[str] = []
    if sample.missing_node_types:
        actions.append("review_missing_custom_nodes_before_any_restart")
    if sample.missing_model_choices:
        actions.append("inspect_approved_model_sources_and_existing_verified_files")
    if sample.unverifiable_model_choices:
        actions.append("inspect_unknown_comfyui_loader_choices")
    actions.append("refresh_or_restart_owned_comfyui_only_after_queue_is_idle")
    actions.append("repeat_read_only_runtime_reconciliation")
    return "pending_runtime_refresh", tuple(actions)


async def reconcile_renderer(
    settings: ArtifexSettings,
    *,
    wait_seconds: float = 0,
    poll_seconds: float = 5,
    require_idle: bool = True,
    client: ComfyUIClient | None = None,
    audit_fn: Callable[..., Awaitable[WorkflowAudit]] = audit_workflows,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> RendererReconcileReport:
    """Wait for real ComfyUI to refresh and report only independently observed facts.

    It NEVER submits/interrupts prompts, restarts a process or updates a config.
    ComfyUI queues are not a distributed lock; even an observed empty queue
    cannot authorize a destructive restart from this command.
    """
    if not 0 <= wait_seconds <= 600:
        raise ValueError("wait_seconds must be between 0 and 600")
    if not 0 < poll_seconds <= 60:
        raise ValueError("poll_seconds must be in (0, 60]")
    primary = settings.render_nodes.primary_node()
    if (
        primary is not None
        and primary[1].base_url.rstrip("/") != settings.comfyui.base_url.rstrip("/")
    ):
        raise ValueError("reconciliation needs one renderer-local ComfyUI endpoint")
    own = client is None
    comfy = client or ComfyUIClient(settings.comfyui)
    start = asyncio.get_running_loop().time()
    count = 0
    try:
        while True:
            count += 1
            queue: object = None
            audit: WorkflowAudit | None = None
            errors: list[str] = []
            try:
                queue = await comfy.queue_snapshot()
            except (httpx.HTTPError, OSError, TypeError, ValueError, RuntimeError) as exc:
                errors.append(f"queue: {type(exc).__name__}: {exc}")
            try:
                audit = await audit_fn(settings, client=comfy)
            except (httpx.HTTPError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                errors.append(f"workflow: {type(exc).__name__}: {exc}")
            sample = _sample(audit, queue, "; ".join(errors) if errors else None)
            status, actions = _decision(sample, require_idle=require_idle)
            if status == "ready" or wait_seconds == 0:
                break
            left = wait_seconds - (asyncio.get_running_loop().time() - start)
            if left <= 0:
                break
            await sleep_fn(min(poll_seconds, left))
        return RendererReconcileReport(
            status=status, ready=status == "ready",
            samples=count, last=sample, next_actions=actions,
        )
    finally:
        if own:
            await comfy.aclose()
