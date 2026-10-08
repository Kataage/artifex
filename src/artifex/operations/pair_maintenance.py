from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from artifex.comfy import ComfyUIClient
from artifex.comfy.admission import ComfySubmissionFence
from artifex.comfy.reconciliation import RendererReconcileReport, reconcile_renderer
from artifex.config.models import ArtifexSettings
from artifex.db import Database
from artifex.domain import AgentState
from artifex.operations.fence import manage_submission_fence
from artifex.runtime import RuntimeStore


class RendererGatewayStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admission_sealed: StrictBool
    upstream_loopback_configured: StrictBool
    external_loopback_clients_fenced: StrictBool
    restart_authorized: StrictBool


class RendererGatewaySeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admission_sealed: StrictBool
    reason: str
    restart_authorized: StrictBool


class RendererGatewayRelease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admission_sealed: StrictBool
    restart_authorized: StrictBool


class RendererGatewayAPI:
    """Authenticated admin API of exactly the configured primary PC-B gateway.

    No redirect following, environment proxy or accidental credential reuse
    across hosts. The endpoint is from the operator's explicit controller
    config, never an HTTP-provided redirect or a discovered URL.
    """

    def __init__(
        self,
        settings: ArtifexSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        primary = settings.render_nodes.primary_node()
        if primary is None:
            raise ValueError("two-PC maintenance requires a configured primary render node")
        endpoint = primary[1].base_url.rstrip("/")
        if endpoint != settings.comfyui.base_url.rstrip("/"):
            raise ValueError(
                "primary render node and comfyui.base_url must be the same authenticated gateway"
            )
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.port is None
            or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment
            or parsed.username is not None or parsed.password is not None
        ):
            raise ValueError("primary gateway must be a bare http(s) hostname and port")
        token_env = settings.comfyui.gateway_token_env
        if token_env is None:
            raise ValueError("comfyui.gateway_token_env is required for pair maintenance")
        token = os.environ.get(token_env)
        if not token or len(token) < 24:
            raise ValueError("configured gateway token is missing or too short")
        self._owns = client is None
        self._client = client or httpx.AsyncClient(
            base_url=endpoint,
            timeout=httpx.Timeout(settings.comfyui.timeout_seconds),
            follow_redirects=False,
            trust_env=False,
        )
        self._auth = {"Authorization": f"Bearer {token}"}

    async def _request(self, method: str, path: str) -> tuple[int, dict[str, object]]:
        response = await self._client.request(
            method, path,
            headers=self._auth,
            json={} if method == "POST" else None,
            follow_redirects=False,
        )
        if response.status_code not in ({200, 409} if path.endswith("/seal") else {200}):
            raise ValueError(f"gateway API rejected {path} with HTTP {response.status_code}")
        body = response.json()
        if not isinstance(body, dict):
            raise TypeError("gateway API returned a non-object response")
        return response.status_code, body

    async def status(self) -> RendererGatewayStatus:
        _, body = await self._request("GET", "/v1/gateway/status")
        result = RendererGatewayStatus.model_validate(body)
        if (
            not result.upstream_loopback_configured
            or result.external_loopback_clients_fenced
            or result.restart_authorized
        ):
            raise ValueError("unexpected gateway safety attestation")
        return result

    async def seal(self) -> bool:
        code, body = await self._request("POST", "/v1/gateway/seal")
        result = RendererGatewaySeal.model_validate(body)
        if result.restart_authorized:
            raise ValueError("unexpected gateway restart authorization")
        if code == 409 and result.admission_sealed:
            raise ValueError("conflicting renderer gateway seal response")
        return code == 200 and result.admission_sealed

    async def release(self) -> None:
        _, body = await self._request("POST", "/v1/gateway/release")
        result = RendererGatewayRelease.model_validate(body)
        if result.admission_sealed or result.restart_authorized:
            raise ValueError("unexpected renderer gateway release response")

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()


PairStatus = Literal[
    "preview", "gateway_unreachable", "blocked", "partial",
    "both_sealed", "runtime_unready", "released",
]


class PairMaintenanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["preview", "seal", "release"]
    status: PairStatus
    completed: bool
    controller_paused: bool
    controller_sealed: bool
    renderer_sealed: bool | None
    next_actions: tuple[str, ...]
    # Neither a gateway nor a two-PC snapshot can fence third-party loopback clients.
    external_loopback_clients_fenced: bool = False
    restart_authorized: bool = False
    production_qualified: bool = False


async def coordinate_pair_maintenance(
    settings: ArtifexSettings,
    database: Database,
    runtime: RuntimeStore,
    comfy: ComfyUIClient,
    fence: ComfySubmissionFence,
    *,
    apply: bool = False,
    release: bool = False,
    wait_seconds: float = 90,
    poll_seconds: float = 5,
    gateway: RendererGatewayAPI | None = None,
    reconcile_fn: Callable[..., Awaitable[RendererReconcileReport]] = reconcile_renderer,
) -> PairMaintenanceReport:
    """Coordinate both admission barriers but NEVER restart either service.

    Failure to reach PC-B after sealing PC-A deliberately leaves PC-A sealed
    and paused. Releasing requires fresh successful *runtime* verification
    and never resumes autonomous production.
    """
    if release and not apply:
        raise ValueError("--release requires --apply")
    if not 0 <= wait_seconds <= 600 or not 0 < poll_seconds <= 60:
        raise ValueError("invalid bounded polling options")
    own = gateway is None
    remote = gateway or RendererGatewayAPI(settings)

    async def report(
        mode: Literal["preview", "seal", "release"],
        status: PairStatus,
        *,
        completed: bool = False,
        renderer: bool | None = None,
        actions: tuple[str, ...],
    ) -> PairMaintenanceReport:
        return PairMaintenanceReport(
            mode=mode, status=status, completed=completed,
            controller_paused=runtime.get_agent_state() is AgentState.PAUSED,
            controller_sealed=await asyncio.to_thread(fence.status),
            renderer_sealed=renderer, next_actions=actions,
        )

    mode: Literal["preview", "seal", "release"] = (
        "release" if release else "seal" if apply else "preview"
    )
    try:
        try:
            remote_before = await remote.status()
        except (httpx.HTTPError, ValidationError, ValueError, TypeError, OSError):
            return await report(
                mode, "gateway_unreachable",
                actions=("verify_pc_b_gateway_url_token_and_service_before_mutation",),
            )
        local_before = await asyncio.to_thread(fence.status)
        if not apply:
            return await report(
                "preview", "preview", renderer=remote_before.admission_sealed,
                actions=(
                    "both_fences_already_sealed; no_restart_permission"
                    if local_before and remote_before.admission_sealed
                    else "apply_coordinated_seal_when_ready",
                ),
            )

        if release:
            if (
                runtime.get_agent_state() is not AgentState.PAUSED
                or not local_before
                or not remote_before.admission_sealed
            ):
                return await report(
                    "release", "blocked", renderer=remote_before.admission_sealed,
                    actions=("both_fences_must_be_sealed_and_controller_paused_before_release",),
                )
            try:
                checked = await reconcile_fn(
                    settings, client=comfy, require_idle=True, wait_seconds=0,
                )
            except (httpx.HTTPError, ValueError, RuntimeError, OSError, TypeError):
                return await report(
                    "release", "runtime_unready", renderer=True,
                    actions=("repair_renderer_and_repeat_live_workflow_audit",),
                )
            if not checked.ready:
                return await report(
                    "release", "runtime_unready", renderer=True,
                    actions=("repair_renderer_and_repeat_live_workflow_audit",),
                )
            try:
                await remote.release()
                remote_after = await remote.status()
            except (httpx.HTTPError, ValidationError, ValueError, TypeError, OSError):
                return await report(
                    "release", "partial", renderer=None,
                    actions=("inspect_pc_b_gateway_state; keep_controller_fenced",),
                )
            if remote_after.admission_sealed:
                return await report(
                    "release", "partial", renderer=True,
                    actions=("renderer_refused_release; keep_controller_fenced",),
                )
            # If local release fails, do not misreport an atomic two-PC release.
            try:
                await fence.release()
            except (OSError, ValueError):
                return await report(
                    "release", "partial", renderer=False,
                    actions=("retry_manual_controller_fence_release_without_resuming",),
                )
            return await report(
                "release", "released", completed=True, renderer=False,
                actions=("verify_deployment_before_explicit_controller_resume",),
            )

        # PC-B status was authenticated and checked before modifying PC-A.
        # Keep existing seal durable across retries, but demand PAUSED state.
        if local_before and runtime.get_agent_state() is not AgentState.PAUSED:
            return await report(
                "seal", "blocked", renderer=remote_before.admission_sealed,
                actions=("pause_running_controller_before_coordinating_sealed_gateway",),
            )
        try:
            local_result = await manage_submission_fence(
                database, runtime, comfy, fence,
                apply=True, wait_seconds=wait_seconds, poll_seconds=poll_seconds,
            )
        except (OSError, ValueError, RuntimeError):
            return await report(
                "seal", "blocked", renderer=remote_before.admission_sealed,
                actions=("inspect_controller_drain_error; never_restart_renderer",),
            )
        if not local_result.submission_blocked:
            return await report(
                "seal", "blocked", renderer=remote_before.admission_sealed,
                actions=local_result.next_actions,
            )
        try:
            if not remote_before.admission_sealed and not await remote.seal():
                return await report(
                    "seal", "partial", renderer=False,
                    actions=("pc_a_is_sealed; wait_for_pc_b_queue_then_retry_seal",),
                )
            remote_after = await remote.status()
        except (httpx.HTTPError, ValidationError, ValueError, TypeError, OSError):
            return await report(
                "seal", "partial", renderer=None,
                actions=("pc_a_is_sealed; inspect_pc_b_state_then_retry_seal",),
            )
        if not remote_after.admission_sealed:
            return await report(
                "seal", "partial", renderer=False,
                actions=("pc_a_is_sealed; pc_b_seal_unconfirmed",),
            )
        return await report(
            "seal", "both_sealed", completed=True, renderer=True,
            actions=(
                "both_artifex_admission_paths_sealed",
                "pc_b_loopback_clients_not_fenced; restart_not_authorized",
            ),
        )
    finally:
        if own:
            await remote.aclose()
