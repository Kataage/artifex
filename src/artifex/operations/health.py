from __future__ import annotations

import os
import shutil
from collections.abc import Awaitable, Callable
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from artifex.comfy import ComfyUIClient
from artifex.config.models import ArtifexSettings
from artifex.db import Database
from artifex.telemetry import TelemetryRepository


class ComponentState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"


class HealthModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentHealth(HealthModel):
    name: str
    state: ComponentState
    detail: str
    blocking: bool = False
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)


class HealthReport(HealthModel):
    components: tuple[ComponentHealth, ...]

    @property
    def ready(self) -> bool:
        return not any(
            item.blocking and item.state is ComponentState.UNHEALTHY
            for item in self.components
        )

    def require(self, name: str) -> ComponentHealth:
        for item in self.components:
            if item.name == name:
                return item
        raise KeyError(f"health component not found: {name}")


class DiscordConnectionProbe(Protocol):
    def __call__(self) -> bool: ...


class HealthChecker:
    def __init__(
        self,
        settings: ArtifexSettings,
        database: Database,
        telemetry: TelemetryRepository,
        *,
        comfy: ComfyUIClient | None = None,
        discord_connected: DiscordConnectionProbe | None = None,
        llm_probe: Callable[[], Awaitable[ComponentHealth]] | None = None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._telemetry = telemetry
        self._comfy = comfy
        self._discord_connected = discord_connected
        self._llm_probe = llm_probe

    async def check_all(self, *, include_worker: bool = True) -> HealthReport:
        components = [
            await self._check_llm(),
            await self._check_comfy(),
            await self._check_evaluator(),
            self._check_database(),
            self._check_storage(),
            self._check_discord(),
        ]
        if include_worker:
            components.append(self._check_worker())
        return HealthReport(components=tuple(components))

    async def _check_llm(self) -> ComponentHealth:
        if self._llm_probe is not None:
            return await self._llm_probe()

        timeout = min(10.0, self._settings.llm.timeout_seconds)
        try:
            async with httpx.AsyncClient(
                base_url=self._settings.llm.base_url.rstrip("/"),
                timeout=httpx.Timeout(timeout),
            ) as client:
                response = await client.get("/health")
                if response.status_code == 404:
                    response = await client.get("/v1/models")
                response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            return ComponentHealth(
                name="llm",
                state=ComponentState.UNHEALTHY,
                detail=f"LLM endpoint unavailable: {exc}",
                blocking=True,
            )
        return ComponentHealth(
            name="llm",
            state=ComponentState.HEALTHY,
            detail="LLM endpoint responded successfully.",
            blocking=True,
        )

    async def _check_comfy(self) -> ComponentHealth:
        client = self._comfy
        owns_client = client is None
        if client is None:
            client = ComfyUIClient(self._settings.comfyui)
        try:
            health = await client.health()
        finally:
            if owns_client:
                await client.aclose()

        if not health.available:
            return ComponentHealth(
                name="comfyui",
                state=ComponentState.UNHEALTHY,
                detail=health.detail or "ComfyUI endpoint unavailable.",
                blocking=True,
            )
        return ComponentHealth(
            name="comfyui",
            state=ComponentState.HEALTHY,
            detail="ComfyUI endpoint responded successfully.",
            blocking=True,
            metrics={
                "version": health.version or "unknown",
                "device_count": len(health.devices),
            },
        )

    async def _check_evaluator(self) -> ComponentHealth:
        config = self._settings.evaluation
        if not config.vision_base_url or not config.vision_model:
            return ComponentHealth(
                name="evaluator",
                state=ComponentState.UNHEALTHY,
                detail="Vision evaluator endpoint/model are not configured.",
                blocking=True,
            )
        timeout = min(10.0, config.vision_timeout_seconds)
        headers: dict[str, str] = {}
        if config.vision_api_key_env:
            token = os.environ.get(config.vision_api_key_env)
            if token:
                headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(
                base_url=config.vision_base_url.rstrip("/"),
                timeout=httpx.Timeout(timeout),
                headers=headers,
            ) as client:
                response = await client.get("/health")
                if response.status_code == 404:
                    response = await client.get("/v1/models")
                response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            return ComponentHealth(
                name="evaluator",
                state=ComponentState.UNHEALTHY,
                detail=f"Vision evaluator unavailable: {exc}",
                blocking=True,
            )
        return ComponentHealth(
            name="evaluator",
            state=ComponentState.HEALTHY,
            detail=f"Vision evaluator is ready: {config.vision_model}.",
            blocking=True,
        )

    def _check_database(self) -> ComponentHealth:
        try:
            with self._database.session() as session:
                value = session.execute(text("SELECT 1")).scalar_one()
            if value != 1:
                raise RuntimeError("database SELECT 1 returned unexpected value")
        except Exception as exc:
            return ComponentHealth(
                name="database",
                state=ComponentState.UNHEALTHY,
                detail=f"Database unavailable: {exc}",
                blocking=True,
            )
        return ComponentHealth(
            name="database",
            state=ComponentState.HEALTHY,
            detail="Database read/write session is available.",
            blocking=True,
        )

    def _check_storage(self) -> ComponentHealth:
        path = self._settings.storage.packs_dir.expanduser().resolve()
        probe = path / ".artifex-write-probe"
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            usage = shutil.disk_usage(path)
        except OSError as exc:
            return ComponentHealth(
                name="storage",
                state=ComponentState.UNHEALTHY,
                detail=f"Archive storage is not writable: {exc}",
                blocking=True,
            )

        free_gib = usage.free / (1024**3)
        minimum = self._settings.storage.minimum_free_gib
        if free_gib < minimum:
            return ComponentHealth(
                name="storage",
                state=ComponentState.UNHEALTHY,
                detail=(
                    f"Low disk space: {free_gib:.2f} GiB free; "
                    f"minimum is {minimum:.2f} GiB."
                ),
                blocking=True,
                metrics={"free_gib": free_gib, "minimum_free_gib": minimum},
            )
        return ComponentHealth(
            name="storage",
            state=ComponentState.HEALTHY,
            detail=f"Archive storage writable; {free_gib:.2f} GiB free.",
            blocking=True,
            metrics={"free_gib": free_gib, "minimum_free_gib": minimum},
        )

    def _check_discord(self) -> ComponentHealth:
        config = self._settings.discord
        if not config.enabled:
            return ComponentHealth(
                name="discord",
                state=ComponentState.DISABLED,
                detail="Discord integration is disabled.",
                blocking=False,
            )
        if not os.environ.get(config.token_env):
            return ComponentHealth(
                name="discord",
                state=ComponentState.UNHEALTHY,
                detail=f"Discord token variable is missing: {config.token_env}",
                blocking=False,
            )
        if self._discord_connected is not None and not self._discord_connected():
            return ComponentHealth(
                name="discord",
                state=ComponentState.UNHEALTHY,
                detail="Discord client is configured but not connected.",
                blocking=False,
            )
        return ComponentHealth(
            name="discord",
            state=ComponentState.HEALTHY,
            detail="Discord configuration/connection is ready.",
            blocking=False,
        )

    def _check_worker(self) -> ComponentHealth:
        age = self._telemetry.heartbeat_age_seconds()
        if age is None:
            return ComponentHealth(
                name="worker",
                state=ComponentState.DEGRADED,
                detail="No daemon heartbeat has been recorded yet.",
                blocking=False,
            )
        stale = self._settings.operations.heartbeat_stale_seconds
        if age > stale:
            return ComponentHealth(
                name="worker",
                state=ComponentState.UNHEALTHY,
                detail=f"Daemon heartbeat is stale ({age:.1f}s > {stale:.1f}s).",
                blocking=False,
                metrics={"age_seconds": age},
            )
        return ComponentHealth(
            name="worker",
            state=ComponentState.HEALTHY,
            detail=f"Daemon heartbeat age is {age:.1f}s.",
            blocking=False,
            metrics={"age_seconds": age},
        )
