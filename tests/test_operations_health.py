from __future__ import annotations

from pathlib import Path

import pytest

from artifex.config.models import ArtifexSettings, DiscordConfig, StorageConfig
from artifex.db import Database
from artifex.discord.models import Notification, NotificationKind
from artifex.domain import AgentState
from artifex.operations import ComponentHealth, ComponentState, HealthChecker, HealthReport
from artifex.operations.supervisor import HealthSupervisor
from artifex.runtime import RuntimeStore
from artifex.telemetry import TelemetryRepository


class FakeComfy:
    async def health(self):
        from artifex.comfy import ComfyHealth

        return ComfyHealth(available=True, version="test", devices=("gpu",))

    async def aclose(self) -> None:
        return None


class CaptureNotifications:
    def __init__(self) -> None:
        self.items: list[Notification] = []

    async def notify(self, notification: Notification) -> bool:
        self.items.append(notification)
        return True


class SequencedChecker:
    def __init__(self, reports: list[HealthReport]) -> None:
        self.reports = reports
        self.index = 0

    async def check_all(self, *, include_worker: bool = True) -> HealthReport:
        del include_worker
        report = self.reports[min(self.index, len(self.reports) - 1)]
        self.index += 1
        return report


def _component(name: str, state: ComponentState, *, blocking: bool = True):
    return ComponentHealth(
        name=name,
        state=state,
        detail=f"{name}:{state.value}",
        blocking=blocking,
    )


@pytest.mark.asyncio
async def test_health_checker_reports_db_storage_and_configured_backends(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'health.sqlite3').as_posix()}")
    database.migrate()
    settings = ArtifexSettings(
        storage=StorageConfig(
            database_url=database.database_url,
            packs_dir=tmp_path / "packs",
            minimum_free_gib=0,
        )
    )
    telemetry = TelemetryRepository(database)

    async def healthy_llm():
        return _component("llm", ComponentState.HEALTHY)

    async def healthy_evaluator():
        return _component("evaluator", ComponentState.HEALTHY)

    checker = HealthChecker(
        settings,
        database,
        telemetry,
        comfy=FakeComfy(),  # type: ignore[arg-type]
        llm_probe=healthy_llm,
        evaluator_probe=healthy_evaluator,
    )
    report = await checker.check_all(include_worker=False)

    assert report.ready is True
    assert report.require("database").state is ComponentState.HEALTHY
    assert report.require("storage").state is ComponentState.HEALTHY
    assert report.require("discord").state is ComponentState.DISABLED
    database.dispose()


@pytest.mark.asyncio
async def test_watchdog_degrades_then_blocks_and_recovers_backend(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'watchdog.sqlite3').as_posix()}")
    database.migrate()
    runtime = RuntimeStore(database)
    runtime.set_agent_state(AgentState.STARTING, expected=AgentState.STOPPED)
    runtime.set_agent_state(AgentState.RUNNING, expected=AgentState.STARTING)
    telemetry = TelemetryRepository(database)

    bad = HealthReport(
        components=(
            _component("llm", ComponentState.UNHEALTHY),
            _component("comfyui", ComponentState.HEALTHY),
            _component("evaluator", ComponentState.HEALTHY),
            _component("database", ComponentState.HEALTHY),
            _component("storage", ComponentState.HEALTHY),
        )
    )
    good = HealthReport(
        components=tuple(
            _component(name, ComponentState.HEALTHY)
            for name in ("llm", "comfyui", "evaluator", "database", "storage")
        )
    )
    checker = SequencedChecker([bad, bad, good])
    settings = ArtifexSettings()
    settings.operations.backend_failure_threshold = 2
    settings.operations.health_interval_seconds = 0.001
    supervisor = HealthSupervisor(
        checker,  # type: ignore[arg-type]
        runtime,
        telemetry,
        settings.operations,
    )

    await supervisor.maintain(force=True)
    assert runtime.get_agent_state() is AgentState.DEGRADED

    await supervisor.maintain(force=True)
    assert runtime.get_agent_state() is AgentState.BLOCKED

    await supervisor.maintain(force=True)
    assert runtime.get_agent_state() is AgentState.RUNNING
    assert runtime.get_agent_reason() == "health:recovered"
    database.dispose()


@pytest.mark.asyncio
async def test_watchdog_never_overrides_operator_block(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'operator.sqlite3').as_posix()}")
    database.migrate()
    runtime = RuntimeStore(database)
    runtime.set_agent_state(AgentState.STARTING, expected=AgentState.STOPPED)
    runtime.set_agent_state(AgentState.BLOCKED, expected=AgentState.STARTING, reason="operator_block")
    telemetry = TelemetryRepository(database)
    good = HealthReport(
        components=tuple(
            _component(name, ComponentState.HEALTHY)
            for name in ("llm", "comfyui", "evaluator", "database", "storage")
        )
    )
    supervisor = HealthSupervisor(
        SequencedChecker([good]),  # type: ignore[arg-type]
        runtime,
        telemetry,
        ArtifexSettings().operations,
    )

    await supervisor.maintain(force=True)

    assert runtime.get_agent_state() is AgentState.BLOCKED
    assert runtime.get_agent_reason() == "operator_block"
    database.dispose()


@pytest.mark.asyncio
async def test_persistent_backend_failure_blocks_and_surfaces_backend_alert(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'backend-alert.sqlite3').as_posix()}")
    database.migrate()
    runtime = RuntimeStore(database)
    runtime.set_agent_state(AgentState.STARTING, expected=AgentState.STOPPED)
    runtime.set_agent_state(AgentState.RUNNING, expected=AgentState.STARTING)
    telemetry = TelemetryRepository(database)
    notifications = CaptureNotifications()

    bad = HealthReport(
        components=(
            _component("llm", ComponentState.UNHEALTHY),
            _component("comfyui", ComponentState.HEALTHY),
            _component("evaluator", ComponentState.HEALTHY),
            _component("database", ComponentState.HEALTHY),
            _component("storage", ComponentState.HEALTHY),
        )
    )
    settings = ArtifexSettings()
    settings.operations.backend_failure_threshold = 2
    supervisor = HealthSupervisor(
        SequencedChecker([bad, bad]),  # type: ignore[arg-type]
        runtime,
        telemetry,
        settings.operations,
        notifications=notifications,
    )

    await supervisor.maintain(force=True)
    await supervisor.maintain(force=True)

    assert runtime.get_agent_state() is AgentState.BLOCKED
    assert any(
        item.kind is NotificationKind.BACKEND and "llm" in item.title.casefold()
        for item in notifications.items
    )
    database.dispose()


def test_discord_health_reports_missing_channel_permission_as_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'discord-health.sqlite3').as_posix()}")
    database.migrate()
    telemetry = TelemetryRepository(database)
    config = ArtifexSettings(
        discord=DiscordConfig(
            enabled=True,
            allowed_user_ids=(100,),
            guild_id=77,
            channel_id=88,
            token_env="ARTIFEX_TEST_DISCORD_TOKEN",
        )
    )
    monkeypatch.setenv("ARTIFEX_TEST_DISCORD_TOKEN", "test-token")
    checker = HealthChecker(
        config, database, telemetry,
        discord_connected=lambda: True,
        discord_notification_probe=lambda: (
            False, "Discord bot lacks Read Message History permission.",
        ),
    )
    result = checker._check_discord()
    assert result.state is ComponentState.DEGRADED
    assert "Read Message History" in result.detail
    assert result.blocking is False

    good = HealthChecker(
        config, database, telemetry,
        discord_connected=lambda: True,
        discord_notification_probe=lambda: (True, "permissions verified"),
    )
    assert good._check_discord().state is ComponentState.HEALTHY

    disconnected = HealthChecker(
        config, database, telemetry,
        discord_connected=lambda: False,
        discord_notification_probe=lambda: (True, "permissions verified"),
    )
    assert disconnected._check_discord().state is ComponentState.UNHEALTHY
    database.dispose()
