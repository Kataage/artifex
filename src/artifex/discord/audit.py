from __future__ import annotations

from artifex.config.models import DiscordConfig
from artifex.discord.models import CommandResponse, OperatorContext
from artifex.telemetry import EventSeverity, TelemetryRepository

# Only real, delivered Discord interactions may satisfy the six control gates.
_REQUIRED = frozenset({"status", "pause", "resume", "approve", "reject", "retry"})


class DiscordQualificationAudit:
    """Persist a successful Discord interaction *after* its response is sent.

    This audit is intentionally not attached to the generic remote-operations
    interface used by CLI commands and tests: CLI invocations are not Discord.
    """

    def __init__(
        self,
        config: DiscordConfig,
        telemetry: TelemetryRepository,
    ) -> None:
        self._config = config
        self._telemetry = telemetry

    def record_completed(
        self,
        *,
        command: str,
        response: CommandResponse,
        operator: OperatorContext,
        interaction_id: int | None,
        guild_id: int | None,
        channel_id: int | None,
    ) -> bool:
        if not self._config.enabled or command not in _REQUIRED or not response.ok:
            return False
        if self._config.guild_id is None or self._config.channel_id is None:
            return False
        if guild_id != self._config.guild_id or channel_id != self._config.channel_id:
            return False
        if not isinstance(interaction_id, int) or interaction_id <= 0:
            return False
        if operator.user_id <= 0 or not (
            operator.user_id in self._config.allowed_user_ids
            or bool(set(operator.role_ids).intersection(self._config.allowed_role_ids))
        ):
            return False
        # A no-op response does not prove that pause/resume controls work.
        if command == "pause" and response.message != "Artifex paused.":
            return False
        if command == "resume" and response.message != "Artifex resumed.":
            return False
        self._telemetry.record(
            "discord.interaction_completed",
            EventSeverity.INFO,
            {
                "command": command,
                "interaction_id": str(interaction_id),
                "guild_id": guild_id,
                "channel_id": channel_id,
                "user_id": operator.user_id,
                "role_ids": list(operator.role_ids),
                "response_ok": True,
            },
        )
        return True
