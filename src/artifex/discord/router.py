from __future__ import annotations

from typing import Protocol

from artifex.config.models import DiscordConfig
from artifex.discord.models import (
    CommandName,
    CommandRequest,
    CommandResponse,
    OperatorContext,
)


class RemoteOperations(Protocol):
    def execute(self, request: CommandRequest) -> CommandResponse: ...


class AuthorizationPolicy:
    def __init__(self, config: DiscordConfig) -> None:
        self._users = frozenset(config.allowed_user_ids)
        self._roles = frozenset(config.allowed_role_ids)

    def authorized(self, context: OperatorContext) -> bool:
        if context.user_id in self._users:
            return True
        return bool(self._roles.intersection(context.role_ids))


class DiscordCommandRouter:
    def __init__(
        self,
        authorization: AuthorizationPolicy,
        operations: RemoteOperations,
    ) -> None:
        self._authorization = authorization
        self._operations = operations

    def route(
        self,
        context: OperatorContext,
        name: str,
        args: tuple[str, ...] = (),
    ) -> CommandResponse:
        if not self._authorization.authorized(context):
            return CommandResponse(
                ok=False,
                message="Not authorized to control Artifex.",
                ephemeral=True,
            )

        try:
            command = CommandName(name.casefold())
        except ValueError:
            return CommandResponse(
                ok=False,
                message=f"Unknown Artifex command: {name}",
            )

        try:
            return self._operations.execute(CommandRequest(name=command, args=args))
        except (KeyError, ValueError, RuntimeError) as exc:
            return CommandResponse(ok=False, message=str(exc))
