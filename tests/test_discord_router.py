from __future__ import annotations

from artifex.config.models import DiscordConfig
from artifex.discord.models import CommandRequest, CommandResponse, OperatorContext
from artifex.discord.router import AuthorizationPolicy, DiscordCommandRouter


class FakeOperations:
    def __init__(self) -> None:
        self.requests: list[CommandRequest] = []

    def execute(self, request: CommandRequest) -> CommandResponse:
        self.requests.append(request)
        return CommandResponse(ok=True, message=request.name.value)


def test_router_denies_unauthorized_users_without_executing() -> None:
    operations = FakeOperations()
    router = DiscordCommandRouter(
        AuthorizationPolicy(
            DiscordConfig(
                allowed_user_ids=(100,),
                allowed_role_ids=(200,),
            )
        ),
        operations,
    )

    response = router.route(OperatorContext(user_id=999, role_ids=(998,)), "pause")

    assert response.ok is False
    assert operations.requests == []


def test_router_allows_user_or_role_and_validates_command() -> None:
    operations = FakeOperations()
    router = DiscordCommandRouter(
        AuthorizationPolicy(
            DiscordConfig(
                allowed_user_ids=(100,),
                allowed_role_ids=(200,),
            )
        ),
        operations,
    )

    by_user = router.route(OperatorContext(user_id=100), "status")
    by_role = router.route(
        OperatorContext(user_id=999, role_ids=(200,)),
        "character",
        ("char-a",),
    )
    unknown = router.route(OperatorContext(user_id=100), "destroy-everything")

    assert by_user.ok is True
    assert by_role.ok is True
    assert unknown.ok is False
    assert [request.name.value for request in operations.requests] == [
        "status",
        "character",
    ]
