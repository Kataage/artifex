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


def test_notice_dismiss_router_does_not_forward_unauthorized_actor() -> None:
    operations = FakeOperations()
    router = DiscordCommandRouter(
        AuthorizationPolicy(
            DiscordConfig(allowed_user_ids=(100,), allowed_role_ids=(200,))
        ),
        operations,
    )
    args = ("pack-uncertain", "Operator reviewed uncertain message", "DISMISS")
    denied = router.route(
        OperatorContext(user_id=999, role_ids=(998,)),
        "notice-dismiss",
        args,
    )
    assert denied.ok is False
    assert operations.requests == []

    approved = router.route(
        OperatorContext(user_id=100),
        "notice-dismiss",
        args,
    )
    assert approved.ok is True
    assert len(operations.requests) == 1
    assert operations.requests[0].operator_user_id == 100
    assert operations.requests[0].args == args


def test_scope_rejects_authorized_users_outside_configured_guild_channel() -> None:
    operations = FakeOperations()
    router = DiscordCommandRouter(
        AuthorizationPolicy(
            DiscordConfig(
                enabled=True,
                guild_id=801,
                channel_id=901,
                allowed_user_ids=(100,),
                allowed_role_ids=(200,),
            )
        ),
        operations,
    )
    invalid = (
        OperatorContext(user_id=100),
        OperatorContext(user_id=100, guild_id=802, channel_id=901),
        OperatorContext(user_id=100, guild_id=801, channel_id=902),
        OperatorContext(
            user_id=101,
            role_ids=(200,),
            guild_id=802,
            channel_id=901,
        ),
        OperatorContext(
            user_id=101,
            role_ids=(200,),
            guild_id=801,
            channel_id=902,
        ),
    )
    for context in invalid:
        result = router.route(
            context,
            "notice-dismiss",
            ("pack-1", "Authorized user at wrong location", "DISMISS"),
        )
        assert result.ok is False
        assert result.ephemeral is True
    assert operations.requests == []

    valid = router.route(
        OperatorContext(
            user_id=100,
            guild_id=801,
            channel_id=901,
        ),
        "notice-dismiss",
        ("pack-1", "Authorized user at intended channel", "DISMISS"),
    )
    assert valid.ok is True
    assert operations.requests[0].operator_user_id == 100

    via_role = router.route(
        OperatorContext(
            user_id=101,
            role_ids=(200,),
            guild_id=801,
            channel_id=901,
        ),
        "approve",
        ("review-1",),
    )
    assert via_role.ok is True


def test_review_button_like_commands_share_same_scope_restriction() -> None:
    operations = FakeOperations()
    router = DiscordCommandRouter(
        AuthorizationPolicy(
            DiscordConfig(guild_id=801, channel_id=901, allowed_user_ids=(100,))
        ),
        operations,
    )
    for command in ("approve", "reject", "retry", "alternate", "details"):
        rejected = router.route(
            OperatorContext(user_id=100, guild_id=801, channel_id=999),
            command,
            ("review-1",),
        )
        assert rejected.ok is False
    assert operations.requests == []
