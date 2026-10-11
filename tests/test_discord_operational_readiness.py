from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import discord

from artifex.config.models import DiscordConfig
from artifex.discord.bot import (
    ArtifexDiscordClient,
    ReviewView,
    assess_notification_channel_access,
)
from artifex.discord.models import OperatorContext
from artifex.discord.router import AuthorizationPolicy, DiscordCommandRouter


class FakeOperations:
    def execute(self, request):
        del request
        raise AssertionError("No operations should run during setup")


def _channel(*, view=True, send=True, history=True):
    member = object()
    channel = SimpleNamespace(
        guild=SimpleNamespace(me=member, id=77),
        permissions_for=lambda target: SimpleNamespace(
            view_channel=view,
            send_messages=send,
            send_messages_in_threads=send,
            read_message_history=history,
        ) if target is member else None,
    )
    return channel


def test_discord_notification_access_requires_member_and_send_permissions() -> None:
    ok, detail = assess_notification_channel_access(_channel())
    assert ok and "verified" in detail

    for channel in (
        None,
        SimpleNamespace(guild=SimpleNamespace(me=None)),
        _channel(view=False),
        _channel(send=False),
    ):
        ok, _ = assess_notification_channel_access(channel)
        assert ok is False


def test_completion_history_access_is_stricter_than_sending() -> None:
    channel = _channel(history=False)
    assert assess_notification_channel_access(channel)[0] is True
    ok, message = assess_notification_channel_access(
        channel, require_history=True,
    )
    assert ok is False
    assert "Read Message History" in message


def test_review_buttons_and_slash_commands_propagate_context() -> None:
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=123),
        guild_id=77,
        channel_id=88,
    )
    button_context = ReviewView._context(interaction)
    slash_context = ArtifexDiscordClient._context(None, interaction)
    expected = OperatorContext(user_id=123, guild_id=77, channel_id=88)
    assert button_context == expected
    assert slash_context == expected


def test_discord_group_registers_all_operator_commands() -> None:
    config = DiscordConfig(allowed_user_ids=(100,), guild_id=77, channel_id=88)
    router = DiscordCommandRouter(AuthorizationPolicy(config), FakeOperations())
    bot = ArtifexDiscordClient(config, router, MagicMock())
    names = {command.name for command in bot._group.commands}
    assert {
        "status", "pause", "resume", "approve", "reject", "retry",
        "notices", "notice", "notice-dismiss",
    } <= names


def test_foreign_guild_notification_target_is_ignored() -> None:
    config = DiscordConfig(allowed_user_ids=(100,), guild_id=77, channel_id=88)
    router = DiscordCommandRouter(AuthorizationPolicy(config), FakeOperations())
    bot = ArtifexDiscordClient(config, router, MagicMock())
    foreign = MagicMock(spec=discord.TextChannel)
    foreign.guild.id = 999
    bot.get_channel = MagicMock(return_value=foreign)
    assert bot._notification_channel() is None
