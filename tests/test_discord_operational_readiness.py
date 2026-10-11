from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from artifex.config.models import DiscordConfig
from artifex.discord.bot import (
    ArtifexDiscordClient,
    ReviewView,
    assess_notification_channel_access,
)
from artifex.discord.models import Notification, NotificationKind, OperatorContext
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



def _public_thread(
    *,
    archived: bool = False,
    locked: bool = False,
    kind: discord.ChannelType = discord.ChannelType.public_thread,
    view: bool = True,
    history: bool = True,
    send: bool = True,
) -> MagicMock:
    member = object()
    thread = MagicMock(spec=discord.Thread)
    thread.type = kind
    thread.archived = archived
    thread.locked = locked
    thread.guild = SimpleNamespace(me=member, id=77)
    thread.permissions_for.return_value = SimpleNamespace(
        view_channel=view,
        send_messages_in_threads=send,
        read_message_history=history,
    )
    return thread


@pytest.mark.parametrize(
    ("thread", "expected"),
    [
        (_public_thread(archived=True), "archived"),
        (_public_thread(locked=True), "locked"),
        (_public_thread(kind=discord.ChannelType.private_thread), "public"),
        (_public_thread(send=False), "Send Messages in Threads"),
    ],
)
def test_runtime_never_treats_unusable_thread_as_send_ready(
    thread: MagicMock, expected: str,
) -> None:
    okay, reason = assess_notification_channel_access(thread)
    assert okay is False
    assert expected in reason


def test_active_unlocked_public_thread_is_send_ready() -> None:
    thread = _public_thread()
    assert assess_notification_channel_access(
        thread, require_history=True,
    )[0] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thread",
    [_public_thread(archived=True), _public_thread(locked=True)],
)
async def test_runtime_does_not_send_or_unarchive_blocked_threads(
    thread: MagicMock,
) -> None:
    config = DiscordConfig(
        enabled=True, guild_id=77, channel_id=88, allowed_user_ids=(100,),
    )
    router = DiscordCommandRouter(AuthorizationPolicy(config), FakeOperations())
    bot = ArtifexDiscordClient(config, router, MagicMock())
    bot.get_channel = MagicMock(return_value=thread)
    assert await bot.notify(Notification(
        kind=NotificationKind.COMPLETION,
        title="Artifex Pack completed",
        body="No automatic thread mutation",
    )) is False
    thread.send.assert_not_called()
    thread.edit.assert_not_called()


@pytest.mark.asyncio
async def test_archived_public_thread_can_still_reconcile_by_read_only_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = _public_thread(archived=True, send=False)
    assert assess_notification_channel_access(thread)[0] is False
    assert assess_notification_channel_access(
        thread, require_send=False, require_history=True,
    )[0] is True
    config = DiscordConfig(
        enabled=True, guild_id=77, channel_id=88, allowed_user_ids=(100,),
    )
    router = DiscordCommandRouter(AuthorizationPolicy(config), FakeOperations())
    bot = ArtifexDiscordClient(config, router, MagicMock())
    bot.get_channel = MagicMock(return_value=thread)
    bot._connection.user = SimpleNamespace(id=777)
    read_history = AsyncMock(return_value="654321")
    monkeypatch.setattr(
        "artifex.discord.bot.find_completion_in_history", read_history,
    )
    assert await bot.find_completion_message("pack-1") == "654321"
    read_history.assert_awaited_once_with(
        thread, pack_id="pack-1", bot_user_id=777,
    )
    thread.send.assert_not_called()
