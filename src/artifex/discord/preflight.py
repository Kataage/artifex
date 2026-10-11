"""Read-only Discord operational probe, with no Gateway login or writes.

Every live request uses GET. Never print or persist the bot credential.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from artifex.config.models import DiscordConfig
from artifex.discord.models import CommandName
from artifex.discord.remediation import discord_repair_actions

API_BASE = "https://discord.com/api/v10"

# Discord permission bitfields, documented by Discord.
_ADMINISTRATOR = 1 << 3
_VIEW_CHANNEL = 1 << 10
_SEND_MESSAGES = 1 << 11
_READ_MESSAGE_HISTORY = 1 << 16
_SEND_IN_THREADS = 1 << 38


def _snowflake(value: object) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError("Discord returned an invalid snowflake identifier")
    if not 0 < int(value) < 2**64:
        raise ValueError("Discord returned an out-of-range snowflake identifier")
    return value


def _bitfield(value: object) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError("Discord returned invalid permissions")
    return int(value)


def effective_channel_permissions(
    *,
    guild_id: str,
    user_id: str,
    role_ids: list[str],
    roles: list[dict[str, Any]],
    channel: dict[str, Any],
) -> int:
    """Compute Discord guild/channel overwrites in documented precedence."""
    everyone = next(
        (role for role in roles if role.get("id") == guild_id),
        None,
    )
    if everyone is None:
        raise ValueError("Discord @everyone guild role missing")
    permissions = _bitfield(everyone.get("permissions"))
    for role in roles:
        if role.get("id") in role_ids:
            permissions |= _bitfield(role.get("permissions"))
    if permissions & _ADMINISTRATOR:
        return (1 << 64) - 1

    overwrites = channel.get("permission_overwrites")
    if not isinstance(overwrites, list):
        raise TypeError("Discord channel permission overwrites unavailable")

    def _combine(items: list[dict[str, Any]]) -> tuple[int, int]:
        deny = 0
        allow = 0
        for item in items:
            deny |= _bitfield(item.get("deny"))
            allow |= _bitfield(item.get("allow"))
        return deny, allow

    def _apply(items: list[dict[str, Any]]) -> None:
        nonlocal permissions
        deny, allow = _combine(items)
        permissions = (permissions & ~deny) | allow

    _apply([
        item for item in overwrites
        if isinstance(item, dict) and item.get("type") == 0
        and item.get("id") == guild_id
    ])
    _apply([
        item for item in overwrites
        if isinstance(item, dict) and item.get("type") == 0
        and item.get("id") in role_ids
    ])
    _apply([
        item for item in overwrites
        if isinstance(item, dict) and item.get("type") == 1
        and item.get("id") == user_id
    ])
    return permissions


def _command_names(commands: object) -> set[str]:
    if not isinstance(commands, list):
        raise TypeError("Discord application commands response is invalid")
    groups = [
        c for c in commands
        if isinstance(c, dict) and c.get("name") == "artifex"
        and c.get("type", 1) == 1
    ]
    if len(groups) != 1:
        return set()
    options = groups[0].get("options", [])
    if not isinstance(options, list):
        raise TypeError("Discord command options are invalid")
    return {
        item["name"] for item in options
        if isinstance(item, dict) and item.get("type") == 1
        and isinstance(item.get("name"), str)
    }


async def probe_discord(
    config: DiscordConfig,
    *,
    live: bool = False,
    client: httpx.AsyncClient | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Check configured identity/registration/permissions, never post messages.

    A successful empty history read is not proof of delivery or historical
    access, so permission bits and one bounded GET are both required.
    """
    credential = token if token is not None else os.environ.get(config.token_env)
    checks: dict[str, bool] = {
        "enabled": config.enabled,
        "operator_allowlist": bool(config.allowed_user_ids or config.allowed_role_ids),
        "guild_configured": config.guild_id is not None,
        "channel_configured": config.channel_id is not None,
        "credential_present": bool(credential),
    }
    messages: list[str] = []
    local_ready = all(checks.values())
    report: dict[str, Any] = {
        "ready": local_ready,
        "live_verified": False,
        "requests_sent": 0,
        "checks": checks,
        "problems": messages,
        "writes_performed": 0,
        "remediation": [],
    }
    if not local_ready:
        messages.append(
            "Enable Discord, configure guild/channel and authorized users, "
            "and set the bot token environment variable."
        )
        report["remediation"] = discord_repair_actions(
            config, checks, live_requested=live, requests_sent=0,
        )
        return report
    if not live:
        messages.append("Local configuration only; no Discord API requests were made.")
        report["remediation"] = discord_repair_actions(
            config, checks, live_requested=False, requests_sent=0,
        )
        return report

    assert config.guild_id is not None
    assert config.channel_id is not None
    assert credential is not None
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"Authorization": f"Bot {credential}"},
            timeout=8.0,
            follow_redirects=False,
        )
    assert client is not None

    async def get(path: str) -> Any:
        report["requests_sent"] += 1
        response = await client.get(
            path,
            headers={"Authorization": f"Bot {credential}"},
        )
        if response.status_code != 200:
            # Never print response bodies or HTTP headers (might contain secrets).
            raise ValueError(
                f"Discord read-only GET failed ({response.status_code})"
            )
        return response.json()

    try:
        bot = await get("/users/@me")
        application = await get("/applications/@me")
        if not isinstance(bot, dict) or bot.get("bot") is not True:
            raise ValueError("Discord token is not a bot credential")
        if not isinstance(application, dict):
            raise TypeError("Discord application information is malformed")
        user_id = _snowflake(bot.get("id"))
        application_id = _snowflake(application.get("id"))
        checks["bot_identity_verified"] = True

        member = await get(f"/guilds/{config.guild_id}/members/{user_id}")
        roles = await get(f"/guilds/{config.guild_id}/roles")
        channel = await get(f"/channels/{config.channel_id}")
        if not isinstance(member, dict) or not isinstance(roles, list):
            raise TypeError("Discord guild membership/roles unavailable")
        if not isinstance(channel, dict):
            raise TypeError("Discord channel response malformed")
        if channel.get("guild_id") != str(config.guild_id):
            raise ValueError("Configured channel belongs to another guild")
        checks["guild_channel_match"] = True

        role_ids_raw = member.get("roles")
        if not isinstance(role_ids_raw, list):
            raise TypeError("Discord bot membership roles unavailable")
        role_ids = [_snowflake(value) for value in role_ids_raw]
        if not all(isinstance(item, dict) for item in roles):
            raise ValueError("Discord guild role details malformed")

        channel_type = channel.get("type")
        if channel_type in (11, 12):  # public/private threads
            # Thread permissions inherit from the parent channel; private
            # thread membership must also be proven and is not probed here.
            if channel_type == 12:
                raise ValueError(
                    "Private thread membership cannot be proven read-only; "
                    "configure a regular text channel."
                )
            parent_id = _snowflake(channel.get("parent_id"))
            parent = await get(f"/channels/{parent_id}")
            if not isinstance(parent, dict) or parent.get("guild_id") != str(config.guild_id):
                raise ValueError("Discord thread parent channel unverifiable")
            permission_channel = parent
        elif channel_type in (0, 5):
            permission_channel = channel
        else:
            raise ValueError("Configured Discord channel is not a supported text channel")

        bits = effective_channel_permissions(
            guild_id=str(config.guild_id),
            user_id=user_id,
            role_ids=role_ids,
            roles=roles,
            channel=permission_channel,
        )
        checks["view_channel"] = bool(bits & _VIEW_CHANNEL)
        checks["send_messages"] = bool(
            bits & (_SEND_IN_THREADS if channel_type in (11, 12) else _SEND_MESSAGES)
        )
        checks["read_message_history"] = bool(bits & _READ_MESSAGE_HISTORY)
        if not all((
            checks["view_channel"], checks["send_messages"],
            checks["read_message_history"] or not config.notify_completion,
        )):
            messages.append("Discord channel is missing view, send or history permission.")

        commands = await get(
            f"/applications/{application_id}/guilds/{config.guild_id}/commands"
        )
        expected = {command.value for command in CommandName}
        missing = sorted(expected - _command_names(commands))
        checks["guild_slash_commands_synced"] = not missing
        if missing:
            messages.append("Missing registered /artifex commands: " + ", ".join(missing))

        # This verifies the API path, not proof of prior message delivery.
        # Avoid treating a zero-message 200 response as evidence of access.
        if checks["view_channel"] and checks["read_message_history"]:
            history = await get(f"/channels/{config.channel_id}/messages?limit=1")
            checks["history_api_accessible"] = isinstance(history, list)
            if not checks["history_api_accessible"]:
                messages.append("Discord returned malformed message-history data.")
        else:
            checks["history_api_accessible"] = False

        live_keys = [
            "bot_identity_verified", "guild_channel_match",
            "view_channel", "send_messages", "guild_slash_commands_synced",
        ]
        if config.notify_completion:
            live_keys.extend(("read_message_history", "history_api_accessible"))
        report["ready"] = all(checks[key] for key in live_keys)
        report["live_verified"] = report["ready"]
    except (ValueError, TypeError, KeyError, httpx.HTTPError) as exc:
        report["ready"] = False
        messages.append(
            str(exc) if isinstance(exc, ValueError)
            else f"Discord GET probe failed: {type(exc).__name__}"
        )
    finally:
        if own_client:
            await client.aclose()
    report["remediation"] = discord_repair_actions(
        config, checks, live_requested=live,
        requests_sent=report["requests_sent"],
    )
    return report
