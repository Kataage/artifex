"""Stable, secret-free repair guidance for the read-only Discord probe.

A hint is not an authorization to change Discord permissions, YAML, the
Task Scheduler, or the running GPU service. No value from an API response is
interpolated into a suggested executable command.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from artifex.config.models import DiscordConfig

Scope = Literal["pc_a", "discord_server"]


class DiscordRepairAction(TypedDict):
    code: str
    scope: Scope
    config_field: str | None
    instruction: str
    safety: Literal["manual_review"]
    automatically_applied: Literal[False]


def discord_repair_actions(
    config: DiscordConfig,
    checks: dict[str, bool],
    *,
    live_requested: bool,
    requests_sent: int,
) -> list[DiscordRepairAction]:
    """Return precise, deterministic steps; never guess missing live facts."""
    actions: list[DiscordRepairAction] = []

    def add(code: str, scope: Scope, field: str | None, instruction: str) -> None:
        actions.append({
            "code": code,
            "scope": scope,
            "config_field": field,
            "instruction": instruction,
            "safety": "manual_review",
            "automatically_applied": False,
        })

    if not config.enabled:
        # An intentionally disabled Discord integration needs no repair.
        return actions

    if not checks.get("operator_allowlist", False):
        add(
            "operator_allowlist", "pc_a", "discord.allowed_user_ids",
            "Identify the actual operator's Discord user ID, or a trusted "
            "role ID, and configure discord.allowed_user_ids or "
            "discord.allowed_role_ids on PC-A. Do not grant everyone access.",
        )
    if not checks.get("guild_configured", False):
        add(
            "guild_id", "pc_a", "discord.guild_id",
            "Enable Discord Developer Mode, copy the intended server ID, "
            "and set discord.guild_id in PC-A YAML. Never guess a server.",
        )
    if not checks.get("channel_configured", False):
        add(
            "channel_id", "pc_a", "discord.channel_id",
            "Enable Discord Developer Mode, copy the intended private text "
            "channel ID, and set discord.channel_id in PC-A YAML.",
        )
    if not checks.get("credential_present", False):
        add(
            "bot_token", "pc_a", "discord.token_env",
            "Create or select your own Discord application bot in the "
            "Developer Portal, and set its bot token in the PC-A process "
            f"environment variable {config.token_env!r}. Never put the "
            "token in YAML, the diagnostic report, or chat messages.",
        )

    if not live_requested or requests_sent == 0:
        return actions

    if checks.get("bot_identity_verified") is not True:
        add(
            "verify_bot_access", "pc_a", "discord.token_env",
            "Check PC-A outbound HTTPS access to Discord and the configured "
            "bot credential. Review HTTP failure codes (401, 403, 429); "
            "respect rate limits and never copy a token into diagnostics.",
        )
        return actions

    if checks.get("guild_channel_match") is not True:
        add(
            "guild_and_channel_access", "discord_server", "discord.guild_id",
            "Confirm the bot was invited to the configured server and "
            "discord.guild_id and discord.channel_id describe the same "
            "server. Correct IDs in PC-A YAML if wrong; do not add the "
            "bot or create channels without approval.",
        )
        return actions

    if checks.get("channel_type_supported") is False:
        if checks.get("private_thread") is True:
            add(
                "private_thread_unverifiable", "discord_server", "discord.channel_id",
                "A private thread requires membership proof that this GET-only "
                "probe cannot establish. Select a normal private text channel "
                "for the first qualification; do not change permissions or "
                "join a thread automatically.",
            )
        else:
            add(
                "unsupported_channel_type", "pc_a", "discord.channel_id",
                "The configured Discord channel is not a text or public thread "
                "channel. Choose an intended private text channel and update "
                "discord.channel_id on PC-A; do not create one automatically.",
            )
        return actions

    if checks.get("public_thread_active") is False:
        add(
            "public_thread_inactive", "discord_server", "discord.channel_id",
            "The configured public thread is archived or its active state "
            "could not be verified. Prefer a normal private text channel, "
            "or manually inspect thread status before retrying. Do not "
            "automatically unarchive, post, or change Discord permissions.",
        )
        return actions

    if checks.get("view_channel") is False:
        add(
            "view_channel", "discord_server", None,
            "In the configured Discord channel's Permissions, grant the bot "
            "View Channel. Check @everyone, bot roles, and bot-specific "
            "overwrites; keep the channel private from other members.",
        )
    if checks.get("send_messages") is False:
        add(
            "send_messages", "discord_server", None,
            "In the configured Discord channel's Permissions, grant the bot "
            "Send Messages (or Send Messages in Threads for a public thread). "
            "Avoid granting Administrator or other unrelated privileges.",
        )
    if config.notify_completion and checks.get("read_message_history") is False:
        add(
            "read_message_history", "discord_server", None,
            "In the configured Discord channel's Permissions, grant the bot "
            "Read Message History to reconcile ambiguous completion notices. "
            "This permission is not evidence that a notice was delivered.",
        )
    if checks.get("guild_slash_commands_synced") is False:
        add(
            "guild_commands", "discord_server", None,
            "Confirm the Artifex bot has the application.commands install "
            "scope and can register commands in the intended guild. The "
            "normal Artifex bot startup synchronizes commands; check its "
            "startup log and avoid restarting an active production daemon "
            "merely to clear this diagnostic.",
        )
    if (
        config.notify_completion
        and checks.get("read_message_history") is True
        and checks.get("history_api_accessible") is False
    ):
        add(
            "history_api", "discord_server", None,
            "Read Message History permission alone did not establish the "
            "history API path. Review the Discord API response code, "
            "membership, and server/channel visibility. Do not resend "
            "uncertain notices based on an empty or failed history lookup.",
        )
    if (
        not actions
        and not all(
            checks.get(name, False)
            for name in ("view_channel", "send_messages", "guild_slash_commands_synced")
        )
    ):
        add(
            "incomplete_live_probe", "pc_a", None,
            "Discord's read-only probe ended before all checks could be "
            "observed. Review the HTTP error and restore access, then "
            "rerun the same onboarding or overview command with "
            "--discord-live. Do not treat missing checks as PASS.",
        )
    return actions
