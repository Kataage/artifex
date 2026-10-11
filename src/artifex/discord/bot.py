from __future__ import annotations

import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import discord
from discord import app_commands

from artifex.config.models import DiscordConfig
from artifex.discord.audit import DiscordQualificationAudit
from artifex.discord.models import (
    CommandResponse,
    Notification,
    NotificationKind,
    OperatorContext,
)
from artifex.discord.notifications import NotificationPolicy, completion_marker
from artifex.discord.router import DiscordCommandRouter
from artifex.discord.summary import DailySummaryBuilder


def token_from_environment(config: DiscordConfig) -> str:
    token = os.environ.get(config.token_env)
    if not token:
        raise RuntimeError(
            f"Discord is enabled but token environment variable is missing: "
            f"{config.token_env}"
        )
    return token


class ReviewView(discord.ui.View):
    def __init__(
        self,
        router: DiscordCommandRouter,
        review_id: str,
        audit: DiscordQualificationAudit | None = None,
    ) -> None:
        super().__init__(timeout=None)
        self._router = router
        self._review_id = review_id
        self._audit = audit

    @staticmethod
    def _context(interaction: discord.Interaction) -> OperatorContext:
        user = interaction.user
        role_ids: tuple[int, ...] = ()
        if isinstance(user, discord.Member):
            role_ids = tuple(role.id for role in user.roles)
        return OperatorContext(
            user_id=user.id,
            role_ids=role_ids,
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
        )

    async def _run(
        self,
        interaction: discord.Interaction,
        command: str,
    ) -> None:
        operator = self._context(interaction)
        response = self._router.route(
            operator,
            command,
            (self._review_id,),
        )
        await _respond(interaction, response)
        if self._audit is not None:
            self._audit.record_completed(
                command=command,
                response=response,
                operator=operator,
                interaction_id=interaction.id,
                guild_id=interaction.guild_id,
                channel_id=interaction.channel_id,
            )

    @discord.ui.button(
        label="Approve",
        style=discord.ButtonStyle.success,
        custom_id="artifex:approve",
    )
    async def approve(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[ReviewView],
    ) -> None:
        del button
        await self._run(interaction, "approve")

    @discord.ui.button(
        label="Regenerate",
        style=discord.ButtonStyle.primary,
        custom_id="artifex:retry",
    )
    async def regenerate(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[ReviewView],
    ) -> None:
        del button
        await self._run(interaction, "retry")

    @discord.ui.button(
        label="Alternate idea",
        style=discord.ButtonStyle.secondary,
        custom_id="artifex:alternate",
    )
    async def alternate(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[ReviewView],
    ) -> None:
        del button
        await self._run(interaction, "alternate")

    @discord.ui.button(
        label="Reject",
        style=discord.ButtonStyle.danger,
        custom_id="artifex:reject",
    )
    async def reject(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[ReviewView],
    ) -> None:
        del button
        await self._run(interaction, "reject")

    @discord.ui.button(
        label="Details",
        style=discord.ButtonStyle.secondary,
        custom_id="artifex:details",
    )
    async def details(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[ReviewView],
    ) -> None:
        del button
        await self._run(interaction, "details")


async def _respond(
    interaction: discord.Interaction,
    response: CommandResponse,
) -> None:
    content = response.message[:1900]
    if interaction.response.is_done():
        await interaction.followup.send(content, ephemeral=response.ephemeral)
    else:
        await interaction.response.send_message(
            content,
            ephemeral=response.ephemeral,
        )


async def find_completion_in_history(
    channel: discord.TextChannel | discord.Thread,
    *,
    pack_id: str,
    bot_user_id: int,
    limit: int = 200,
) -> str | None:
    """Return only a message ID whose exact author and Pack marker match."""
    marker = completion_marker(pack_id)
    async for message in channel.history(limit=limit):
        if (
            message.author.id == bot_user_id
            and message.content.startswith("**Artifex Pack completed**\n")
            and marker in message.content.splitlines()
        ):
            return str(message.id)
    return None


def assess_notification_channel_access(
    channel: discord.TextChannel | discord.Thread | None,
    *,
    require_history: bool = False,
    require_send: bool = True,
) -> tuple[bool, str]:
    """Check cached access without granting permission or mutating a thread.

    Sending and read-only history reconciliation have different requirements.
    An archived thread must never be implicitly unarchived by a notification,
    but its existing message history can still establish positive delivery.
    """
    if channel is None:
        return False, "Configured Discord notification channel is not cached or accessible."
    member = channel.guild.me
    if member is None:
        return False, "Discord bot membership is not resolved for the channel."
    permissions = channel.permissions_for(member)
    if not permissions.view_channel:
        return False, "Discord bot lacks View Channel permission."
    if isinstance(channel, discord.Thread):
        if channel.type is not discord.ChannelType.public_thread:
            return False, "Only public Discord threads are supported for notifications."
        if require_send:
            if channel.archived is not False:
                return False, "Discord public thread is archived or its state is unknown."
            if channel.locked is not False:
                return False, "Discord public thread is locked or its state is unknown."
            if not permissions.send_messages_in_threads:
                return False, "Discord bot lacks Send Messages in Threads permission."
    elif require_send and not permissions.send_messages:
        return False, "Discord bot lacks Send Messages permission."
    if require_history and not permissions.read_message_history:
        return False, "Discord bot lacks Read Message History permission."
    return True, "Discord notification channel permissions verified."


class ArtifexDiscordClient(discord.Client):
    def __init__(
        self,
        config: DiscordConfig,
        router: DiscordCommandRouter,
        summary: DailySummaryBuilder,
        audit: DiscordQualificationAudit | None = None,
    ) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        super().__init__(intents=intents)
        self._config = config
        self._router = router
        self._summary = summary
        self._audit = audit
        self._notifications = NotificationPolicy(config)
        self._last_summary_date: str | None = None
        self._summary_task: asyncio.Task[None] | None = None
        self.tree = app_commands.CommandTree(self)
        self._group = app_commands.Group(
            name="artifex",
            description="Control and review the local Artifex production daemon.",
        )
        self._register_commands()
        self.tree.add_command(self._group)

    async def setup_hook(self) -> None:
        if self._config.guild_id is not None:
            guild = discord.Object(id=self._config.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()
        self._summary_task = asyncio.create_task(self._daily_summary_loop())

    async def close(self) -> None:
        if self._summary_task is not None:
            self._summary_task.cancel()
            try:
                await self._summary_task
            except asyncio.CancelledError:
                pass
        await super().close()

    def notification_readiness(self) -> tuple[bool, str]:
        if not self.is_ready():
            return False, "Discord bot is not connected and ready."
        if self._config.channel_id is None:
            return False, "Discord notification channel_id is not configured."
        return assess_notification_channel_access(
            self._notification_channel(),
            require_history=self._config.notify_completion,
        )

    async def notify(self, notification: Notification) -> bool:
        if not self._notifications.enabled(notification.kind):
            return False
        channel = self._notification_channel()
        accessible, _ = assess_notification_channel_access(channel)
        if not accessible or channel is None:
            return False

        message = f"**{notification.title}**\n{notification.body}"[:1900]
        view: discord.ui.View | None = None
        if (
            notification.kind is NotificationKind.REVIEW
            and notification.review_id is not None
        ):
            view = ReviewView(self._router, notification.review_id, self._audit)
        if view is None:
            await channel.send(message)
        else:
            await channel.send(message, view=view)
        return True

    async def notify_completion(self, notification: Notification) -> str | None:
        """Return the Discord API-accepted completion message ID.

        None means the send was definitely not attempted. If the POST may
        have happened but the response ID is missing or invalid, raise:
        the durable outbox must quarantine that Pack rather than resend it.
        """
        if notification.kind is not NotificationKind.COMPLETION:
            raise ValueError("Completion receipts require a completion notification")
        if not self._notifications.enabled(notification.kind):
            return None
        channel = self._notification_channel()
        accessible, _ = assess_notification_channel_access(channel)
        if not accessible or channel is None:
            return None

        message = f"**{notification.title}**\n{notification.body}"[:1900]
        posted = await channel.send(message)
        message_id = posted.id
        if (
            type(message_id) is not int
            or not 0 < message_id < 2**64
        ):
            raise ValueError("Discord returned an invalid message ID after send")
        return str(message_id)

    async def find_completion_message(self, pack_id: str) -> str | None:
        """Look for a Pack marker in messages posted by this exact bot.

        Absence from bounded history is inconclusive and never authorizes a
        resend. Exceptions from the Discord API propagate to the maintenance
        worker, which leaves the Pack quarantined as uncertain.
        """
        channel = self._notification_channel()
        bot_user = self.user
        if channel is None or bot_user is None:
            return None
        accessible, reason = assess_notification_channel_access(
            channel, require_history=True, require_send=False
        )
        if not accessible:
            raise PermissionError(reason)
        return await find_completion_in_history(
            channel, pack_id=pack_id, bot_user_id=bot_user.id
        )

    async def verify_completion_message(
        self, pack_id: str, message_id: str,
    ) -> bool:
        """GET an exact saved message; require same bot and Pack marker.

        A missing, edited, or foreign message is not independently verified.
        An HTTP/permission failure propagates to maintenance for safe retry.
        This operation never posts, unarchives, or changes a notice state.
        """
        channel = self._notification_channel()
        bot_user = self.user
        if channel is None or bot_user is None:
            raise PermissionError("Configured Discord channel/bot is unavailable")
        accessible, reason = assess_notification_channel_access(
            channel, require_history=True, require_send=False,
        )
        if not accessible:
            raise PermissionError(reason)
        if (
            not isinstance(message_id, str)
            or not message_id.isascii()
            or not message_id.isdecimal()
            or not 0 < int(message_id) < 2**64
        ):
            raise ValueError("Invalid Discord message ID for read-only verification")
        try:
            message = await channel.fetch_message(int(message_id))
        except discord.NotFound:
            # 404 cannot authorize a resend: the post may have been deleted.
            return False
        return (
            message.id == int(message_id)
            and message.author.id == bot_user.id
            and message.content.startswith("**Artifex Pack completed**\n")
            and completion_marker(pack_id) in message.content.splitlines()
        )

    def _notification_channel(
        self,
    ) -> discord.TextChannel | discord.Thread | None:
        if self._config.channel_id is None:
            return None
        channel = self.get_channel(self._config.channel_id)
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return None
        if (
            self._config.guild_id is not None
            and channel.guild.id != self._config.guild_id
        ):
            return None
        return channel

    def _context(self, interaction: discord.Interaction) -> OperatorContext:
        user = interaction.user
        role_ids: tuple[int, ...] = ()
        if isinstance(user, discord.Member):
            role_ids = tuple(role.id for role in user.roles)
        return OperatorContext(
            user_id=user.id,
            role_ids=role_ids,
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
        )

    async def _route(
        self,
        interaction: discord.Interaction,
        name: str,
        *args: str,
    ) -> None:
        operator = self._context(interaction)
        response = self._router.route(operator, name, tuple(args))
        await _respond(interaction, response)
        if self._audit is not None:
            self._audit.record_completed(
                command=name,
                response=response,
                operator=operator,
                interaction_id=interaction.id,
                guild_id=interaction.guild_id,
                channel_id=interaction.channel_id,
            )

    def _register_commands(self) -> None:
        @self._group.command(name="status", description="Show daemon and inventory status.")
        async def status(interaction: discord.Interaction) -> None:
            await self._route(interaction, "status")

        @self._group.command(name="pause", description="Pause autonomous scheduling.")
        async def pause(interaction: discord.Interaction) -> None:
            await self._route(interaction, "pause")

        @self._group.command(name="resume", description="Resume autonomous scheduling.")
        async def resume(interaction: discord.Interaction) -> None:
            await self._route(interaction, "resume")

        @self._group.command(name="current", description="Show the current active Pack.")
        async def current(interaction: discord.Interaction) -> None:
            await self._route(interaction, "current")

        @self._group.command(name="queue", description="Show queued Packs.")
        async def queue(interaction: discord.Interaction) -> None:
            await self._route(interaction, "queue")

        @self._group.command(name="recent", description="Show recent terminal Packs.")
        async def recent(interaction: discord.Interaction) -> None:
            await self._route(interaction, "recent")

        @self._group.command(name="next", description="Show the next pending review.")
        async def next_review(interaction: discord.Interaction) -> None:
            await self._route(interaction, "next")

        @self._group.command(name="approve", description="Approve a pending review.")
        async def approve(interaction: discord.Interaction, review_id: str) -> None:
            await self._route(interaction, "approve", review_id)

        @self._group.command(name="reject", description="Reject a pending review.")
        async def reject(interaction: discord.Interaction, review_id: str) -> None:
            await self._route(interaction, "reject", review_id)

        @self._group.command(name="retry", description="Retry a review item or Scene.")
        async def retry(interaction: discord.Interaction, target_id: str) -> None:
            await self._route(interaction, "retry", target_id)

        @self._group.command(name="skip", description="Skip a review item or subject.")
        async def skip(interaction: discord.Interaction, target_id: str) -> None:
            await self._route(interaction, "skip", target_id)

        @self._group.command(
            name="alternate",
            description="Retire the reviewed Pack and request a new idea.",
        )
        async def alternate(interaction: discord.Interaction, review_id: str) -> None:
            await self._route(interaction, "alternate", review_id)

        @self._group.command(name="details", description="Show review details.")
        async def details(interaction: discord.Interaction, review_id: str) -> None:
            await self._route(interaction, "details", review_id)

        @self._group.command(name="character", description="Inspect a character profile.")
        async def character(interaction: discord.Interaction, character_id: str) -> None:
            await self._route(interaction, "character", character_id)

        @self._group.command(name="series", description="Inspect a Series.")
        async def series(interaction: discord.Interaction, series_id: str) -> None:
            await self._route(interaction, "series", series_id)

        @self._group.command(
            name="signals",
            description="Inspect Trend/Seasonal signals and source health.",
        )
        async def signals(interaction: discord.Interaction) -> None:
            await self._route(interaction, "signals")

        @self._group.command(
            name="notices",
            description="Show completion notification status and unresolved Packs.",
        )
        async def notices(interaction: discord.Interaction) -> None:
            await self._route(interaction, "notices")

        @self._group.command(
            name="notice",
            description="Inspect a single Pack completion notification.",
        )
        async def notice(interaction: discord.Interaction, pack_id: str) -> None:
            await self._route(interaction, "notice", pack_id)

        @self._group.command(
            name="notice-dismiss",
            description="Explicitly stop retries for an uncertain Pack notice.",
        )
        async def notice_dismiss(
            interaction: discord.Interaction,
            pack_id: str,
            reason: str,
            confirmation: str,
        ) -> None:
            await self._route(
                interaction,
                "notice-dismiss",
                pack_id,
                reason,
                confirmation,
            )

    async def _daily_summary_loop(self) -> None:
        timezone = ZoneInfo(self._config.daily_summary_timezone)
        while not self.is_closed():
            now = datetime.now(timezone)
            date_key = now.date().isoformat()
            if (
                now.hour == self._config.daily_summary_hour_local
                and self._last_summary_date != date_key
            ):
                sent = await self.notify(
                    Notification(
                        kind=NotificationKind.DAILY_SUMMARY,
                        title="Artifex daily summary",
                        body=self._summary.build(),
                    )
                )
                if sent:
                    self._last_summary_date = date_key
            await asyncio.sleep(60)


def run_discord_bot(
    config: DiscordConfig,
    router: DiscordCommandRouter,
    summary: DailySummaryBuilder,
) -> None:
    if not config.enabled:
        raise RuntimeError("Discord integration is disabled")
    token = token_from_environment(config)
    client = ArtifexDiscordClient(config, router, summary)
    client.run(token, log_handler=None)
