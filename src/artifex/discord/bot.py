from __future__ import annotations

import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import discord
from discord import app_commands

from artifex.config.models import DiscordConfig
from artifex.discord.models import (
    CommandResponse,
    Notification,
    NotificationKind,
    OperatorContext,
)
from artifex.discord.notifications import NotificationPolicy
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
    ) -> None:
        super().__init__(timeout=None)
        self._router = router
        self._review_id = review_id

    @staticmethod
    def _context(interaction: discord.Interaction) -> OperatorContext:
        user = interaction.user
        role_ids: tuple[int, ...] = ()
        if isinstance(user, discord.Member):
            role_ids = tuple(role.id for role in user.roles)
        return OperatorContext(user_id=user.id, role_ids=role_ids)

    async def _run(
        self,
        interaction: discord.Interaction,
        command: str,
    ) -> None:
        response = self._router.route(
            self._context(interaction),
            command,
            (self._review_id,),
        )
        await _respond(interaction, response)

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


class ArtifexDiscordClient(discord.Client):
    def __init__(
        self,
        config: DiscordConfig,
        router: DiscordCommandRouter,
        summary: DailySummaryBuilder,
    ) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        super().__init__(intents=intents)
        self._config = config
        self._router = router
        self._summary = summary
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

    async def notify(self, notification: Notification) -> bool:
        if not self._notifications.enabled(notification.kind):
            return False
        channel = self._notification_channel()
        if channel is None:
            return False

        message = f"**{notification.title}**\n{notification.body}"[:1900]
        view: discord.ui.View | None = None
        if (
            notification.kind is NotificationKind.REVIEW
            and notification.review_id is not None
        ):
            view = ReviewView(self._router, notification.review_id)
        await channel.send(message, view=view)
        return True

    def _notification_channel(
        self,
    ) -> discord.TextChannel | discord.Thread | None:
        if self._config.channel_id is None:
            return None
        channel = self.get_channel(self._config.channel_id)
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            return channel
        return None

    def _context(self, interaction: discord.Interaction) -> OperatorContext:
        user = interaction.user
        role_ids: tuple[int, ...] = ()
        if isinstance(user, discord.Member):
            role_ids = tuple(role.id for role in user.roles)
        return OperatorContext(user_id=user.id, role_ids=role_ids)

    async def _route(
        self,
        interaction: discord.Interaction,
        name: str,
        *args: str,
    ) -> None:
        response = self._router.route(self._context(interaction), name, tuple(args))
        await _respond(interaction, response)

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
