"""Durable completion dispatch with fail-closed ambiguous Discord outcomes.

Discord's send API does not offer a Pack-scoped idempotency key. Claiming
before send prevents a controller restart from blindly duplicating a message,
but can leave an uncertain or undelivered notice. Never count such a notice
as successfully delivered or retry it without reconciliation.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import CompletionNotificationRow, PackRow
from artifex.discord.models import Notification, NotificationKind
from artifex.discord.notifications import completion_marker
from artifex.domain import PackState
from artifex.telemetry import EventSeverity, TelemetryRepository


def _valid_message_id(value: object) -> bool:
    """Accept only a genuine Discord 64-bit snowflake string."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= 20
        and value.isascii()
        and value.isdecimal()
        and 0 < int(value) < 2**64
    )


class CompletionSink(Protocol):
    def is_ready(self) -> bool: ...
    async def notify_completion(self, notification: Notification) -> str | None: ...
    async def find_completion_message(self, pack_id: str) -> str | None: ...


class CompletionNotificationMaintenance:
    def __init__(
        self,
        database: Database,
        telemetry: TelemetryRepository,
        sink: CompletionSink,
        *,
        batch_size: int = 10,
        reconcile_batch_size: int = 3,
        reconcile_interval_seconds: float = 120.0,
    ) -> None:
        if batch_size < 1 or reconcile_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if reconcile_interval_seconds < 0:
            raise ValueError("reconcile interval cannot be negative")
        self._database = database
        self._telemetry = telemetry
        self._sink = sink
        self._batch_size = batch_size
        self._reconcile_batch_size = reconcile_batch_size
        self._reconcile_interval_seconds = reconcile_interval_seconds
        self._last_reconcile_monotonic: float | None = None

        # Only reconcile claims from a *previous* process. This is called
        # before the daemon begins its maintenance loop, not during a send.
        with self._database.session() as session:
            abandoned = session.scalars(
                select(CompletionNotificationRow).where(
                    CompletionNotificationRow.state == "sending"
                )
            ).all()
            abandoned_ids = [item.pack_id for item in abandoned]
            for item in abandoned:
                item.state = "uncertain"
                item.updated_at = datetime.now(UTC)
        for pack_id in abandoned_ids:
            self._telemetry.record(
                "discord.completion_uncertain",
                EventSeverity.WARNING,
                {"pack_id": pack_id, "reason": "process_restart_during_send"},
            )

    async def maintain(self) -> int:
        if not self._sink.is_ready():
            return 0
        with self._database.session() as session:
            ids = session.scalars(
                select(CompletionNotificationRow.pack_id)
                .where(CompletionNotificationRow.state == "pending")
                .order_by(
                    CompletionNotificationRow.created_at.asc(),
                    CompletionNotificationRow.pack_id.asc(),
                )
                .limit(self._batch_size)
            ).all()

        delivered = 0
        for pack_id in ids:
            with self._database.session() as session:
                item = session.get(CompletionNotificationRow, pack_id)
                pack = session.get(PackRow, pack_id)
                if item is None or item.state != "pending":
                    continue
                if (
                    pack is None
                    or pack.state != PackState.FINALIZED.value
                    or pack.checkpoint_json.get("archive_complete") is not True
                    or not pack.checkpoint_json.get("archive_manifest_sha256")
                ):
                    # Do not transmit an unverified or withdrawn Pack.
                    item.state = "uncertain"
                    item.updated_at = datetime.now(UTC)
                    continue
                kind = pack.format_type
                item.state = "sending"
                item.updated_at = datetime.now(UTC)

            notice = Notification(
                kind=NotificationKind.COMPLETION,
                title="Artifex Pack completed",
                body=(
                    f"Pack: {pack_id}\nFormat: {kind}\nArchive: finalized\n"
                    f"{completion_marker(pack_id)}"
                ),
            )
            try:
                message_id = await self._sink.notify_completion(notice)
            except Exception as exc:  # noqa: BLE001
                # It is unknown whether the remote API accepted the message.
                self._set_state(pack_id, "uncertain")
                self._telemetry.record(
                    "discord.completion_uncertain",
                    EventSeverity.WARNING,
                    {"pack_id": pack_id, "reason": type(exc).__name__},
                )
                continue

            if message_id is None:
                # The sink definitively did not issue a send (no usable channel).
                self._set_state(pack_id, "pending")
                continue

            if not _valid_message_id(message_id):
                # A POST could already have been accepted by Discord.
                # Never convert an invalid receipt into a safe retry.
                self._set_state(pack_id, "uncertain")
                self._telemetry.record(
                    "discord.completion_uncertain",
                    EventSeverity.WARNING,
                    {"pack_id": pack_id, "reason": "invalid_send_message_id"},
                )
                continue

            self._set_state(
                pack_id, "sent", discord_message_id=message_id,
            )
            self._telemetry.record(
                "discord.completion_sent",
                EventSeverity.INFO,
                {"pack_id": pack_id, "discord_message_id": message_id},
            )
            delivered += 1
        await self._reconcile_uncertain()
        return delivered

    async def _reconcile_uncertain(self) -> None:
        """Find confirmed posts; never retry based on a negative history scan."""
        now = time.monotonic()
        if (
            self._last_reconcile_monotonic is not None
            and now - self._last_reconcile_monotonic
            < self._reconcile_interval_seconds
        ):
            return
        self._last_reconcile_monotonic = now

        with self._database.session() as session:
            ids = session.scalars(
                select(CompletionNotificationRow.pack_id)
                .where(CompletionNotificationRow.state == "uncertain")
                .order_by(
                    CompletionNotificationRow.updated_at.asc(),
                    CompletionNotificationRow.pack_id.asc(),
                )
                .limit(self._reconcile_batch_size)
            ).all()

        for pack_id in ids:
            try:
                message_id = await self._sink.find_completion_message(pack_id)
            except Exception as exc:  # noqa: BLE001
                self._telemetry.record(
                    "discord.completion_reconcile_failed",
                    EventSeverity.WARNING,
                    {"pack_id": pack_id, "reason": type(exc).__name__},
                )
                continue

            if message_id is not None and not _valid_message_id(message_id):
                self._telemetry.record(
                    "discord.completion_reconcile_failed",
                    EventSeverity.WARNING,
                    {"pack_id": pack_id, "reason": "invalid_remote_message_id"},
                )
                continue

            with self._database.session() as session:
                item = session.get(CompletionNotificationRow, pack_id)
                if item is None or item.state != "uncertain":
                    continue
                if message_id is None:
                    # Rotate past this Pack to avoid repeatedly scanning only
                    # the oldest uncertain records. Absence proves nothing.
                    item.updated_at = datetime.now(UTC)
                    continue
                item.state = "sent"
                item.discord_message_id = message_id
                item.updated_at = datetime.now(UTC)

            self._telemetry.record(
                "discord.completion_reconciled",
                EventSeverity.INFO,
                {"pack_id": pack_id, "discord_message_id": message_id},
            )

    def _set_state(
        self, pack_id: str, state: str, *, discord_message_id: str | None = None,
    ) -> None:
        if discord_message_id is not None and (
            state != "sent" or not _valid_message_id(discord_message_id)
        ):
            raise ValueError("Only a valid Discord send receipt can be stored")
        with self._database.session() as session:
            item = session.get(CompletionNotificationRow, pack_id)
            if item is None:
                raise KeyError(f"completion notice missing: {pack_id}")
            if item.state != "sending":
                raise RuntimeError(
                    f"completion notice state changed during dispatch: {pack_id}"
                )
            item.state = state
            if discord_message_id is not None:
                item.discord_message_id = discord_message_id
            item.updated_at = datetime.now(UTC)
