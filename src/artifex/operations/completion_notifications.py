"""Durable completion dispatch with fail-closed ambiguous Discord outcomes.

Discord's send API does not offer a Pack-scoped idempotency key. Claiming
before send prevents a controller restart from blindly duplicating a message,
but can leave an uncertain or undelivered notice. Never count such a notice
as successfully delivered or retry it without reconciliation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import CompletionNotificationRow, PackRow
from artifex.discord.models import Notification, NotificationKind
from artifex.domain import PackState
from artifex.telemetry import EventSeverity, TelemetryRepository


class CompletionSink(Protocol):
    def is_ready(self) -> bool: ...
    async def notify(self, notification: Notification) -> bool: ...


class CompletionNotificationMaintenance:
    def __init__(
        self,
        database: Database,
        telemetry: TelemetryRepository,
        sink: CompletionSink,
        *,
        batch_size: int = 10,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._database = database
        self._telemetry = telemetry
        self._sink = sink
        self._batch_size = batch_size

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
                body=f"Pack: {pack_id}\nFormat: {kind}\nArchive: finalized",
            )
            try:
                sent = await self._sink.notify(notice)
            except Exception as exc:  # noqa: BLE001
                # It is unknown whether the remote API accepted the message.
                self._set_state(pack_id, "uncertain")
                self._telemetry.record(
                    "discord.completion_uncertain",
                    EventSeverity.WARNING,
                    {"pack_id": pack_id, "reason": type(exc).__name__},
                )
                continue

            if not sent:
                # The sink explicitly returned without attempting delivery,
                # e.g. the configured channel is unavailable.
                self._set_state(pack_id, "pending")
                continue

            self._set_state(pack_id, "sent")
            self._telemetry.record(
                "discord.completion_sent",
                EventSeverity.INFO,
                {"pack_id": pack_id},
            )
            delivered += 1
        return delivered

    def _set_state(self, pack_id: str, state: str) -> None:
        with self._database.session() as session:
            item = session.get(CompletionNotificationRow, pack_id)
            if item is None:
                raise KeyError(f"completion notice missing: {pack_id}")
            if item.state != "sending":
                raise RuntimeError(
                    f"completion notice state changed during dispatch: {pack_id}"
                )
            item.state = state
            item.updated_at = datetime.now(UTC)
