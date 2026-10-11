"""Local read-only evidence snapshot for Discord completion notifications.

This is a bounded aggregate of *saved* SQLite facts, not a Discord API probe,
fresh remote-message check, control-response receipt, or stage qualification.
Never create/migrate the DB or turn a missing schema into zero notices.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import make_url

from artifex.config.models import ArtifexSettings

SnapshotStatus = Literal[
    "disabled", "missing_database", "unsupported_backend",
    "missing_schema", "unavailable", "observed", "inconsistent",
]

_REQUIRED_COLUMNS = frozenset({
    "state", "discord_message_id", "discord_verified_at",
})
_STATES = frozenset({"pending", "sending", "uncertain", "sent", "dismissed"})


class DiscordCompletionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SnapshotStatus
    total: int | None = Field(default=None, ge=0)
    pending: int | None = Field(default=None, ge=0)
    sending: int | None = Field(default=None, ge=0)
    uncertain: int | None = Field(default=None, ge=0)
    dismissed: int | None = Field(default=None, ge=0)
    sent_api_only: int | None = Field(default=None, ge=0)
    sent_history_verified: int | None = Field(default=None, ge=0)
    sent_without_receipt: int | None = Field(default=None, ge=0)
    inconsistent_rows: int | None = Field(default=None, ge=0)
    next_action: str
    independent_current_discord_check: Literal[False] = False
    operator_read_receipt_verified: Literal[False] = False
    qualifies_discord_controls: Literal[False] = False


def _unavailable(status: SnapshotStatus, guidance: str) -> DiscordCompletionEvidence:
    return DiscordCompletionEvidence(status=status, next_action=guidance)


def inspect_discord_completion_evidence(
    settings: ArtifexSettings,
) -> DiscordCompletionEvidence:
    """Inspect the configured PC-A SQLite file in explicit read-only mode."""
    if not settings.discord.enabled:
        return _unavailable("disabled", "Discord is disabled in PC-A settings.")
    url = make_url(settings.storage.database_url)
    if url.get_backend_name() != "sqlite" or url.database in (None, "", ":memory:"):
        return _unavailable(
            "unsupported_backend",
            "This read-only snapshot supports an existing on-disk SQLite DB only; "
            "it does not inspect or modify other database backends.",
        )
    assert url.database is not None
    path = Path(url.database).expanduser().absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        return _unavailable(
            "unavailable", "SQLite path is symlinked; refusing an ambiguous snapshot.",
        )
    if not path.is_file():
        return _unavailable(
            "missing_database",
            "PC-A SQLite completion evidence has not been found. "
            "Do not treat an absent database as zero notices.",
        )

    try:
        # mode=ro prevents SQLite from creating a database or writing rows.
        # Never initialize/migrate schema from a read-only overview.
        with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=2) as db:
            db.execute("PRAGMA query_only=ON")
            columns = {
                str(row[1]) for row in db.execute(
                    "PRAGMA table_info(completion_notifications)"
                )
            }
            if not _REQUIRED_COLUMNS <= columns:
                return _unavailable(
                    "missing_schema",
                    "SQLite completion evidence schema is unavailable or outdated. "
                    "Inspect PC-A migration status before qualification.",
                )
            aggregates = db.execute(
                "SELECT state, discord_message_id IS NOT NULL, "
                "discord_verified_at IS NOT NULL, COUNT(*) "
                "FROM completion_notifications "
                "GROUP BY state, discord_message_id IS NOT NULL, "
                "discord_verified_at IS NOT NULL"
            ).fetchall()
    except (OSError, sqlite3.Error):
        return _unavailable(
            "unavailable",
            "PC-A SQLite completion evidence could not be read safely. "
            "Inspect database access; no stage was changed.",
        )

    counts = {
        "pending": 0, "sending": 0, "uncertain": 0, "dismissed": 0,
        "sent_api_only": 0, "sent_history_verified": 0,
        "sent_without_receipt": 0, "inconsistent_rows": 0,
    }
    total = 0
    for state, has_id, has_verification, count in aggregates:
        total += count
        if state not in _STATES or (has_verification and (
            state != "sent" or not has_id
        )) or (state != "sent" and has_id):
            counts["inconsistent_rows"] += count
        elif state == "sent":
            if has_verification:
                counts["sent_history_verified"] += count
            elif has_id:
                counts["sent_api_only"] += count
            else:
                counts["sent_without_receipt"] += count
        else:
            counts[state] += count

    inconsistent = counts["inconsistent_rows"] > 0
    if inconsistent:
        guidance = (
            "Discord completion records have contradictory state/receipt fields. "
            "Review local SQLite provenance; do not resend or claim delivery."
        )
    elif counts["uncertain"] or counts["sending"]:
        guidance = (
            "Inspect unresolved notices with /artifex notices and the safe "
            "read-only history reconciliation. Never resend an uncertain notice."
        )
    elif counts["sent_api_only"] or counts["sent_without_receipt"]:
        guidance = (
            "Some API-accepted notices lack independent message verification. "
            "Keep Read Message History access available; this snapshot never "
            "performs a Discord GET and cannot prove current delivery."
        )
    elif counts["sent_history_verified"]:
        guidance = (
            "Saved message identities were previously verified against Discord. "
            "This is not a current GET, operator read receipt, or proof of "
            "the six authorized Discord control interactions."
        )
    else:
        guidance = (
            "No completion notices are stored. Wait for genuine finalized Pack "
            "production; do not send artificial notifications for qualification."
        )

    return DiscordCompletionEvidence(
        status="inconsistent" if inconsistent else "observed",
        total=total, next_action=guidance, **counts,
    )
