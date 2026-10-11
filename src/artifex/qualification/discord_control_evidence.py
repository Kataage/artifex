"""Bounded read-only triage of saved Discord control interaction evidence.

This is a diagnostic only: `discord_controls` PASS still requires the
qualification service to revalidate the persisted exact six interaction events.
Never infer a Discord response from a configuration preflight or a completion
notification receipt.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import make_url

from artifex.config.models import ArtifexSettings

_REQUIRED = ("approve", "pause", "reject", "resume", "retry", "status")
_SCAN_LIMIT = 5000

ControlStatus = Literal[
    "disabled", "no_session", "missing_config", "missing_database",
    "unsupported_backend", "missing_schema", "unavailable",
    "observed", "incomplete",
]


class DiscordControlEvidence(BaseModel):
    """Historical matching candidates, not a production qualification pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ControlStatus
    session_id: str | None = None
    required_commands: tuple[str, ...] = _REQUIRED
    observed_commands: tuple[str, ...] = ()
    missing_commands: tuple[str, ...] = ()
    matched_distinct_interactions: int | None = Field(default=None, ge=0)
    examined_recent_events: int | None = Field(default=None, ge=0)
    bounded_scan_limit: int = _SCAN_LIMIT
    scan_may_be_truncated: bool = False
    next_action: str
    independent_discord_response_check: Literal[False] = False
    recorded_stage_revalidated: Literal[False] = False
    qualifies_discord_controls: Literal[False] = False


def _unknown(
    status: ControlStatus, message: str, session_id: str | None,
) -> DiscordControlEvidence:
    return DiscordControlEvidence(
        status=status, session_id=session_id, next_action=message,
    )


def _created_at(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    # SQLAlchemy's SQLite DateTime storage omits tzinfo even with timezone=True.
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def inspect_discord_control_evidence(
    settings: ArtifexSettings,
    *,
    session_id: str | None,
    session_started: datetime | None,
    now: datetime,
) -> DiscordControlEvidence:
    """Inspect at most 5000 saved control receipts from the configured PC-A DB."""
    if not settings.discord.enabled:
        return _unknown("disabled", "Discord is disabled in PC-A settings.", session_id)
    if session_id is None or session_started is None:
        return _unknown(
            "no_session",
            "Start an explicit real-machine qualification session before counting "
            "Discord interactions; historical events from another session are not proof.",
            session_id,
        )
    cfg = settings.discord
    if cfg.guild_id is None or cfg.channel_id is None or not (
        cfg.allowed_user_ids or cfg.allowed_role_ids
    ):
        return _unknown(
            "missing_config",
            "Configure the target Discord guild, channel, and authorized users/roles "
            "before attempting the six real control interactions.",
            session_id,
        )
    url = make_url(settings.storage.database_url)
    if url.get_backend_name() != "sqlite" or url.database in ("", None, ":memory:"):
        return _unknown(
            "unsupported_backend",
            "This PC-A snapshot supports only existing on-disk SQLite; it does not "
            "inspect other database engines or create a new database.",
            session_id,
        )
    assert url.database is not None
    path = Path(url.database).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        return _unknown(
            "unavailable", "SQLite location is symlinked; no evidence was trusted.",
            session_id,
        )
    if not path.is_file():
        return _unknown(
            "missing_database",
            "The PC-A SQLite event database is missing. This is not zero interactions.",
            session_id,
        )

    try:
        with closing(sqlite3.connect(
            f"{path.as_uri()}?mode=ro", uri=True, timeout=2,
        )) as db:
            db.execute("PRAGMA query_only=ON")
            columns = {
                str(row[1]) for row in db.execute("PRAGMA table_info(agent_events)")
            }
            if not {"id", "event_type", "created_at", "payload_json"} <= columns:
                return _unknown(
                    "missing_schema",
                    "The operational events schema is missing or outdated. "
                    "Do not infer a successful Discord control.",
                    session_id,
                )
            rows = db.execute(
                "SELECT id, created_at, payload_json FROM agent_events "
                "WHERE event_type = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                ("discord.interaction_completed", _SCAN_LIMIT),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return _unknown(
            "unavailable",
            "Read-only PC-A Discord interaction evidence could not be read. "
            "No qualification status was changed.",
            session_id,
        )

    start = session_started.astimezone(UTC) if session_started.tzinfo else (
        session_started.replace(tzinfo=UTC)
    )
    stop = now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)
    observed: set[str] = set()
    seen_interactions: set[str] = set()
    # Align ordering and eligibility with QualificationService's original
    # bounded first-pass scan. This diagnostic intentionally does not read
    # mutable qualification PASS details or verify a recorded stage.
    for _event_id, timestamp, raw_payload in rows:
        event_time = _created_at(timestamp)
        if event_time is None or not start <= event_time <= stop:
            continue
        try:
            payload = json.loads(raw_payload)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("response_ok") is not True:
            continue
        command = payload.get("command")
        if not isinstance(command, str) or command not in _REQUIRED:
            continue
        if (payload.get("guild_id") != cfg.guild_id
                or payload.get("channel_id") != cfg.channel_id):
            continue
        user_id = payload.get("user_id")
        roles = payload.get("role_ids")
        if type(user_id) is not int or not isinstance(roles, list):
            continue
        if not all(type(role) is int for role in roles):
            continue
        if user_id not in cfg.allowed_user_ids and not set(roles).intersection(
            cfg.allowed_role_ids
        ):
            continue
        interaction_id = payload.get("interaction_id")
        if (not isinstance(interaction_id, str) or not interaction_id.isascii()
                or not interaction_id.isdecimal() or int(interaction_id) <= 0
                or interaction_id in seen_interactions):
            continue
        seen_interactions.add(interaction_id)
        observed.add(command)

    missing = tuple(sorted(set(_REQUIRED) - observed))
    found = tuple(sorted(observed))
    truncated = len(rows) == _SCAN_LIMIT
    if missing:
        advice = (
            "Missing saved authorized Discord response evidence for: "
            + ", ".join(missing)
            + ". Perform those controls through genuine authorized Discord "
            "interactions during this session; never synthesize event rows."
        )
        if truncated:
            advice += (
                " The bounded scan may exclude older receipts; inspect the "
                "qualification session before interpreting missing controls."
            )
    else:
        advice = (
            "All six control types have candidate saved receipts. Run the "
            "authoritative qualify collect and qualify verify commands for this "
            "specific session. These candidates are not a stage PASS and have "
            "not been independently revalidated."
        )
    return DiscordControlEvidence(
        status="incomplete" if missing else "observed",
        session_id=session_id,
        observed_commands=found,
        missing_commands=missing,
        matched_distinct_interactions=len(found),
        examined_recent_events=len(rows),
        scan_may_be_truncated=truncated,
        next_action=advice,
    )
