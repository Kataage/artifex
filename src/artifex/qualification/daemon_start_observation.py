"""Read-only PC-A daemon-loop provenance for starting unattended GPU monitoring.

Daemon status is a *start prerequisite* only. A fresh operations heartbeat
plus a still-open continuous-loop marker is not an authenticated lifetime
receipt, eight-hour monitor, or permission to submit any GPU work.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy.engine import make_url

from artifex.config.models import ArtifexSettings

DaemonStartState = Literal[
    "observed_active_loop",
    "database_unavailable",
    "heartbeat_missing",
    "heartbeat_invalid",
    "heartbeat_stale",
    "loop_marker_missing",
    "loop_not_active",
    "loop_marker_invalid",
]


class DaemonStartObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: DaemonStartState
    ready: bool
    heartbeat_age_seconds: float | None = None
    # No run IDs, raw SQL values, filesystem paths or exception messages.
    database_read_only: Literal[True] = True
    processes_launched: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    continuous_run_proven: Literal[False] = False
    production_qualified: Literal[False] = False


def _observed(
    state: DaemonStartState, age: float | None = None,
) -> DaemonStartObservation:
    return DaemonStartObservation(
        state=state, ready=state == "observed_active_loop",
        heartbeat_age_seconds=round(age, 3) if age is not None else None,
    )


def observe_active_production_daemon(
    settings: ArtifexSettings,
    *,
    now: datetime | None = None,
) -> DaemonStartObservation:
    """Read an existing SQLite DB, never create/migrate it or edit any row."""
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        return _observed("heartbeat_invalid")
    try:
        url = make_url(settings.storage.database_url)
        if url.get_backend_name() != "sqlite" or not url.database:
            return _observed("database_unavailable")
        source_path = Path(url.database).expanduser()
        if source_path.is_symlink():
            return _observed("database_unavailable")
        db_path = source_path.resolve(strict=True)
        if not db_path.is_file():
            return _observed("database_unavailable")
        # Read-only URI prevents accidental implicit DB creation. Avoid
        # executing any PRAGMA, migration or writes on the production DB.
        with sqlite3.connect(
            db_path.as_uri() + "?mode=ro", uri=True, timeout=5,
        ) as conn:
            row = conn.execute(
                "SELECT value_json FROM settings WHERE key = ?",
                ("runtime.heartbeat",),
            ).fetchone()
            if row is None:
                return _observed("heartbeat_missing")
            raw_value = json.loads(row[0])
            if not isinstance(raw_value, dict):
                return _observed("heartbeat_invalid")
            stamp = raw_value.get("at")
            if not isinstance(stamp, str):
                return _observed("heartbeat_invalid")
            heartbeat = datetime.fromisoformat(stamp)
            if heartbeat.tzinfo is None or heartbeat.utcoffset() is None:
                return _observed("heartbeat_invalid")
            age = (current - heartbeat).total_seconds()
            if age < -30:
                return _observed("heartbeat_invalid")
            max_age = max(
                settings.operations.heartbeat_stale_seconds,
                settings.operations.heartbeat_interval_seconds * 2,
            )
            if age > max_age:
                return _observed("heartbeat_stale", age)
            # Only an actual run_forever lifecycle writes these protected
            # event types. A manual run_once cannot manufacture a start.
            # Ordering by ID avoids Windows/SQLite date coercion ambiguity.
            last = conn.execute(
                "SELECT event_type, payload_json FROM agent_events "
                "WHERE event_type IN (?, ?) ORDER BY id DESC LIMIT 1",
                ("daemon.loop_started", "daemon.loop_stopped"),
            ).fetchone()
            if last is None:
                return _observed("loop_marker_missing", age)
            event_type, payload_json = last
            payload = json.loads(payload_json)
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("run_id"), str)
                or re.fullmatch(r"[0-9a-f]{32}", payload["run_id"]) is None
            ):
                return _observed("loop_marker_invalid", age)
            if event_type != "daemon.loop_started":
                return _observed("loop_not_active", age)
            return _observed("observed_active_loop", age)
    except (
        OSError, sqlite3.Error, ValueError, TypeError, KeyError,
        json.JSONDecodeError,
    ):
        # Never include the SQLite location, exception detail, event payload,
        # bearer token, or daemon run ID in the externally returned report.
        return _observed("database_unavailable")
