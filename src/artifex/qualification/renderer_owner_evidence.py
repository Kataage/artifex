"""Persist and validate remote PC-B snapshots without issuing stage PASS.

Snapshots come from the existing authenticated render-node attestation
channel. A valid snapshot is observational evidence, never authorization
to restart a GPU process or proof of uninterrupted production operation.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from artifex.render_node.models import RemoteRendererOwnerAudit

_OBSERVATION_NAME = re.compile(r"^owner-observation-[0-9a-f]{32}\.json$")
_REQUIRED_CHECKS = frozenset({
    "native_windows",
    "protected_configuration",
    "scheduler_policy",
    "scheduler_running",
    "receipt",
    "process_identity",
    "launcher_identity",
    "tcp_ownership",
    "snapshot_consistency",
})
_MAX_BYTES = 64 * 1024


def persist_owner_observation(
    root: Path, session_id: str, observed: RemoteRendererOwnerAudit,
) -> dict[str, object]:
    """Record one unique, size-limited snapshot in this qualification session."""
    target_dir = root / session_id
    if any(item.is_symlink() for item in (target_dir, *target_dir.parents)):
        raise ValueError("Refusing symlinked qualification evidence directory")
    if not target_dir.is_dir():
        raise ValueError("Qualification session directory does not exist")
    payload = {
        "session_id": session_id,
        "node_id": observed.node_id,
        "observed": observed.model_dump(mode="json"),
    }
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if len(raw) > _MAX_BYTES:
        raise ValueError("Remote PC-B ownership evidence too large")
    name = f"owner-observation-{uuid4().hex}.json"
    target = target_dir / name
    with target.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    return {
        "file": name,
        "node_id": observed.node_id,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "captured_utc": observed.audit.captured_utc.isoformat(),
        "status": observed.audit.status,
    }


def verify_owner_observation(
    root: Path,
    session_id: str,
    node_id: str,
    entries: tuple[dict[str, object], ...],
    *,
    created_at: datetime,
    now: datetime | None = None,
    max_age_seconds: int = 300,
) -> str | None:
    """Return a fail-closed explanation, or None for a fresh stable snapshot."""
    if not entries:
        return "fresh authenticated PC-B owner snapshot is missing"
    # Only the last explicit observation counts; never fall back to an older
    # stable snapshot if the newest is blocked or has been modified.
    latest = entries[-1]
    name, expected_sha = latest.get("file"), latest.get("sha256")
    if (
        not isinstance(name, str)
        or not _OBSERVATION_NAME.fullmatch(name)
        or not isinstance(expected_sha, str)
        or not re.fullmatch("[0-9a-f]{64}", expected_sha)
        or latest.get("node_id") != node_id
    ):
        return "PC-B snapshot reference is invalid or node changed"
    target = root / session_id / name
    if any(item.is_symlink() for item in (target, *target.parents)):
        return "PC-B owner snapshot path is symlinked"
    try:
        if not target.is_file() or target.stat().st_size > _MAX_BYTES:
            return "PC-B owner snapshot file is absent or oversized"
        raw = target.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            return "PC-B owner snapshot integrity hash differs"
        data: Any = json.loads(raw)
        if not isinstance(data, dict):
            return "PC-B snapshot data is not an object"
        if data.get("session_id") != session_id or data.get("node_id") != node_id:
            return "PC-B snapshot is bound to another session or node"
        observed = RemoteRendererOwnerAudit.model_validate(data.get("observed"))
        if observed.node_id != node_id:
            return "PC-B observation node ID mismatch"
    except (OSError, ValueError, TypeError):
        return "PC-B snapshot data cannot be verified"
    audit = observed.audit
    if audit.status != "observed_stable" or not audit.process_observation_verified:
        return f"PC-B actual ownership observation is {audit.status}, not stable"
    if not _REQUIRED_CHECKS.issubset(audit.checks):
        return "PC-B observation lacks mandatory checks"
    if any(audit.checks[key].status != "pass" for key in _REQUIRED_CHECKS):
        return "PC-B observation includes an unproven required check"
    if audit.actual_listener_pid is None or audit.actual_process_started_utc is None:
        return "PC-B observation lacks actual renderer identity"
    captured = audit.captured_utc
    start = created_at
    current = now or datetime.now(UTC)
    if captured.tzinfo is None or start.tzinfo is None or current.tzinfo is None:
        return "PC-B evidence timestamps lack timezone"
    if captured < start - timedelta(seconds=30):
        return "PC-B evidence predates the qualification session"
    if captured > current + timedelta(seconds=30):
        return "PC-B evidence timestamp is in the future"
    if current - captured > timedelta(seconds=max_age_seconds):
        return "PC-B owner snapshot is stale; refresh it before verification"
    return None
