"""Durable, bounded and read-only survival trace handoff on PC-B.

Only the local PC-B observer writes reports. The authenticated GET serves the
newest *named* immutable report; it cannot execute probes or accept a path.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from artifex.config.models import ArtifexSettings

_MAX_BYTES = 12 * 1024 * 1024
_MAX_ENTRIES = 512
_TRACE_NAME = re.compile(r"^survival-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}\.json$")


def automatic_survival_output(settings: ArtifexSettings) -> Path:
    """Generate a unique output under the PC-B-configured evidence directory."""
    root = settings.render_agent.survival_evidence_dir.expanduser().absolute()
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise ValueError("Survival spool directory is symlinked")
    return root / (
        "survival-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex + ".json"
    )


def latest_survival_trace(settings: ArtifexSettings) -> tuple[str, str]:
    """Return the latest report's exact SHA and UTF-8 JSON contents.

    The newest matching file is authoritative even if malformed/unsafe.
    Never fall back to an older apparently successful observation.
    """
    root = settings.render_agent.survival_evidence_dir.expanduser().absolute()
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise ValueError("Survival spool path is symlinked")
    if not root.is_dir():
        raise FileNotFoundError("Survival spool unavailable")
    entries: list[Path] = []
    for item in root.iterdir():
        if _TRACE_NAME.fullmatch(item.name):
            entries.append(item)
            if len(entries) > _MAX_ENTRIES:
                raise ValueError("Survival evidence spool inventory exceeds bound")
    if not entries:
        raise FileNotFoundError("No saved native survival evidence")
    newest = max(entries, key=lambda path: path.name)
    if newest.is_symlink() or not newest.is_file():
        raise ValueError("Newest survival evidence file is unsafe")
    if not 0 < newest.stat().st_size <= _MAX_BYTES:
        raise ValueError("Newest survival evidence size invalid")
    raw = newest.read_bytes()
    if not 0 < len(raw) <= _MAX_BYTES:
        raise ValueError("Newest survival evidence exceeds hard limit")
    # Never expose arbitrary paths/command strings from the source directory.
    return hashlib.sha256(raw).hexdigest(), raw.decode("utf-8")
