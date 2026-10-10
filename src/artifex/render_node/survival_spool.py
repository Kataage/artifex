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
    # The filename contains only second-resolution UTC and a random UUID.
    # Lexical ordering can select an older PASS over a new blocked report
    # written in the same second. Compare actual write times instead.
    # On equal timestamps, fail closed: UUID order is not a time order.
    timed = [(path.lstat().st_mtime_ns, path) for path in entries]
    most_recent = max(stamp for stamp, _ in timed)
    latest = [path for stamp, path in timed if stamp == most_recent]
    if len(latest) != 1:
        raise ValueError("Survival evidence newest write time is ambiguous")
    newest = latest[0]
    if newest.is_symlink() or not newest.is_file():
        raise ValueError("Newest survival evidence file is unsafe")
    before = newest.stat()
    if not 0 < before.st_size <= _MAX_BYTES:
        raise ValueError("Newest survival evidence size invalid")
    raw = newest.read_bytes()
    # A concurrent observer may still be writing an exclusively created
    # file. Never return partial/inconsistent contents or an older PASS.
    after = newest.stat()
    if (
        newest.is_symlink()
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_size != after.st_size
        or len(raw) != after.st_size
        or not 0 < len(raw) <= _MAX_BYTES
    ):
        raise ValueError("Newest survival evidence changed during read")
    # Never expose arbitrary paths/command strings from the source directory.
    return hashlib.sha256(raw).hexdigest(), raw.decode("utf-8")
