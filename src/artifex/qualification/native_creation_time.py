"""Timezone-invariant identity comparison for Windows native process creation.

A PID alone is not ownership. Compare timezone-aware creation instants as
well: Windows CIM may emit the same instant using UTC or a local offset.
Malformed or naive timestamps must never be treated as matching identities.
"""
from __future__ import annotations

from datetime import UTC, datetime


def native_creation_instant(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def same_native_creation_instant(left: str | None, right: str | None) -> bool:
    earlier = native_creation_instant(left)
    later = native_creation_instant(right)
    return earlier is not None and later is not None and earlier == later
