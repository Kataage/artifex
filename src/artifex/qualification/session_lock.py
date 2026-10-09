"""Cross-process bounded qualification session file lock.

Uses only Python's Windows / POSIX standard libraries and protects one byte of
a persistent per-session lock file. Never delete a lock file while processes
may hold it: on Windows a deleted/recreated path could break mutual exclusion.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def qualification_session_lock(
    session_dir: Path,
    *,
    timeout_seconds: float = 60.0,
) -> Iterator[None]:
    """Serialize competing writers to one session across threads/processes.

    A bounded lock failure leaves the evidence unchanged, not silently PASS.
    """
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be nonnegative")
    if not session_dir.is_dir() or any(
        path.is_symlink() for path in (session_dir, *session_dir.parents)
    ):
        raise ValueError("Unsafe or missing qualification session directory")
    path = session_dir / "qualification.lock"
    if path.is_symlink():
        raise ValueError("Refusing symlinked qualification session lock")
    # a+b creates a persistent lock file if absent without ever truncating it.
    # The same byte is locked by every writer (not a PID-only advisory flag).
    with path.open("a+b") as handle:
        deadline = time.monotonic() + timeout_seconds
        if os.name == "nt":
            import msvcrt

            def acquire() -> None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release() -> None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire() -> None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        while True:
            try:
                acquire()
                break
            except (OSError, BlockingIOError) as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Qualification session is busy; no evidence was modified"
                    ) from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            release()
