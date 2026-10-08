from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO


class RendererSupervisorLease:
    """Non-blocking, cross-process ownership of ONE managed renderer on PC-B.

    The lock is held for the complete supervisor lifecycle, including adopted
    child monitoring. It is automatically released by the OS if the owning
    supervisor crashes. Never use the presence of the lock *file* as evidence
    that the lock is held, and never remove the lock file to force takeover.
    """

    def __init__(self, receipt_path: Path) -> None:
        receipt = receipt_path.expanduser().absolute()
        self.path = receipt.with_name(receipt.name + ".supervisor.lock")
        self._handle: BinaryIO | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError("Renderer supervisor already holds its exclusive lease")
        if any(item.is_symlink() for item in (self.path, *self.path.parents)):
            raise ValueError("Refusing symlinked renderer supervisor lock path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError("Invalid renderer supervisor lock file")
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                # Windows LockFile locks the byte range even for an empty file.
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined] - Windows-only API
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ValueError) as exc:
            handle.close()
            raise RuntimeError(
                "Another Artifex renderer supervisor may be running or the "
                "exclusive lock is inaccessible; refusing a second manager"
            ) from exc
        self._handle = handle

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined] - Windows-only API
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    @property
    def held(self) -> bool:
        return self._handle is not None
