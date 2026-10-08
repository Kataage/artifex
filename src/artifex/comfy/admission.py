from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from artifex.comfy.errors import ComfyErrorKind, ComfyUIError


class SubmissionFencedError(ComfyUIError):
    """Explicit non-retryable refusal before an HTTP POST to ComfyUI."""

    def __init__(self) -> None:
        super().__init__(
            ComfyErrorKind.CANCELLED,
            "Artifex ComfyUI submission is fenced for controlled maintenance",
            retryable=False,
        )


class ComfySubmissionFence:
    """Durable cross-process SQLite admission barrier for Artifex clients.

    Each /prompt submission holds an independent SQLite IMMEDIATE transaction
    until the POST and its retries have completed. A maintenance seal must get
    that same transaction before changing the durable state; no Artifex /prompt
    can slip between the seal's final validation and its commit.

    IMPORTANT: This protects only ComfyUIClient instances using the SAME
    admission database path. It does not prevent a direct external /prompt,
    cannot authorize killing a renderer and must never be called a network
    firewall or exclusive ComfyUI admission gateway.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().absolute()

    def _check_path(self) -> None:
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError("Refusing symlinked submission-fence database path")
        if self.path.exists() and not self.path.is_file():
            raise ValueError("Submission-fence path is not a regular file")

    def _begin(self) -> sqlite3.Connection:
        self._check_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=20,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS submission_fence "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), blocked INTEGER NOT NULL "
                "CHECK (blocked IN (0, 1)))"
            )
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _blocked(connection: sqlite3.Connection) -> bool:
        row = connection.execute(
            "SELECT blocked FROM submission_fence WHERE id = 1"
        ).fetchone()
        return bool(row[0]) if row is not None else False

    @staticmethod
    def _finish(connection: sqlite3.Connection, *, commit: bool) -> None:
        try:
            if commit:
                connection.commit()
            else:
                connection.rollback()
        finally:
            connection.close()

    async def _acquire(self) -> sqlite3.Connection:
        # SQLite's transaction wait must not block the asyncio event loop;
        # a competing in-flight POST is allowed to finish before sealing.
        task = asyncio.create_task(asyncio.to_thread(self._begin))
        try:
            return await asyncio.shield(task)
        except BaseException:
            # Cancellation while waiting for BEGIN IMMEDIATE cannot orphan
            # an already-acquired SQLite transaction.
            try:
                connection = await task
            except BaseException:
                pass
            else:
                await asyncio.to_thread(self._finish, connection, commit=False)
            raise

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        connection = await self._acquire()
        try:
            if self._blocked(connection):
                raise SubmissionFencedError()
            yield
        finally:
            # Admission has no writes. Roll back on errors, cancellation and
            # success to release the lock before any follow-on read requests.
            await asyncio.to_thread(self._finish, connection, commit=False)

    async def seal(self, validate: Callable[[], Awaitable[bool]]) -> bool:
        """Atomically re-check drain after all earlier gated POSTs have exited.

        Returns false when validation refuses sealing. The validator is called
        *inside* the barrier, thus other compliant submitters must wait.
        """
        connection = await self._acquire()
        commit = False
        try:
            if self._blocked(connection):
                return True
            if not await validate():
                return False
            connection.execute(
                "INSERT INTO submission_fence(id, blocked) VALUES (1, 1) "
                "ON CONFLICT(id) DO UPDATE SET blocked = 1"
            )
            commit = True
            return True
        finally:
            await asyncio.to_thread(self._finish, connection, commit=commit)

    async def release(self) -> None:
        """Explicit maintenance completion, never an automatic daemon resume."""
        connection = await self._acquire()
        try:
            connection.execute(
                "INSERT INTO submission_fence(id, blocked) VALUES (1, 0) "
                "ON CONFLICT(id) DO UPDATE SET blocked = 0"
            )
            await asyncio.to_thread(self._finish, connection, commit=True)
        except BaseException:
            # A completed commit already closes; avoid masking its errors.
            # Closing an already-closed connection is safe in sqlite3.
            await asyncio.to_thread(connection.close)
            raise

    def status(self) -> bool:
        """Read-only status; absent file means never sealed."""
        self._check_path()
        if not self.path.exists():
            return False
        connection = sqlite3.connect(
            f"file:{self.path.as_posix()}?mode=ro",
            uri=True,
            timeout=10,
        )
        try:
            return self._blocked(connection)
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return False
            raise
        finally:
            connection.close()
