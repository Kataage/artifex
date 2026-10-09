"""Real OS thread/process mutual exclusion for qualification session writers."""
from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from artifex.qualification.session_lock import qualification_session_lock


def test_native_session_lock_serializes_threads_and_preserves_file(tmp_path: Path) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    arrived = threading.Event()
    acquired = threading.Event()

    def writer() -> None:
        arrived.set()
        with qualification_session_lock(directory, timeout_seconds=3):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with qualification_session_lock(directory):
            future = pool.submit(writer)
            assert arrived.wait(3)
            assert not acquired.wait(0.15)
        future.result(timeout=4)
    assert acquired.is_set()
    assert (directory / "qualification.lock").exists()
    with qualification_session_lock(directory, timeout_seconds=0):
        pass


def test_native_session_lock_blocks_another_real_python_process(tmp_path: Path) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    script = """
import sys
from pathlib import Path
from artifex.qualification.session_lock import qualification_session_lock
try:
    with qualification_session_lock(Path(sys.argv[1]), timeout_seconds=0.2):
        raise SystemExit(10)  # A second process must not acquire this lock.
except TimeoutError:
    raise SystemExit(0)
"""
    with qualification_session_lock(directory):
        child = subprocess.run(
            [sys.executable, "-c", script, str(directory)],
            capture_output=True, text=True, timeout=8, check=False,
        )
        assert child.returncode == 0, child.stderr
    with qualification_session_lock(directory, timeout_seconds=0.1):
        pass


def test_native_session_lock_timeout_and_missing_directory_fail_closed(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "session"
    with (
        pytest.raises(ValueError, match="missing"),
        qualification_session_lock(directory),
    ):
        pytest.fail("nonexistent session must not be locked")
    directory.mkdir()
    with (
        pytest.raises(ValueError, match="nonnegative"),
        qualification_session_lock(directory, timeout_seconds=-1),
    ):
        pytest.fail("invalid timeout")
    # A stable lock file is not deleted and cannot be confused with data.
    with qualification_session_lock(directory):
        assert (directory / "qualification.lock").is_file()
