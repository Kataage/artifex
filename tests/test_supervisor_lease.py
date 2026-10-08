from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from artifex.config.models import ArtifexSettings
from artifex.render_node.comfy_process import ManagedComfyUI
from artifex.render_node.supervisor_lease import RendererSupervisorLease


def _lease_path(tmp_path: Path) -> Path:
    return tmp_path / "render" / "owner.json"


def _subprocess_attempt(path: Path) -> subprocess.CompletedProcess[str]:
    """Try acquiring the exact same lease in a fresh Python interpreter."""
    script = (
        "import sys; "
        "from pathlib import Path; "
        "from artifex.render_node.supervisor_lease import RendererSupervisorLease; "
        "lease = RendererSupervisorLease(Path(sys.argv[1])); "
        "\ntry:\n"
        " lease.acquire()\n"
        "except RuntimeError:\n"
        " print('denied')\n"
        "else:\n"
        " print('acquired')\n"
        " lease.release()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True, text=True, timeout=20, check=False,
    )


def test_cross_process_exclusive_lease_denies_second_interpreter(
    tmp_path: Path,
) -> None:
    target = _lease_path(tmp_path)
    first = RendererSupervisorLease(target)
    first.acquire()
    try:
        assert first.held
        with pytest.raises(RuntimeError, match="already holds"):
            first.acquire()
        denied = _subprocess_attempt(target)
        assert denied.returncode == 0, denied.stderr
        assert denied.stdout.strip() == "denied"
    finally:
        first.release()

    assert not first.held
    assert first.path.is_file(), "the lock file need not be deleted on release"
    acquired = _subprocess_attempt(target)
    assert acquired.returncode == 0, acquired.stderr
    assert acquired.stdout.strip() == "acquired"


def test_supervisor_lease_is_released_on_abnormal_os_process_exit(
    tmp_path: Path,
) -> None:
    target = _lease_path(tmp_path)
    script = (
        "import os,sys; "
        "from pathlib import Path; "
        "from artifex.render_node.supervisor_lease import RendererSupervisorLease; "
        "lease=RendererSupervisorLease(Path(sys.argv[1])); "
        "lease.acquire(); os._exit(0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
    lease = RendererSupervisorLease(target)
    lease.acquire()
    lease.release()


def test_supervisor_lease_refuses_symlinked_parent_or_file(tmp_path: Path) -> None:
    link = tmp_path / "alias"
    folder = tmp_path / "real"
    folder.mkdir()
    try:
        link.symlink_to(folder, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    with pytest.raises(ValueError, match="symlink"):
        RendererSupervisorLease(link / "owner.json").acquire()


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_agent.comfyui_process.enabled = True
    settings.render_agent.gateway.enabled = True
    settings.render_agent.comfyui_process.ownership_receipt_path = _lease_path(tmp_path)
    return settings


def test_two_managed_renderer_instances_cannot_claim_same_protected_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    first = ManagedComfyUI(settings)
    second = ManagedComfyUI(settings)
    observed: list[str] = []
    monkeypatch.setattr(
        first, "_start_with_lease", lambda _stop: observed.append("first"),
    )
    monkeypatch.setattr(
        second, "_start_with_lease", lambda _stop: observed.append("second"),
    )
    first.start(threading.Event())
    try:
        assert observed == ["first"]
        with pytest.raises(RuntimeError, match="second manager"):
            second.start(threading.Event())
        assert observed == ["first"]
    finally:
        first.close()

    second.start(threading.Event())
    assert observed == ["first", "second"]
    second.close()


def test_failed_protected_start_releases_exclusive_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    manager = ManagedComfyUI(settings)

    def refuse_start(_stop: threading.Event) -> None:
        raise RuntimeError("invalid ComfyUI configuration")

    monkeypatch.setattr(manager, "_start_with_lease", refuse_start)
    with pytest.raises(RuntimeError, match="invalid ComfyUI configuration"):
        manager.start(threading.Event())
    assert not manager._supervisor_lease.held

    replacement = RendererSupervisorLease(
        settings.render_agent.comfyui_process.ownership_receipt_path
    )
    replacement.acquire()
    replacement.release()


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific lock behavior")
def test_native_windows_lock_survives_a_held_supervisor_process(
    tmp_path: Path,
) -> None:
    target = _lease_path(tmp_path)
    first = RendererSupervisorLease(target)
    first.acquire()
    try:
        assert _subprocess_attempt(target).stdout.strip() == "denied"
    finally:
        first.release()
    assert _subprocess_attempt(target).stdout.strip() == "acquired"
