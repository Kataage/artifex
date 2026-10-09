from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from artifex.config.models import ArtifexSettings
from artifex.render_node.comfy_process import ManagedComfyUI
from artifex.render_node.process_identity import (
    ComfyReceiptStore,
    WindowsProcessIdentity,
    expected_receipt,
    matches_owned_process,
    windows_process_identity,
)
from artifex.render_node.socket_audit import RendererSocketAudit


class FakeChild:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.dead = False
        self.terminated = 0
        self.killed = 0

    def poll(self) -> int | None:
        return 1 if self.dead else None

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1


class CycleEvent:
    def __init__(self, count: int = 1) -> None:
        self.calls = 0
        self.count = count

    def is_set(self) -> bool:
        return False

    def wait(self, _seconds: float = 0.0) -> bool:
        self.calls += 1
        return self.calls > self.count


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    python = tmp_path / "bin" / "python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"test")
    root = tmp_path / "ComfyUI"
    root.mkdir()
    settings.render_agent.node_id = "gpu-b"
    settings.render_agent.comfyui_process.enabled = True
    settings.render_agent.comfyui_process.executable = python
    settings.render_agent.comfyui_process.working_directory = root
    settings.render_agent.comfyui_process.arguments = (
        "main.py", "--listen", "127.0.0.1", "--port", "8188",
    )
    settings.render_agent.comfyui_process.log_path = tmp_path / "comfy.log"
    settings.render_agent.comfyui_process.ownership_receipt_path = (
        tmp_path / "owner.json"
    )
    settings.render_agent.comfyui_process.poll_seconds = 0.01
    settings.render_agent.comfyui_process.restart_backoff_seconds = 0.0
    settings.render_agent.gateway.enabled = True
    settings.comfyui.base_url = "http://127.0.0.1:8188"
    return settings


def _identity(settings: ArtifexSettings, pid: int, *, start: str = "2026-10-09T01:00:00.0000000Z") -> WindowsProcessIdentity:
    cfg = settings.render_agent.comfyui_process
    assert cfg.executable is not None
    return WindowsProcessIdentity(
        ProcessId=pid, CreationDate=start,
        ExecutablePath=str(cfg.executable.resolve()),
        CommandLine=subprocess.list2cmdline(
            [str(cfg.executable.resolve()), *cfg.arguments]
        ),
    )


def _mock_socket(
    monkeypatch: pytest.MonkeyPatch, *,
    foreign_pid: int | None = None,
) -> None:
    def audit(_settings: ArtifexSettings, **kwargs: Any) -> RendererSocketAudit:
        owner = kwargs.get("owned_pid")
        if owner is None:
            return RendererSocketAudit(
                status="missing_listener", upstream_port=8188, expected_owned_pid=None,
            )
        pid = foreign_pid if foreign_pid is not None else owner
        return RendererSocketAudit(
            status="owned_loopback_observed" if pid == owner else "foreign_listener",
            upstream_port=8188, expected_owned_pid=owner,
            listener_pids=(pid,), listener_addresses=("127.0.0.1",),
            owner_verified=pid == owner,
        )

    monkeypatch.setattr(
        "artifex.render_node.comfy_process.audit_renderer_sockets", audit,
    )


def _start_original(
    settings: ArtifexSettings, monkeypatch: pytest.MonkeyPatch,
    identity_by_pid: dict[int, WindowsProcessIdentity],
) -> tuple[ManagedComfyUI, FakeChild]:
    original = FakeChild(101)
    _mock_socket(monkeypatch)
    monkeypatch.setattr(
        "artifex.render_node.comfy_process.windows_process_identity",
        lambda pid: identity_by_pid.get(pid),
    )
    manager = ManagedComfyUI(
        settings, process_factory=lambda *args, **kwargs: original,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(manager, "_healthy", lambda: manager.process is not None)
    manager.start(threading.Event())
    return manager, original


def test_protected_manager_reattaches_exact_survivor_and_never_kills_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    identities = {101: _identity(settings, 101)}
    manager, original = _start_original(settings, monkeypatch, identities)
    store = ComfyReceiptStore(settings.render_agent.comfyui_process.ownership_receipt_path)
    assert store.load() is not None
    manager.close()
    assert original.terminated == original.killed == 0

    def no_spawn(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("a matching live orphan must be reattached, never respawned")

    recovered = ManagedComfyUI(settings, process_factory=no_spawn)
    monkeypatch.setattr(recovered, "_healthy", lambda: True)
    recovered.start(threading.Event())
    assert recovered.process is None
    assert recovered._adopted == store.load()
    recovered.watch(CycleEvent())  # type: ignore[arg-type]
    recovered.close()
    assert original.terminated == original.killed == 0
    assert store.load() is not None


@pytest.mark.parametrize("reason", ["pid_reuse", "changed_config", "foreign_listener"])
def test_reattach_rejects_pid_reuse_launch_drift_and_foreign_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    settings = _settings(tmp_path)
    identities = {101: _identity(settings, 101)}
    first, original = _start_original(settings, monkeypatch, identities)
    first.close()
    if reason == "pid_reuse":
        identities[101] = _identity(
            settings, 101, start="2026-10-09T02:00:00.0000000Z",
        )
    elif reason == "changed_config":
        settings.render_agent.comfyui_process.arguments = (
            "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch",
        )
    else:
        _mock_socket(monkeypatch, foreign_pid=999)

    recovered = ManagedComfyUI(settings)
    monkeypatch.setattr(recovered, "_healthy", lambda: True)
    with pytest.raises(RuntimeError, match="(receipt|socket ownership)"):
        recovered.start(threading.Event())
    assert original.terminated == original.killed == 0


def test_protected_launch_rechecks_receipt_under_held_lease_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A receipt appearing after initial startup checks must not be overwritten."""
    settings = _settings(tmp_path)
    store = ComfyReceiptStore(
        settings.render_agent.comfyui_process.ownership_receipt_path,
    )
    foreign_receipt = expected_receipt(settings, _identity(settings, 404))
    calls: list[str] = []

    def no_spawn(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("A newly appeared receipt forbids all process launches")

    manager = ManagedComfyUI(settings, process_factory=no_spawn)
    monkeypatch.setattr(manager, "_healthy", lambda: False)

    def free_upstream() -> None:
        calls.append("checked")
        if len(calls) == 1:
            # Models a different writer leaving evidence between the
            # initial free-port check and the actual launch boundary.
            store.save(foreign_receipt)

    monkeypatch.setattr(manager, "_require_free_upstream", free_upstream)
    with pytest.raises(RuntimeError, match="receipt appeared before launch"):
        manager.start(threading.Event())
    assert calls == ["checked"]
    assert store.load() == foreign_receipt
    assert manager.process is None
    assert not manager._supervisor_lease.held


def test_protected_spawn_refuses_missing_supervisor_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)

    def no_spawn(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Unleased process launch must be impossible")

    manager = ManagedComfyUI(settings, process_factory=no_spawn)
    with pytest.raises(RuntimeError, match="exclusive supervisor lease"):
        manager._start_owned()
    assert manager.process is None

def test_stale_receipt_after_natural_death_allows_safe_new_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    identities = {101: _identity(settings, 101)}
    first, original = _start_original(settings, monkeypatch, identities)
    first.close()
    original.dead = True
    identities.pop(101)
    identities[202] = _identity(settings, 202, start="2026-10-09T02:00:00.0000000Z")
    next_child = FakeChild(202)
    recovered = ManagedComfyUI(
        settings,
        process_factory=lambda *args, **kwargs: next_child,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(recovered, "_healthy", lambda: recovered.process is not None)
    recovered.start(threading.Event())
    assert recovered.process is next_child
    receipt = ComfyReceiptStore(
        settings.render_agent.comfyui_process.ownership_receipt_path
    ).load()
    assert receipt is not None and receipt.identity.pid == 202
    recovered.close()
    assert next_child.terminated == 0


def test_adopted_child_natural_exit_can_restart_without_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    identities = {101: _identity(settings, 101)}
    first, original = _start_original(settings, monkeypatch, identities)
    first.close()
    new_child = FakeChild(202)
    identities[202] = _identity(settings, 202, start="2026-10-09T02:00:00.0000000Z")
    recovered = ManagedComfyUI(
        settings,
        process_factory=lambda *args, **kwargs: new_child,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        recovered, "_healthy",
        lambda: not original.dead if recovered.process is None
        else not recovered.process.dead,
    )
    recovered.start(threading.Event())
    assert recovered._adopted is not None
    original.dead = True
    identities.pop(101)
    recovered.watch(CycleEvent(count=2))  # type: ignore[arg-type]
    assert recovered.process is new_child
    receipt = ComfyReceiptStore(
        settings.render_agent.comfyui_process.ownership_receipt_path
    ).load()
    assert receipt is not None and receipt.identity.pid == 202
    recovered.close()
    assert original.terminated == original.killed == 0


def test_corrupted_or_symlinked_receipt_never_grants_adoption(
    tmp_path: Path,
) -> None:
    store = ComfyReceiptStore(tmp_path / "owner.json")
    store.path.write_text('{"unexpected":"schema"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        store.load()
    store.path.unlink()
    outside = tmp_path / "outside"
    outside.write_text("{}", encoding="utf-8")
    try:
        store.path.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlinked"):
        store.load()


def test_receipt_is_config_specific_and_not_permission_to_terminate(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    process = _identity(settings, 101)
    receipt = expected_receipt(settings, process)
    assert matches_owned_process(settings, receipt, process)
    assert receipt.identity.started_utc
    settings.render_agent.node_id = "unrelated-node"
    assert not matches_owned_process(settings, receipt, process)
    settings.render_agent.node_id = "gpu-b"
    modified = process.model_copy(update={"command_line": "python.exe something.py"})
    assert not matches_owned_process(settings, receipt, modified)


@pytest.mark.skipif(os.name != "nt", reason="Windows CIM process identity required")
def test_native_windows_cim_identity_is_stable_and_present() -> None:
    first = windows_process_identity(os.getpid())
    second = windows_process_identity(os.getpid())
    assert first is not None and second is not None
    assert first == second
    assert first.pid == os.getpid()
    assert first.command_line and first.executable
    assert first.started_utc
