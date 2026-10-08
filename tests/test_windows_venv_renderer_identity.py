from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from artifex.config.models import ArtifexSettings
from artifex.render_node.comfy_process import ManagedComfyUI
from artifex.render_node.process_identity import (
    WindowsProcessIdentity,
    expected_receipt,
    matches_owned_process,
    verified_launcher_child,
)
from artifex.render_node.socket_audit import RendererSocketAudit


class FakeProcess:
    pid = 101
    dead = False
    terminate_calls = 0
    kill_calls = 0

    def poll(self) -> int | None:
        return 1 if self.dead else None

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


class FiniteStop:
    def __init__(self, limit: int = 1) -> None:
        self.limit = limit
        self.calls = 0

    def wait(self, _: float) -> bool:
        self.calls += 1
        return self.calls > self.limit


def _settings(tmp_path: Path) -> ArtifexSettings:
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"mock python")
    working = tmp_path / "ComfyUI"
    working.mkdir()
    cfg = ArtifexSettings()
    cfg.render_agent.node_id = "gpu-b"
    cfg.render_agent.gateway.enabled = True
    cfg.render_agent.comfyui_process.enabled = True
    cfg.render_agent.comfyui_process.executable = python
    cfg.render_agent.comfyui_process.working_directory = working
    cfg.render_agent.comfyui_process.arguments = (
        "main.py", "--listen", "127.0.0.1", "--port", "8188",
    )
    cfg.render_agent.comfyui_process.ownership_receipt_path = tmp_path / "receipt.json"
    cfg.render_agent.comfyui_process.log_path = tmp_path / "renderer.log"
    cfg.render_agent.comfyui_process.poll_seconds = 0.01
    cfg.comfyui.base_url = "http://127.0.0.1:8188"
    return cfg


def _identities(
    settings: ArtifexSettings,
) -> tuple[WindowsProcessIdentity, WindowsProcessIdentity]:
    cfg = settings.render_agent.comfyui_process
    assert cfg.executable is not None
    executable = str(cfg.executable.resolve())
    launcher = WindowsProcessIdentity(
        ProcessId=101, ParentProcessId=42,
        CreationDate="2026-10-09T01:00:00Z",
        ExecutablePath=executable,
        CommandLine=subprocess.list2cmdline([executable, *cfg.arguments]),
    )
    actual_exe = r"C:\Python312\python.exe"
    child = WindowsProcessIdentity(
        ProcessId=202, ParentProcessId=101,
        CreationDate="2026-10-09T01:00:01Z",
        ExecutablePath=actual_exe,
        CommandLine=subprocess.list2cmdline([actual_exe, *cfg.arguments]),
    )
    return launcher, child


def _socket(
    settings: ArtifexSettings,
    *, owned_pid: int | None = None, **kwargs: Any,
) -> RendererSocketAudit:
    if owned_pid is None:
        return RendererSocketAudit(
            status="owner_unverified", upstream_port=8188,
            expected_owned_pid=None, listener_pids=(202,),
            listener_addresses=("127.0.0.1",),
        )
    return RendererSocketAudit(
        status="owned_loopback_observed" if owned_pid == 202 else "foreign_listener",
        upstream_port=8188, expected_owned_pid=owned_pid,
        listener_pids=(202,), listener_addresses=("127.0.0.1",),
        owner_verified=owned_pid == 202,
    )


@pytest.mark.parametrize("drift", [
    "parent", "creation", "command", "launcher_command",
    "pid_reuse", "non_python", "bad_parent_clock",
])
def test_venv_shim_receipt_fails_closed_on_identity_drift(
    tmp_path: Path, drift: str,
) -> None:
    settings = _settings(tmp_path)
    launcher, child = _identities(settings)
    receipt = expected_receipt(settings, child, launcher_identity=launcher)
    assert receipt.schema_version == 2
    assert matches_owned_process(settings, receipt, child)
    if drift == "parent":
        candidate = child.model_copy(update={"parent_pid": 444})
    elif drift == "creation":
        candidate = child.model_copy(update={"started_utc": "2026-10-09T01:01:01Z"})
    elif drift == "command":
        candidate = child.model_copy(update={"command_line": "python.exe fake.py"})
    elif drift == "launcher_command":
        receipt = receipt.model_copy(update={
            "launcher_identity": launcher.model_copy(update={
                "command_line": "python.exe foreign.py",
            }),
        })
        candidate = child
    elif drift == "pid_reuse":
        candidate = child.model_copy(update={"pid": 303})
    elif drift == "non_python":
        candidate = child.model_copy(update={"executable": r"C:\Foreign\evil.exe"})
    else:
        candidate = child.model_copy(update={"started_utc": "2026-10-08T23:59:59Z"})
    assert not matches_owned_process(settings, receipt, candidate)


def test_venv_receipt_config_drift_fails_after_supervisor_crash(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    launcher, child = _identities(settings)
    receipt = expected_receipt(settings, child, launcher_identity=launcher)
    assert receipt.launcher_identity == launcher
    assert verified_launcher_child(settings, launcher, child)
    settings.render_agent.comfyui_process.arguments += ("--other",)
    assert not matches_owned_process(settings, receipt, child)


def test_non_python_launcher_cannot_claim_descendant(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    launcher, child = _identities(settings)
    assert settings.render_agent.comfyui_process.executable is not None
    tool = settings.render_agent.comfyui_process.executable.with_name("worker.exe")
    tool.write_bytes(b"mock")
    settings.render_agent.comfyui_process.executable = tool
    assert not verified_launcher_child(settings, launcher, child)


def test_protected_supervisor_records_child_pid_and_reattaches_after_launcher_loss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    launcher, child = _identities(settings)
    identities = {101: launcher, 202: child}
    import artifex.render_node.comfy_process as process_module

    state = {"launched": False}
    def audit(settings: ArtifexSettings, **kwargs: Any) -> RendererSocketAudit:
        if not state["launched"]:
            return RendererSocketAudit(
                status="missing_listener", upstream_port=8188, expected_owned_pid=None,
            )
        return _socket(settings, **kwargs)

    monkeypatch.setattr(process_module, "audit_renderer_sockets", audit)
    monkeypatch.setattr(
        process_module, "windows_process_identity",
        lambda pid: identities.get(pid),
    )
    mock = FakeProcess()
    def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
        state["launched"] = True
        return mock

    manager = ManagedComfyUI(
        settings, process_factory=spawn,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(manager, "_healthy", lambda: state["launched"])
    manager.start(threading.Event())
    receipt = manager._current_receipt
    assert receipt is not None and receipt.schema_version == 2
    assert receipt.identity.pid == 202
    assert receipt.launcher_identity == launcher

    # The venv wrapper exits first, while ComfyUI's real interpreter lives.
    mock.dead = True
    identities.pop(101)
    manager.watch(FiniteStop(limit=2))  # type: ignore[arg-type]
    assert mock.terminate_calls == mock.kill_calls == 0
    manager.close()

    def forbid_spawn(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Matching live child must be adopted, never respawned")

    adopted = ManagedComfyUI(settings, process_factory=forbid_spawn)
    monkeypatch.setattr(adopted, "_healthy", lambda: True)
    adopted.start(threading.Event())
    assert adopted.process is None
    assert adopted._adopted == receipt
    adopted.watch(FiniteStop())  # type: ignore[arg-type]
    adopted.close()
    assert mock.terminate_calls == mock.kill_calls == 0


def test_ambiguous_foreign_listener_is_never_claimed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    launcher, child = _identities(settings)
    identities = {101: launcher, 202: child.model_copy(update={"parent_pid": 777})}
    import artifex.render_node.comfy_process as process_module

    monkeypatch.setattr(
        process_module, "windows_process_identity",
        lambda pid: identities.get(pid),
    )
    mock = FakeProcess()
    active = {"value": False}
    def launch(*args: Any, **kwargs: Any) -> FakeProcess:
        active["value"] = True
        return mock

    monkeypatch.setattr(
        process_module, "audit_renderer_sockets",
        lambda s, **kwargs: _socket(s, **kwargs) if active["value"] else
        RendererSocketAudit(
            status="missing_listener", upstream_port=8188, expected_owned_pid=None,
        ),
    )
    manager = ManagedComfyUI(settings, process_factory=launch)  # type: ignore[arg-type]
    monkeypatch.setattr(manager, "_healthy", lambda: active["value"])
    with pytest.raises(RuntimeError, match="not a verified child"):
        manager.start(threading.Event())
    assert mock.terminate_calls == mock.kill_calls == 0



@pytest.mark.skipif(os.name != "nt", reason="real Windows CIM and native TCP ownership")
def test_native_venv_python_launcher_matches_actual_mock_tcp_listener(
    tmp_path: Path,
) -> None:
    """No GPU: check actual CIM parent, TCP owner and durable ownership receipt."""
    import artifex.render_node.comfy_process as process_module
    from artifex.render_node.process_identity import windows_process_identity
    from artifex.render_node.socket_audit import audit_renderer_sockets

    # Reserve a unique ephemeral port and release it before mock process starts.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    settings = _settings(tmp_path)
    settings.render_agent.comfyui_process.executable = Path(sys.executable)
    settings.render_agent.comfyui_process.working_directory = tmp_path
    settings.comfyui.base_url = f"http://127.0.0.1:{port}"
    marker = tmp_path / "listener-ready.txt"
    release = tmp_path / "listener-release.txt"
    code = (
        "import os,sys,socket,time,pathlib;"
        "s=socket.socket();s.bind(('127.0.0.1',int(sys.argv[1])));s.listen(1);"
        "pathlib.Path(sys.argv[2]).write_text(str(os.getpid()));"
        "deadline=time.monotonic()+25;"
        "release=pathlib.Path(sys.argv[3]);"
        "\nwhile not release.exists() and time.monotonic()<deadline: time.sleep(0.1)\n"
        "s.close()"
    )
    args = ("-c", code, str(port), str(marker), str(release))
    settings.render_agent.comfyui_process.arguments = args
    child = subprocess.Popen(
        [sys.executable, *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(tmp_path),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    manager = ManagedComfyUI(settings)
    try:
        deadline = time.monotonic() + 12
        while not marker.exists() and time.monotonic() < deadline:
            if child.poll() is not None:
                pytest.fail("Mock Python listener exited before ready")
            time.sleep(0.1)
        assert marker.exists(), "Mock listener never became ready"
        actual_pid = int(marker.read_text())
        report = audit_renderer_sockets(settings, owned_pid=actual_pid)
        assert report.status == "owned_loopback_observed", report
        manager.process = child
        manager._record_owned_process()
        manager._verify_owned_listener()
        receipt = manager._current_receipt
        assert receipt is not None
        assert receipt.identity.pid == actual_pid
        assert matches_owned_process(settings, receipt, receipt.identity)
        if actual_pid != child.pid:
            assert receipt.schema_version == 2
            assert receipt.launcher_identity is not None
            assert receipt.launcher_identity.pid == child.pid
            assert receipt.identity.parent_pid == child.pid
            assert receipt.identity.executable
        else:
            assert receipt.schema_version == 1
        # Closing the supervisor must not kill the owned test listener.
        manager.close()
        assert windows_process_identity(actual_pid) is not None
    finally:
        release.write_text("stop", encoding="utf-8")
        if child.poll() is None:
            child.wait(timeout=15)
        else:
            child.wait(timeout=5)
        manager.close()
