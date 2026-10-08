from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifex.render_node.socket_audit import audit_renderer_sockets
from artifex.windows_tasks import (
    _renderer_replacement_guard,
    _run_powershell,
    _status_script,
)

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows", reason="native Windows mock-process checks"
)


def _fake_settings(receipt_path: Path, port: int = 8188) -> SimpleNamespace:
    return SimpleNamespace(
        comfyui=SimpleNamespace(base_url=f"http://127.0.0.1:{port}"),
        render_agent=SimpleNamespace(
            comfyui_process=SimpleNamespace(ownership_receipt_path=receipt_path),
        ),
    )


def test_native_guard_refuses_live_test_owned_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Use real CIM PID inspection, but never create or stop a GPU process."""
    import artifex.config as config_module
    import artifex.render_node.process_identity as identity_module

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == b"ready"
        assert identity_module.windows_process_identity(child.pid) is not None
        settings = _fake_settings(tmp_path / "ownership.json")
        monkeypatch.setattr(config_module, "load_settings", lambda user_config: settings)
        monkeypatch.setattr(
            identity_module.ComfyReceiptStore, "load",
            lambda self: SimpleNamespace(identity=SimpleNamespace(pid=child.pid)),
        )
        with (
            pytest.raises(RuntimeError, match="still alive"),
            _renderer_replacement_guard(tmp_path / "render-node.yaml"),
        ):
            pytest.fail("A living test child must block scheduled-task replacement")
        assert child.poll() is None
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)
        if child.stdout is not None:
            child.stdout.close()


def test_native_guard_refuses_test_owned_tcp_listener(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Read real Windows TCP state of an ephemeral test-only loopback socket."""
    import artifex.config as config_module
    import artifex.render_node.process_identity as identity_module

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        settings = _fake_settings(tmp_path / "ownership.json", listener.getsockname()[1])
        monkeypatch.setattr(config_module, "load_settings", lambda user_config: settings)
        monkeypatch.setattr(identity_module.ComfyReceiptStore, "load", lambda self: None)
        report = audit_renderer_sockets(settings)
        assert report.status == "owner_unverified", report
        assert report.listener_pids
        with (
            pytest.raises(RuntimeError, match="not proven unoccupied"),
            _renderer_replacement_guard(tmp_path / "render-node.yaml"),
        ):
            pytest.fail("A listening mock upstream must prevent replacement")


def test_native_unknown_task_settings_remain_null_without_registration() -> None:
    """Shadow Get-ScheduledTask inside PowerShell; do not touch real tasks."""
    fake_task = r"""
$fakeTask = [pscustomobject]@{
  Description='Artifex managed autostart v1/renderer'
  State='Ready'
  Actions=@([pscustomobject]@{
    Execute='python.exe'
    Arguments='-m artifex.cli render-node serve'
    WorkingDirectory='C:\\Artifex'
  })
  Settings=[pscustomobject]@{
    AllowHardTerminate=$null
    MultipleInstances=$null
    ExecutionTimeLimit=$null
  }
}
function Get-ScheduledTask {
  param([string]$TaskName, [string]$TaskPath)
  return $fakeTask
}
"""
    status = json.loads(_run_powershell(fake_task + _status_script("renderer")))
    assert status["installed"] is True
    assert status["managed"] is True
    assert status["allow_hard_terminate"] is None
    assert status["multiple_instances"] is None
    assert status["execution_time_limit_seconds"] is None
