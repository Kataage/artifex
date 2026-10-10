"""GPU-free native Windows integration of real protected ComfyUI process lifecycle."""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from artifex.config.models import ArtifexSettings
from artifex.render_node.comfy_process import ManagedComfyUI
from artifex.render_node.process_identity import (
    ComfyReceiptStore,
    matches_owned_process,
    windows_process_identity,
)

# Disposable loopback fixture: no ComfyUI imports, assets or GPU requests.
# It advertises exactly the minimal /system_stats shape required for health.
_MOCK_SERVER = (
    "import json,os,pathlib,sys,time\n"
    "from http.server import BaseHTTPRequestHandler,HTTPServer\n"
    "port=int(sys.argv[4]);marker=pathlib.Path(sys.argv[5]);"
    "release=pathlib.Path(sys.argv[6])\n"
    "class Handler(BaseHTTPRequestHandler):\n"
    " def do_GET(self):\n"
    "  data=json.dumps({'system':{},'devices':[]}).encode()\n"
    "  self.send_response(200 if self.path=='/system_stats' else 404)\n"
    "  self.send_header('Content-Type','application/json')\n"
    "  self.send_header('Content-Length',str(len(data)))\n"
    "  self.end_headers();self.wfile.write(data)\n"
    " def log_message(self,*args):pass\n"
    "server=HTTPServer(('127.0.0.1',port),Handler);server.timeout=.2\n"
    "marker.write_text(str(os.getpid()),encoding='utf-8')\n"
    "deadline=time.monotonic()+65\n"
    "while not release.exists() and time.monotonic()<deadline:"
    " server.handle_request()\n"
    "server.server_close()\n"
)


class _TwoPolls:
    """Finite poll signal that never touches Task Scheduler."""

    def __init__(self) -> None:
        self.calls = 0

    def wait(self, _seconds: float) -> bool:
        self.calls += 1
        return self.calls > 2


@pytest.mark.skipif(os.name != "nt", reason="native Windows CIM/TCP required")
def test_native_managed_comfy_spawn_detach_and_reattach_without_gpu(
    tmp_path: Path,
) -> None:
    """Real ManagedComfyUI.start -> .close -> .start/reattach without a restart."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    marker = tmp_path / "ready.pid"
    release = tmp_path / "release.flag"
    working = tmp_path / "mock"
    working.mkdir()

    settings = ArtifexSettings()
    settings.render_agent.node_id = "gpu-b"
    settings.render_agent.gateway.enabled = True
    settings.comfyui.base_url = f"http://127.0.0.1:{port}"
    config = settings.render_agent.comfyui_process
    config.enabled = True
    config.executable = Path(sys.executable)
    config.working_directory = working
    config.ownership_receipt_path = tmp_path / "receipt.json"
    config.log_path = tmp_path / "mock-server.log"
    config.startup_timeout_seconds = 55
    config.arguments = (
        "-c", _MOCK_SERVER, "--listen", "127.0.0.1",
        "--port", str(port), str(marker), str(release),
    )
    first = ManagedComfyUI(settings)
    second = ManagedComfyUI(
        settings,
        process_factory=lambda *args, **kwargs: pytest.fail(
            "Exact live survivor must be reattached, not spawned a second time"
        ),
    )
    receipt_store = ComfyReceiptStore(config.ownership_receipt_path)
    actual_pid: int | None = None
    try:
        # These are the actual protected startup path, Windows CIM, TCP
        # inventory and exclusive-lease/receipt writes (not substituted mocks).
        first.start(threading.Event())
        assert marker.is_file()
        actual_pid = int(marker.read_text(encoding="utf-8"))
        receipt = receipt_store.load()
        assert receipt is not None
        assert first._current_receipt == receipt
        assert receipt.identity.pid == actual_pid
        assert matches_owned_process(settings, receipt, receipt.identity)
        assert windows_process_identity(actual_pid) == receipt.identity

        launcher = first.process
        assert launcher is not None
        if launcher.pid != actual_pid:
            assert receipt.schema_version == 2
            assert receipt.launcher_identity is not None
            assert receipt.launcher_identity.pid == launcher.pid
            assert receipt.identity.parent_pid == launcher.pid
        else:
            assert receipt.schema_version == 1

        # Closing only the supervisor must preserve the live test listener.
        first.close()
        assert windows_process_identity(actual_pid) is not None
        assert receipt_store.load() == receipt

        # Reattachment must prove the identical child, and never call Popen.
        second.start(threading.Event())
        assert second.process is None
        assert second._adopted == receipt
        second.watch(_TwoPolls())  # type: ignore[arg-type]
        assert windows_process_identity(actual_pid) is not None
        second.close()
        assert receipt_store.load() == receipt
    finally:
        # Only our disposable mock cooperatively exits. No terminate()/kill(),
        # and no command is ever directed at real ComfyUI or Task Scheduler.
        release.write_text("release", encoding="utf-8")
        second.close()
        first.close()
        if actual_pid is not None:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if windows_process_identity(actual_pid) is None:
                    break
                time.sleep(0.2)
