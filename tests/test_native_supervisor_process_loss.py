"""Native Windows cross-process supervisor-loss recovery; no real GPU or ComfyUI.

A disposable mock HTTP listener survives an *isolated* supervisor subprocess
that exits without calling close(). A SECOND Python process must reacquire
the Windows lease and attach the exact existing listener with no second spawn.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from artifex.config.models import ArtifexSettings
from artifex.render_node.process_identity import (
    ComfyReceiptStore,
    windows_process_identity,
)

# Every operation stays in the test runner's unique tmp_path and an ephemeral
# local 127.0.0.1 port. The fixture exits cooperatively on its release flag.
_SERVER = """import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

port = int(sys.argv[4])
marker, release = Path(sys.argv[5]), Path(sys.argv[6])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps({"system": {}, "devices": []}).encode()
        self.send_response(200 if self.path == "/system_stats" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass

server = HTTPServer(("127.0.0.1", port), Handler)
server.timeout = 0.2
marker.write_text(str(os.getpid()), encoding="utf-8")
deadline = time.monotonic() + 160
while not release.exists() and time.monotonic() < deadline:
    server.handle_request()
server.server_close()
"""

# Not imported in the parent. Both supervisor instances execute this script
# via real native Windows processes. In "loss" mode os._exit simulates the
# supervisor exiting without cleanup, so the OS must release its file lease.
_SUPERVISOR = """import os
import subprocess
import sys
import threading
from pathlib import Path
from artifex.config.models import ArtifexSettings
from artifex.render_node.comfy_process import ManagedComfyUI

settings = ArtifexSettings.model_validate_json(
    Path(sys.argv[1]).read_text(encoding="utf-8")
)
mode = sys.argv[2]

def forbid_spawn(*args, **kwargs):
    raise RuntimeError("second supervisor tried to spawn another renderer")

manager = ManagedComfyUI(
    settings,
    process_factory=forbid_spawn if mode == "reattach" else subprocess.Popen,
)
manager.start(threading.Event())
if mode == "loss":
    if manager._current_receipt is None or not manager._supervisor_lease.held:
        raise RuntimeError("first supervisor did not establish ownership")
    # Deliberately do not call close(), to exercise the OS releasing a lease
    # after ONLY this disposable supervisor's unexpected process exit.
    os._exit(0)
if mode != "reattach" or manager.process is not None or manager._adopted is None:
    raise RuntimeError("second supervisor failed to reattach exact live child")
manager._verify_owned_listener()
manager.close()
"""


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows CIM and lease")
def test_os_releases_abrupt_supervisor_lease_and_child_reattaches(
    tmp_path: Path,
) -> None:
    """Run TWO actual supervisors and one independent fixture child on Windows."""
    port = _port()
    root = tmp_path / "disposable"
    root.mkdir()
    marker, release = root / "ready.pid", root / "release.flag"
    server = root / "mock_server.py"
    supervisor = root / "mock_supervisor.py"
    config_path = root / "settings.json"
    server.write_text(_SERVER, encoding="utf-8")
    supervisor.write_text(_SUPERVISOR, encoding="utf-8")

    settings = ArtifexSettings()
    settings.render_agent.node_id = "fixture-gpu-b"
    settings.render_agent.gateway.enabled = True
    settings.comfyui.base_url = f"http://127.0.0.1:{port}"
    cfg = settings.render_agent.comfyui_process
    cfg.enabled = True
    cfg.executable = Path(sys.executable)
    cfg.working_directory = root
    cfg.arguments = (
        str(server), "--listen", "127.0.0.1", "--port", str(port),
        str(marker), str(release),
    )
    cfg.ownership_receipt_path = root / "owner-receipt.json"
    cfg.log_path = root / "mock-server.log"
    cfg.startup_timeout_seconds = 70
    config_path.write_text(settings.model_dump_json(), encoding="utf-8")
    store = ComfyReceiptStore(cfg.ownership_receipt_path)
    original_pid: int | None = None

    def run_supervisor(mode: str) -> None:
        # Never send a signal to any real Artifex/ComfyUI process; this only
        # launches a newly written, isolated temporary test supervisor.
        result = subprocess.run(
            [sys.executable, str(supervisor), str(config_path), mode],
            cwd=root, capture_output=True, text=True, encoding="utf-8",
            timeout=85, check=False, shell=False,
        )
        assert result.returncode == 0, (
            f"Disposable {mode} supervisor exited {result.returncode}: "
            f"{result.stderr[-1500:]}"
        )

    try:
        run_supervisor("loss")
        assert marker.exists(), "real spawned fixture listener did not bind"
        original_pid = int(marker.read_text(encoding="utf-8"))
        original_receipt = store.load()
        assert original_receipt is not None
        assert original_receipt.identity.pid == original_pid
        # A real independent process must still own the original listener.
        assert windows_process_identity(original_pid) == original_receipt.identity
        # No explicit manager.close: the first OS process is now gone.
        run_supervisor("reattach")
        assert store.load() == original_receipt
        assert windows_process_identity(original_pid) == original_receipt.identity
    finally:
        # Only this mock listener observes the private flag and closes itself.
        # There is NO terminate()/kill() of a renderer or a supervisor PID.
        release.write_text("release", encoding="utf-8")
        if original_pid is not None:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if windows_process_identity(original_pid) is None:
                    break
                time.sleep(0.2)
