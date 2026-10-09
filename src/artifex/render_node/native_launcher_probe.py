"""Disposable native Windows virtualenv launcher-to-TCP-listener evidence probe.

Only the current local Python (or an explicitly chosen Python executable) is
invoked with a tiny loopback-only fixture. The real renderer is NEVER opened,
adopted, restarted, terminated, or used for any GPU job. A matching socket PID
is insufficient: the original launch CIM identity and exact argv/ancestry must
also match, with repeat checks before reporting observational success.
"""
from __future__ import annotations

import json
import ntpath
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings
from artifex.render_node.process_identity import (
    WindowsProcessIdentity,
    expected_receipt,
    matches_owned_process,
    verified_launcher_child,
    windows_process_identity,
)
from artifex.render_node.socket_audit import audit_renderer_sockets

# The mock only listens on an ephemeral local port. It never imports ComfyUI,
# opens a project folder, or accepts a network client.
_FIXTURE = (
    "import json,os,pathlib,socket,sys,time\n"
    "marker=pathlib.Path(sys.argv[1]); release=pathlib.Path(sys.argv[2])\n"
    "s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
    "s.bind(('127.0.0.1',0)); s.listen(1)\n"
    "marker.write_text(json.dumps({'pid':os.getpid(),'port':s.getsockname()[1],"
    "'nonce':sys.argv[3]}),encoding='utf-8')\n"
    "deadline=time.monotonic()+25\n"
    "while not release.exists() and time.monotonic()<deadline: time.sleep(.05)\n"
    "s.close()\n"
)


class LauncherEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    observed_at_utc: datetime
    host: str
    status: Literal["observed", "blocked", "unsupported"]
    reason: str
    launcher_pid: int | None = None
    listener_pid: int | None = None
    ephemeral_port: int | None = Field(default=None, ge=1, le=65535)
    relation: Literal["direct", "venv_child", "unverified"] = "unverified"
    launcher_identity: WindowsProcessIdentity | None = None
    listener_identity: WindowsProcessIdentity | None = None
    original_launcher_verified: bool = False
    listener_identity_stable: bool = False
    tcp_owner_stable: bool = False
    verified_runtime_provenance: bool = False
    test_listener_only: Literal[True] = True
    actual_comfyui_inspected: Literal[False] = False
    production_child_survival_qualified: Literal[False] = False
    real_machine_gpu_qualified: Literal[False] = False
    renderer_restart_authorized: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False


def _chosen_python(python: Path | None) -> Path:
    chosen = (python or Path(sys.executable)).expanduser().absolute()
    if (
        not chosen.is_file()
        or any(parent.is_symlink() for parent in (chosen, *chosen.parents))
        or ntpath.basename(str(chosen)).casefold() != "python.exe"
    ):
        raise ValueError("A native, existing, non-symlinked python.exe is required")
    return chosen


def inspect_native_python_listener(
    *,
    python: Path | None = None,
    timeout_seconds: float = 16.0,
) -> LauncherEvidence:
    """Observe only a temporary fixture. No production ownership is assumed."""
    import socket

    if os.name != "nt":
        return LauncherEvidence(
            observed_at_utc=datetime.now(UTC),
            host=socket.gethostname(), status="unsupported",
            reason="native_windows_required",
        )
    if not 5 <= timeout_seconds <= 30:
        raise ValueError("Fixture timeout must be between 5 and 30 seconds")
    executable = _chosen_python(python)
    report: dict[str, object] = {
        "observed_at_utc": datetime.now(UTC),
        "host": socket.gethostname(),
        "status": "blocked",
        "reason": "not_observed",
    }
    # Temporary, uniquely named marker/release files are never in a ComfyUI
    # directory. No configured ComfyUI executable, PID or port is touched.
    with tempfile.TemporaryDirectory(prefix="artifex-identity-") as temporary:
        root = Path(temporary)
        marker, release = root / "ready.json", root / "release.flag"
        nonce = root.name
        args = ("-c", _FIXTURE, str(marker), str(release), nonce)
        config = ArtifexSettings()
        config.render_agent.comfyui_process.executable = executable
        config.render_agent.comfyui_process.working_directory = root
        config.render_agent.comfyui_process.arguments = args
        # Never pass production API tokens to the disposable child process.
        env = {
            k: v for k, v in os.environ.items()
            if not (k.upper().startswith("ARTIFEX_") and any(
                part in k.upper() for part in ("TOKEN", "SECRET", "API_KEY")
            ))
        }
        child = subprocess.Popen(
            [str(executable), *args],
            cwd=str(root), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            shell=False, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        report["launcher_pid"] = child.pid
        try:
            # Capture the launching Python identity before a venv shim may
            # exit. Missing original evidence must never be backfilled by
            # guessing from a matching TCP listener.
            launcher = windows_process_identity(child.pid)
            report["launcher_identity"] = launcher
            if launcher is None:
                report["reason"] = "original_launcher_identity_unavailable"
                return LauncherEvidence.model_validate(report)
            report["original_launcher_verified"] = True
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline and not marker.is_file():
                if child.poll() is not None and time.monotonic() > deadline - 2:
                    break
                time.sleep(0.05)
            if not marker.is_file() or marker.stat().st_size > 1024:
                report["reason"] = "fixture_listener_not_ready"
                return LauncherEvidence.model_validate(report)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("nonce") != nonce
                or not isinstance(payload.get("pid"), int)
                or not isinstance(payload.get("port"), int)
                or not 0 < payload["pid"] < 2**31
                or not 0 < payload["port"] <= 65535
            ):
                report["reason"] = "fixture_nonce_or_pid_invalid"
                return LauncherEvidence.model_validate(report)
            listener_pid = payload["pid"]
            report["listener_pid"] = listener_pid
            report["ephemeral_port"] = payload["port"]
            config.comfyui.base_url = f"http://127.0.0.1:{payload['port']}"
            listener = windows_process_identity(listener_pid)
            report["listener_identity"] = listener
            if listener is None:
                report["reason"] = "fixture_listener_identity_missing"
                return LauncherEvidence.model_validate(report)
            if listener_pid == launcher.pid:
                relation: Literal["direct", "venv_child", "unverified"] = "direct"
                candidate = expected_receipt(config, listener)
            elif verified_launcher_child(config, launcher, listener):
                relation = "venv_child"
                candidate = expected_receipt(
                    config, listener, launcher_identity=launcher,
                )
            else:
                report["reason"] = "launcher_ancestry_or_arguments_unverified"
                return LauncherEvidence.model_validate(report)
            report["relation"] = relation
            if not matches_owned_process(config, candidate, listener):
                report["reason"] = "fixture_receipt_identity_mismatch"
                return LauncherEvidence.model_validate(report)
            first = audit_renderer_sockets(config, owned_pid=listener_pid)
            second = audit_renderer_sockets(config, owned_pid=listener_pid)
            stable_socket = (
                first.status == second.status == "owned_loopback_observed"
                and first.listener_pids == second.listener_pids == (listener_pid,)
                and first.listener_addresses == second.listener_addresses
            )
            report["tcp_owner_stable"] = stable_socket
            again = windows_process_identity(listener_pid)
            current_launcher = windows_process_identity(launcher.pid)
            stable = again == listener and (
                current_launcher is None or current_launcher == launcher
            )
            report["listener_identity_stable"] = stable
            if not stable or not stable_socket:
                report["reason"] = "fixture_identity_or_socket_changed"
                return LauncherEvidence.model_validate(report)
            report["verified_runtime_provenance"] = True
            report["status"] = "observed"
            report["reason"] = "isolated_mock_python_listener_verified"
            return LauncherEvidence.model_validate(report)
        except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError):
            report["reason"] = "native_probe_failed"
            return LauncherEvidence.model_validate(report)
        finally:
            # Only instruct OUR disposable listener to exit via its private
            # flag. Never kill/terminate a process by PID, including this
            # process: if it cannot exit, its own 25-second deadline applies.
            release.write_text("release", encoding="utf-8")
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
