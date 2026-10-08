from __future__ import annotations

import json
import ntpath
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from artifex.config.models import ArtifexSettings

_MAX_RECEIPT_BYTES = 8192
# PID is an integer passed by the caller, never arbitrary script text.
_PROCESS_SCRIPT = (
    "$ErrorActionPreference='Stop'; "
    "$entry=Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}' "
    "-ErrorAction Stop; "
    "if ($null -eq $entry) {{ 'null' }} else {{ "
    "$identity=[pscustomobject]@{{"
    "ProcessId=[int]$entry.ProcessId; "
    "CreationDate=$entry.CreationDate.ToUniversalTime().ToString('o'); "
    "ExecutablePath=[string]$entry.ExecutablePath; "
    "ParentProcessId=[int]$entry.ParentProcessId; "
    "CommandLine=[string]$entry.CommandLine"
    "}}; ConvertTo-Json -InputObject $identity -Compress "
    "}}"
)


class WindowsProcessIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    pid: int = Field(alias="ProcessId", gt=0)
    started_utc: str = Field(alias="CreationDate", min_length=15)
    executable: str = Field(alias="ExecutablePath", min_length=1)
    command_line: str = Field(alias="CommandLine", min_length=1)
    parent_pid: int | None = Field(default=None, alias="ParentProcessId", ge=0)


class ComfyOwnershipReceipt(BaseModel):
    """Durable historical evidence, NEVER authority to terminate a process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    host: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    executable: str = Field(min_length=1)
    working_directory: str = Field(min_length=1)
    expected_command_line: str = Field(min_length=1)
    identity: WindowsProcessIdentity
    # Schema 2 only: immutable proof that a verified Windows venv launcher
    # directly spawned the process actually listening on the ComfyUI port.
    launcher_identity: WindowsProcessIdentity | None = None


def windows_process_identity(pid: int) -> WindowsProcessIdentity | None:
    """Read PID + creation time + full image path + command line from CIM."""
    if os.name != "nt":
        raise OSError("process identity inspection requires Windows")
    if pid <= 0 or pid >= 2**31:
        raise ValueError("invalid PID for Windows process inspection")
    script = _PROCESS_SCRIPT.format(pid=pid)
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", timeout=15,
        check=False, shell=False,
    )
    if completed.returncode:
        raise OSError(
            f"Windows process identity inspection exited {completed.returncode}"
        )
    if len(completed.stdout) > 16384:
        raise ValueError("Windows process identity output too large")
    data: Any = json.loads(completed.stdout.lstrip("\ufeff"))
    return None if data is None else WindowsProcessIdentity.model_validate(data)


def _same_windows_path(left: str, right: str) -> bool:
    return ntpath.normcase(ntpath.normpath(left)) == ntpath.normcase(ntpath.normpath(right))


class ComfyReceiptStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().absolute()

    def _check(self) -> None:
        if any(part.is_symlink() for part in (self.path, *self.path.parents)):
            raise ValueError("Refusing symlinked ComfyUI ownership receipt path")
        if self.path.exists() and not self.path.is_file():
            raise ValueError("ComfyUI ownership receipt is not a regular file")

    def load(self) -> ComfyOwnershipReceipt | None:
        self._check()
        if not self.path.exists():
            return None
        if self.path.stat().st_size > _MAX_RECEIPT_BYTES:
            raise ValueError("ComfyUI ownership receipt exceeds size limit")
        try:
            return ComfyOwnershipReceipt.model_validate_json(
                self.path.read_bytes()
            )
        except (ValidationError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("Invalid ComfyUI ownership receipt") from exc

    def save(self, receipt: ComfyOwnershipReceipt) -> None:
        self._check()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._check()
        raw = receipt.model_dump_json(indent=2).encode("utf-8")
        if len(raw) > _MAX_RECEIPT_BYTES:
            raise ValueError("ComfyUI ownership receipt exceeds size limit")
        # A fresh filename avoids overwriting symlinks. Windows os.replace()
        # changes the receipt atomically once it is completely written.
        import uuid

        staging = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with staging.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self._check()
            staging.replace(self.path)
        finally:
            staging.unlink(missing_ok=True)

    def retire(self, expected: ComfyOwnershipReceipt) -> None:
        actual = self.load()
        if actual != expected:
            raise RuntimeError("ComfyUI ownership receipt changed; refusing retirement")
        self.path.unlink()


def expected_receipt(
    settings: ArtifexSettings,
    identity: WindowsProcessIdentity,
    *,
    launcher_identity: WindowsProcessIdentity | None = None,
) -> ComfyOwnershipReceipt:
    config = settings.render_agent.comfyui_process
    if config.executable is None or config.working_directory is None:
        raise ValueError("Missing managed ComfyUI executable or working directory")
    executable = str(config.executable.expanduser().resolve(strict=True))
    workdir = str(config.working_directory.expanduser().resolve(strict=True))
    from urllib.parse import urlsplit

    port = urlsplit(settings.comfyui.base_url).port
    if port is None:
        raise ValueError("Missing local ComfyUI port")
    command = subprocess.list2cmdline([executable, *config.arguments])
    return ComfyOwnershipReceipt(
        host=socket.gethostname(),
        node_id=settings.render_agent.node_id,
        port=port,
        executable=executable,
        working_directory=workdir,
        expected_command_line=command,
        identity=identity,
        schema_version=2 if launcher_identity is not None else 1,
        launcher_identity=launcher_identity,
    )


def _started_not_before(process: WindowsProcessIdentity, launcher: WindowsProcessIdentity) -> bool:
    try:
        child_time = datetime.fromisoformat(process.started_utc.replace("Z", "+00:00"))
        launch_time = datetime.fromisoformat(launcher.started_utc.replace("Z", "+00:00"))
        return child_time >= launch_time
    except ValueError:
        return False


def verified_launcher_child(
    settings: ArtifexSettings,
    launcher: WindowsProcessIdentity,
    child: WindowsProcessIdentity,
) -> bool:
    """Require direct Windows ancestry and exact launch argv, not merely a port."""
    config = settings.render_agent.comfyui_process
    if config.executable is None or not config.arguments:
        return False
    try:
        executable = str(config.executable.expanduser().resolve(strict=True))
    except (OSError, ValueError):
        return False
    expected_command = subprocess.list2cmdline([executable, *config.arguments])
    # Only Python venv shims may redirect to another Python executable.
    if ntpath.basename(executable).casefold() not in {"python.exe", "pythonw.exe"}:
        return False
    if not _same_windows_path(launcher.executable, executable):
        return False
    if launcher.command_line != expected_command:
        return False
    if child.pid == launcher.pid or child.parent_pid != launcher.pid:
        return False
    if ntpath.basename(child.executable).casefold() != ntpath.basename(executable).casefold():
        return False
    expected_tail = " " + subprocess.list2cmdline(config.arguments)
    return (
        child.command_line.endswith(expected_tail)
        and _started_not_before(child, launcher)
    )


def matches_owned_process(
    settings: ArtifexSettings,
    receipt: ComfyOwnershipReceipt,
    process: WindowsProcessIdentity,
) -> bool:
    """Every value must agree. PID equality alone is not sufficient."""
    try:
        expected = expected_receipt(
            settings, process, launcher_identity=receipt.launcher_identity,
        )
    except (ValueError, OSError):
        return False
    if receipt.schema_version not in {1, 2}:
        return False
    if receipt.schema_version == 1:
        execution_verified = (
            receipt.launcher_identity is None
            and _same_windows_path(process.executable, expected.executable)
            and process.command_line == expected.expected_command_line
        )
    else:
        execution_verified = (
            receipt.launcher_identity is not None
            and verified_launcher_child(settings, receipt.launcher_identity, process)
        )
    return (
        execution_verified
        and receipt.host == expected.host
        and receipt.node_id == expected.node_id
        and receipt.port == expected.port
        and _same_windows_path(receipt.executable, expected.executable)
        and _same_windows_path(receipt.working_directory, expected.working_directory)
        and receipt.expected_command_line == expected.expected_command_line
        and receipt.identity == process
    )
