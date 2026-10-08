from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings

# Fixed script, no user-controlled interpolation into PowerShell.
_CONNECTION_SCRIPT = (
    "$ErrorActionPreference='Stop'; "
    "$c=@(Get-NetTCPConnection -ErrorAction Stop | "
    "Select-Object LocalAddress,LocalPort,RemoteAddress,RemotePort,"
    "State,OwningProcess); "
    "ConvertTo-Json -InputObject $c -Compress -Depth 3"
)
_LOOPBACK = frozenset({"127.0.0.1", "::1", "0:0:0:0:0:0:0:1"})


class WindowsTcpSocket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    local_address: str = Field(alias="LocalAddress")
    local_port: int = Field(alias="LocalPort", ge=0, le=65535)
    remote_address: str = Field(alias="RemoteAddress")
    remote_port: int = Field(alias="RemotePort", ge=0, le=65535)
    state: str = Field(alias="State")
    owning_process: int = Field(alias="OwningProcess", ge=0)


AuditStatus = Literal[
    "not_windows", "probe_failed", "missing_listener", "lan_exposed",
    "foreign_listener", "owner_unverified", "unexpected_local_clients",
    "owned_loopback_observed",
]


class RendererSocketAudit(BaseModel):
    """A point-in-time TCP inventory, NEVER proof no future /prompt can arrive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AuditStatus
    upstream_port: int
    expected_owned_pid: int | None
    listener_pids: tuple[int, ...] = ()
    listener_addresses: tuple[str, ...] = ()
    unexpected_client_pids: tuple[int, ...] = ()
    owner_verified: bool = False
    probe_error: str | None = None
    future_local_submissions_excluded: bool = False
    restart_authorized: bool = False
    production_qualified: bool = False


def _parse_connections(value: str) -> tuple[WindowsTcpSocket, ...]:
    """Reject a partial/ambiguous TCP inventory rather than assuming isolation."""
    payload: Any = json.loads(value.lstrip("\ufeff"))
    if not isinstance(payload, list):
        raise ValueError("Windows TCP probe must return a JSON array")
    return tuple(WindowsTcpSocket.model_validate(item) for item in payload)


def _windows_connections() -> tuple[WindowsTcpSocket, ...]:
    if os.name != "nt":
        raise OSError("Get-NetTCPConnection requires Windows")
    result = subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-NonInteractive",
            "-Command", _CONNECTION_SCRIPT,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        # Do not expose raw stderr: it may contain local paths or OS details.
        raise OSError(f"TCP inventory failed with exit status {result.returncode}")
    return _parse_connections(result.stdout)


def audit_renderer_sockets(
    settings: ArtifexSettings,
    *,
    owned_pid: int | None = None,
    gateway_pid: int | None = None,
    windows: bool | None = None,
    probe: Callable[[], tuple[WindowsTcpSocket, ...]] = _windows_connections,
) -> RendererSocketAudit:
    """Read-only, conservative ownership and direct-client snapshot on PC-B.

    Called with an actual Popen.pid by an in-process manager to compare the
    listener owner. CLI without that handle only provides diagnostic evidence.
    """
    parsed = urlsplit(settings.comfyui.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.port is None
        or parsed.path not in {"", "/"}
        or parsed.username is not None or parsed.password is not None
        or parsed.query or parsed.fragment
    ):
        raise ValueError("socket audit must target the renderer's local ComfyUI URL")
    port = parsed.port
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return RendererSocketAudit(
            status="not_windows", upstream_port=port, expected_owned_pid=owned_pid,
        )
    try:
        connections = probe()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return RendererSocketAudit(
            status="probe_failed", upstream_port=port, expected_owned_pid=owned_pid,
            probe_error=f"{type(exc).__name__}: {exc}",
        )

    listeners = tuple(
        item for item in connections
        if item.local_port == port and item.state.casefold() in {"listen", "listening"}
    )
    pids = tuple(sorted({item.owning_process for item in listeners}))
    addresses = tuple(sorted({item.local_address for item in listeners}))
    direct = tuple(sorted({
        item.owning_process for item in connections
        if item.remote_port == port
        and item.remote_address in _LOOPBACK
        and item.state.casefold() in {"established", "synsent", "syn_sent"}
        and item.owning_process not in {gateway_pid, owned_pid}
    }))
    owner_verified = bool(owned_pid and pids == (owned_pid,) and listeners)
    if not listeners:
        state: AuditStatus = "missing_listener"
    elif any(address not in _LOOPBACK for address in addresses):
        state = "lan_exposed"
    elif owned_pid is not None and not owner_verified:
        state = "foreign_listener"
    elif direct:
        state = "unexpected_local_clients"
    elif not owner_verified:
        state = "owner_unverified"
    else:
        state = "owned_loopback_observed"
    return RendererSocketAudit(
        status=state, upstream_port=port,
        expected_owned_pid=owned_pid, listener_pids=pids,
        listener_addresses=addresses,
        unexpected_client_pids=direct,
        owner_verified=owner_verified,
    )
