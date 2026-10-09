"""Read-only, fail-closed PC-B port / owner inventory before possible startup.

This command observes live Windows sockets twice and, when a local ComfyUI
listener exists, uses the established full CIM/Scheduler/receipt owner audit.
It NEVER starts, reattaches, terminates or permits starting a GPU process.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings
from artifex.render_node.owner_audit import observe_renderer_owner
from artifex.render_node.socket_audit import (
    WindowsTcpSocket,
    _windows_connections,
)

_LOOPBACK = frozenset({"127.0.0.1", "::1", "0:0:0:0:0:0:0:1"})
_REQUIRED_OWNER = frozenset({
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
})

Status = Literal[
    "unsupported", "probe_failed", "configuration_blocked", "changing_ports",
    "unprotected_listener", "unverified_listener", "owned_observed",
    "no_listener",
]


class PortListener(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["comfyui", "attestation", "gateway"]
    address: str
    port: int = Field(ge=1, le=65535)
    pid: int = Field(ge=0)


class RendererStartupInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    captured_utc: datetime
    status: Status
    renderer_node_id: str
    configured_comfyui_port: int
    configured_attestation_port: int
    configured_gateway_port: int
    config_protected: bool
    windows_native: bool
    snapshots_consistent: bool
    listeners: tuple[PortListener, ...] = ()
    owner_audit_status: str | None = None
    actual_owned_listener_pid: int | None = Field(default=None, ge=1)
    owner_checks: dict[str, Literal["pass", "fail", "unknown"]] = Field(
        default_factory=dict,
    )
    blockers: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    restart_authorized: Literal[False] = False
    launch_authorized: Literal[False] = False
    reattach_authorized: Literal[False] = False
    mutated_services: Literal[False] = False
    production_qualified: Literal[False] = False


def _listeners(
    inventory: tuple[WindowsTcpSocket, ...],
    ports: dict[int, Literal["comfyui", "attestation", "gateway"]],
) -> tuple[PortListener, ...]:
    listeners = [
        PortListener(
            role=ports[item.local_port], address=item.local_address,
            port=item.local_port, pid=item.owning_process,
        )
        for item in inventory
        if item.local_port in ports
        and item.state.casefold() in {"listen", "listening"}
    ]
    return tuple(sorted(
        set(listeners),
        key=lambda value: (value.role, value.port, value.address, value.pid),
    ))


def inspect_renderer_startup(
    settings: ArtifexSettings,
    *,
    config_path: Path,
    windows: bool | None = None,
    socket_probe: Callable[[], tuple[WindowsTcpSocket, ...]] = _windows_connections,
    owner_probe: Callable[..., dict[str, Any]] = observe_renderer_owner,
) -> RendererStartupInspection:
    """Never treat a momentarily free port, receipt, or PID as start authority."""
    proc = settings.render_agent.comfyui_process
    gateway = settings.render_agent.gateway
    agent = settings.render_agent
    base = urlsplit(settings.comfyui.base_url)
    comfy_port = base.port
    if comfy_port is None:
        raise ValueError("PC-B ComfyUI upstream must specify a port")
    ports = [comfy_port, agent.port, gateway.port]
    # Equal ports make the snapshot ambiguous even when nothing is listening.
    unique = len(set(ports)) == 3
    local = (
        base.scheme == "http"
        and base.hostname in {"127.0.0.1", "::1", "localhost"}
        and base.username is None and base.password is None
        and base.path in {"", "/"} and not base.query and not base.fragment
    )
    args = proc.arguments
    try:
        index = args.index("--listen")
    except ValueError:
        loopback_argv = False
    else:
        loopback_argv = (
            index + 1 < len(args) and args[index + 1] in _LOOPBACK
        )
    protected = (
        local and unique and proc.enabled and gateway.enabled
        and agent.require_token and agent.token_env is not None
        and proc.executable is not None and proc.working_directory is not None
        and loopback_argv
    )
    native = os.name == "nt" if windows is None else windows
    blockers: list[str] = []
    actions: list[str] = []
    if not protected:
        blockers.append(
            "Protected managed ComfyUI, unique ports, loopback launch and authenticated gateway are required"
        )
        actions.append(
            "Review PC-B config; re-run onboard renderer-auto in preview mode. "
            "Do not switch a live ComfyUI to managed ownership."
        )
    if not native:
        blockers.append("Live Windows CIM/TCP inspection requires native Windows PC-B")
        actions.append("Run this inspection on the actual Windows PC-B.")
        state: Status = "unsupported"
        return RendererStartupInspection(
            captured_utc=datetime.now(UTC), status=state,
            renderer_node_id=agent.node_id,
            configured_comfyui_port=comfy_port,
            configured_attestation_port=agent.port,
            configured_gateway_port=gateway.port,
            config_protected=protected, windows_native=False,
            snapshots_consistent=False, blockers=tuple(blockers),
            next_actions=tuple(actions),
        )
    lookup: dict[int, Literal["comfyui", "attestation", "gateway"]] = {}
    if unique:
        lookup = {
            comfy_port: "comfyui", agent.port: "attestation",
            gateway.port: "gateway",
        }
    else:
        blockers.append("Configured listener ports overlap and cannot be attributed")
    try:
        first = _listeners(socket_probe(), lookup)
        second = _listeners(socket_probe(), lookup)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        blockers.append(f"Windows TCP inventory cannot be verified: {type(exc).__name__}")
        actions.append("Inspect Windows Get-NetTCPConnection permissions without restarting ComfyUI.")
        return RendererStartupInspection(
            captured_utc=datetime.now(UTC), status="probe_failed",
            renderer_node_id=agent.node_id, configured_comfyui_port=comfy_port,
            configured_attestation_port=agent.port,
            configured_gateway_port=gateway.port,
            config_protected=protected, windows_native=True,
            snapshots_consistent=False, blockers=tuple(blockers),
            next_actions=tuple(actions),
        )
    consistent = first == second
    if not consistent:
        blockers.append("Windows TCP listener PID or address changed during observation")
        actions.append("Observe the live ports again without launching any service.")
        state = "changing_ports"
    else:
        comfy = [x for x in second if x.role == "comfyui"]
        if not comfy:
            state = "no_listener"
            blockers.append(
                "No ComfyUI listener observed; process/startup ownership is not established"
            )
            actions.append(
                "Inspect Windows startup task and ownership receipt separately; "
                "do not infer permission to launch from a free port."
            )
        elif any(x.address not in _LOOPBACK for x in comfy):
            state = "unprotected_listener"
            blockers.append("ComfyUI upstream is exposed outside local loopback")
            actions.append("Review ComfyUI binding during a separately authorized maintenance window.")
        else:
            # Even if the sockets look perfect, an arbitrary Python listener
            # is never treated as an Artifex-owned GPU child.
            state = "unverified_listener"
    audit_status: str | None = None
    owned_pid: int | None = None
    owner_checks: dict[str, Literal["pass", "fail", "unknown"]] = {}
    if consistent and any(x.role == "comfyui" for x in second):
        try:
            audit = owner_probe(settings, config=config_path)
            audit_status = str(audit.get("status", "inconclusive"))
            raw_checks = audit.get("checks")
            if isinstance(raw_checks, dict):
                owner_checks = {
                    key: value["status"]
                    for key, value in raw_checks.items()
                    if isinstance(key, str) and isinstance(value, dict)
                    and value.get("status") in {"pass", "fail", "unknown"}
                }
            reported_pid = audit.get("actual_listener_pid")
            actual_pid = (
                reported_pid if type(reported_pid) is int and reported_pid > 0
                else None
            )
            comfy = [x for x in second if x.role == "comfyui"]
            stable_owner = (
                protected and state == "unverified_listener"
                and audit_status == "observed_stable"
                and audit.get("process_observation_verified") is True
                and _REQUIRED_OWNER.issubset(owner_checks)
                and all(owner_checks[key] == "pass" for key in _REQUIRED_OWNER)
                and actual_pid is not None
                and {x.pid for x in comfy} == {actual_pid}
            )
            if stable_owner:
                state = "owned_observed"
                owned_pid = actual_pid
            else:
                blockers.append(
                    "Original ComfyUI process identity, TCP ownership or Scheduler/receipt not fully verified"
                )
                actions.append("Inspect render-node owner-audit and Windows Task Scheduler read-only.")
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            blockers.append(
                f"Native PC-B ownership audit could not complete: {type(exc).__name__}"
            )
            actions.append("Run render-node owner-audit without modifying process state.")
    if not protected and state in {"owned_observed", "no_listener"}:
        state = "configuration_blocked"
    if state == "unverified_listener" and not blockers:
        blockers.append("Existing listener cannot be adopted from port/PID observation")
        actions.append(
            "Leave the existing ComfyUI untouched; inspect actual ownership and task evidence."
        )
    if state == "owned_observed":
        # A verified running renderer should be left running; it is not
        # permission to launch a second process or restart its current child.
        actions.append(
            "Existing protected renderer observed; keep it running. "
            "Use regular qualification for sustained operational evidence."
        )
    return RendererStartupInspection(
        captured_utc=datetime.now(UTC), status=state,
        renderer_node_id=agent.node_id,
        configured_comfyui_port=comfy_port,
        configured_attestation_port=agent.port,
        configured_gateway_port=gateway.port,
        config_protected=protected, windows_native=True,
        snapshots_consistent=consistent, listeners=second,
        owner_audit_status=audit_status,
        actual_owned_listener_pid=owned_pid,
        owner_checks=owner_checks, blockers=tuple(blockers),
        next_actions=tuple(dict.fromkeys(actions)),
    )
