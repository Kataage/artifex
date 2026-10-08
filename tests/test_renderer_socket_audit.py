from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.render_node.comfy_process import ManagedComfyUI
from artifex.render_node.socket_audit import (
    RendererSocketAudit,
    WindowsTcpSocket,
    _parse_connections,
    audit_renderer_sockets,
)


def _socket(
    local: str, port: int, remote: str, remote_port: int,
    state: str, pid: int,
) -> WindowsTcpSocket:
    return WindowsTcpSocket(
        LocalAddress=local, LocalPort=port, RemoteAddress=remote,
        RemotePort=remote_port, State=state, OwningProcess=pid,
    )


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.comfyui.base_url = "http://127.0.0.1:8188"
    return settings


def test_windows_inventory_parser_rejects_partial_and_malformed_payloads() -> None:
    socket = {
        "LocalAddress": "127.0.0.1", "LocalPort": 8188,
        "RemoteAddress": "0.0.0.0", "RemotePort": 0,
        "State": "Listen", "OwningProcess": 101,
    }
    parsed = _parse_connections("\ufeff" + json.dumps([socket]))
    assert len(parsed) == 1 and parsed[0].owning_process == 101
    for payload in (json.dumps(socket), "not-json", json.dumps([{}])):
        with pytest.raises((ValueError, TypeError)):
            _parse_connections(payload)


def test_tcp_audit_observes_owned_loopback_but_never_authorizes_restart() -> None:
    listener = _socket("127.0.0.1", 8188, "0.0.0.0", 0, "Listen", 101)
    proxy = _socket("127.0.0.1", 53000, "127.0.0.1", 8188, "Established", 200)
    report = audit_renderer_sockets(
        _settings(), windows=True, owned_pid=101, gateway_pid=200,
        probe=lambda: (listener, proxy),
    )
    assert report.status == "owned_loopback_observed"
    assert report.owner_verified and report.listener_pids == (101,)
    assert report.unexpected_client_pids == ()
    assert not report.future_local_submissions_excluded
    assert not report.restart_authorized and not report.production_qualified


@pytest.mark.parametrize(
    ("sockets", "pid", "state"),
    [
        ((), 101, "missing_listener"),
        ((_socket("0.0.0.0", 8188, "0.0.0.0", 0, "Listen", 101),), 101, "lan_exposed"),
        ((_socket("::", 8188, "::", 0, "Listen", 101),), 101, "lan_exposed"),
        ((_socket("127.0.0.1", 8188, "0.0.0.0", 0, "Listen", 999),), 101, "foreign_listener"),
        ((_socket("127.0.0.1", 8188, "0.0.0.0", 0, "Listen", 101),), None, "owner_unverified"),
        ((
            _socket("127.0.0.1", 8188, "0.0.0.0", 0, "Listen", 101),
            _socket("127.0.0.1", 55000, "127.0.0.1", 8188, "Established", 400),
        ), 101, "unexpected_local_clients"),
    ],
)
def test_socket_audit_rejects_exposure_foreign_owner_or_direct_client(
    sockets: tuple[WindowsTcpSocket, ...],
    pid: int | None,
    state: str,
) -> None:
    result = audit_renderer_sockets(
        _settings(), windows=True,
        owned_pid=pid, gateway_pid=200, probe=lambda: sockets,
    )
    assert result.status == state
    assert result.restart_authorized is False


def test_socket_audit_fails_closed_on_unsupported_os_and_probe_errors() -> None:
    skipped = audit_renderer_sockets(_settings(), windows=False)
    assert skipped.status == "not_windows" and not skipped.owner_verified

    def broken() -> tuple[WindowsTcpSocket, ...]:
        raise OSError("cannot inspect TCP")

    failed = audit_renderer_sockets(
        _settings(), windows=True, probe=broken, owned_pid=101,
    )
    assert failed.status == "probe_failed"
    assert "cannot inspect TCP" in (failed.probe_error or "")
    assert not failed.restart_authorized
    settings = _settings()
    settings.comfyui.base_url = "http://PC-B-LAN:8188"
    with pytest.raises(ValueError, match="local ComfyUI"):
        audit_renderer_sockets(settings, windows=True)


class LiveChild:
    pid = 101

    def __init__(self, *, exited: bool = False) -> None:
        self.exited = exited
        self.terminated = 0
        self.killed = 0

    def poll(self) -> int | None:
        return 1 if self.exited else None

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1


class FakeStop:
    def __init__(self, stop_after: int) -> None:
        self.stop_after = stop_after
        self.calls = 0

    def wait(self, seconds: float) -> bool:
        self.calls += 1
        return self.calls > self.stop_after


def test_closing_supervisor_never_terminates_a_live_gpu_child() -> None:
    manager = ManagedComfyUI(_settings())
    live = LiveChild()
    manager.process = live  # type: ignore[assignment]
    manager.close()
    assert live.terminated == live.killed == 0
    assert manager.process is None


def test_unhealthy_but_live_child_must_not_be_force_restarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ManagedComfyUI(_settings())
    manager.settings.render_agent.comfyui_process.enabled = True
    manager.settings.render_agent.comfyui_process.poll_seconds = 0.01
    live = LiveChild()
    manager.process = live  # type: ignore[assignment]
    monkeypatch.setattr(manager, "_healthy", lambda: False)
    with pytest.raises(RuntimeError, match="refusing an unsafe restart"):
        manager.watch(FakeStop(stop_after=10))  # type: ignore[arg-type]
    assert live.terminated == live.killed == 0
    assert manager.process is live
    manager.close()


def test_exited_owned_child_can_be_respawned_without_killing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ManagedComfyUI(_settings())
    manager.settings.render_agent.comfyui_process.enabled = True
    manager.settings.render_agent.comfyui_process.restart_backoff_seconds = 0
    child = LiveChild(exited=True)
    manager.process = child  # type: ignore[assignment]
    monkeypatch.setattr(manager, "_healthy", lambda: False)
    calls: list[str] = []
    monkeypatch.setattr(manager, "_start_owned", lambda: calls.append("spawn"))
    monkeypatch.setattr(manager, "_wait_ready", lambda stop: calls.append("ready"))
    manager.watch(FakeStop(stop_after=2))  # type: ignore[arg-type]
    assert calls == ["spawn", "ready"]
    assert child.terminated == child.killed == 0


def test_protected_supervisor_checks_real_listener_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ManagedComfyUI(_settings())
    manager.settings.render_agent.gateway.enabled = True
    manager.process = LiveChild()  # type: ignore[assignment]

    def probe(*args: Any, **kwargs: Any) -> RendererSocketAudit:
        return RendererSocketAudit(
            status="foreign_listener", upstream_port=8188,
            expected_owned_pid=101, listener_pids=(222,),
        )

    monkeypatch.setattr("artifex.render_node.comfy_process.audit_renderer_sockets", probe)
    with pytest.raises(RuntimeError, match="ownership is not proven"):
        manager._verify_owned_listener()
    assert manager.process is not None


def test_cli_socket_audit_is_read_only_and_fails_on_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.write_text("render_agent: {}\n", encoding="utf-8")
    runner = CliRunner()
    exposed = RendererSocketAudit(
        status="lan_exposed", upstream_port=8188,
        expected_owned_pid=None, listener_pids=(101,),
        listener_addresses=("0.0.0.0",),
    )
    monkeypatch.setattr("artifex.cli.audit_renderer_sockets", lambda settings: exposed)
    blocked = runner.invoke(
        app, ["render-node", "socket-audit", "--config", str(config)],
    )
    assert blocked.exit_code == 1
    assert '"restart_authorized": false' in blocked.output

    unknown_owner = exposed.model_copy(update={
        "status": "owner_unverified",
        "listener_addresses": ("127.0.0.1",),
    })
    monkeypatch.setattr(
        "artifex.cli.audit_renderer_sockets", lambda settings: unknown_owner,
    )
    observed = runner.invoke(
        app, ["render-node", "socket-audit", "--config", str(config)],
    )
    assert observed.exit_code == 0
    assert '"owner_verified": false' in observed.output

    missing = runner.invoke(
        app, ["render-node", "socket-audit", "--config", str(tmp_path / "no.yaml")],
    )
    assert missing.exit_code == 1
