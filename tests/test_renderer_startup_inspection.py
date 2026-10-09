from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.render_node.socket_audit import (
    WindowsTcpSocket,
    _windows_connections,
)
from artifex.render_node.startup_inspection import (
    RendererStartupInspection,
    inspect_renderer_startup,
)

CHECKS = (
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
)


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_agent.comfyui_process.enabled = True
    settings.render_agent.comfyui_process.executable = Path("D:/AI/venv/Scripts/python.exe")
    settings.render_agent.comfyui_process.working_directory = Path("D:/AI/ComfyUI")
    settings.render_agent.comfyui_process.arguments = (
        "main.py", "--listen", "127.0.0.1", "--port", "8188",
    )
    settings.render_agent.gateway.enabled = True
    settings.render_agent.gateway.bind_host = "0.0.0.0"
    settings.render_agent.gateway.port = 8191
    settings.render_agent.port = 8190
    settings.render_agent.require_token = True
    settings.render_agent.token_env = "ARTIFEX_RENDER_NODE_TOKEN"
    settings.comfyui.base_url = "http://127.0.0.1:8188"
    return settings


def _socket(
    port: int,
    pid: int = 101,
    address: str = "127.0.0.1",
) -> WindowsTcpSocket:
    return WindowsTcpSocket(
        LocalAddress=address, LocalPort=port,
        RemoteAddress="0.0.0.0", RemotePort=0,
        OwningProcess=pid, State="Listen",
    )


def _audit(
    settings: ArtifexSettings, *,
    config: Path,
    status: str = "observed_stable",
) -> dict[str, Any]:
    return {
        "status": status,
        "actual_listener_pid": 101,
        "process_observation_verified": status == "observed_stable",
        "checks": {key: {"status": "pass", "reason": "verified"} for key in CHECKS},
        "restart_authorized": False,
        "child_survival_qualified": False,
        "production_qualified": False,
        "mutated_services": False,
    }


def test_stable_verified_process_remains_running_no_launch_permission() -> None:
    sockets = (_socket(8188), _socket(8190, 102, "0.0.0.0"),
               _socket(8191, 102, "0.0.0.0"))
    called: list[str] = []

    def probe() -> tuple[WindowsTcpSocket, ...]:
        called.append("TCP")
        return sockets

    def owner(settings: ArtifexSettings, *, config: Path) -> dict[str, Any]:
        called.append("OWNER")
        return _audit(settings, config=config)

    result = inspect_renderer_startup(
        _settings(), config_path=Path("config/render-node.yaml"),
        windows=True, socket_probe=probe, owner_probe=owner,
    )
    assert result.status == "owned_observed"
    assert result.actual_owned_listener_pid == 101
    assert result.config_protected and result.snapshots_consistent
    assert result.service_ports_coherent is True
    assert len(result.listeners) == 3
    assert len(result.owner_checks) == 9
    assert called == ["TCP", "TCP", "OWNER"]
    assert result.restart_authorized is False
    assert result.launch_authorized is False
    assert result.reattach_authorized is False
    assert result.mutated_services is False
    assert result.production_qualified is False


@pytest.mark.parametrize(
    ("attestation", "gateway"),
    [
        (None, _socket(8191, 102, "0.0.0.0")),
        (_socket(8190, 102, "0.0.0.0"), None),
        (_socket(8190, 102, "0.0.0.0"), _socket(8191, 103, "0.0.0.0")),
        (_socket(8190, 101, "0.0.0.0"), _socket(8191, 101, "0.0.0.0")),
        (_socket(8190, 0, "0.0.0.0"), _socket(8191, 0, "0.0.0.0")),
    ],
)
def test_missing_foreign_or_divergent_service_pids_block_observed_owner(
    attestation: WindowsTcpSocket | None,
    gateway: WindowsTcpSocket | None,
) -> None:
    sockets = (_socket(8188),) + tuple(
        item for item in (attestation, gateway) if item is not None
    )
    report = inspect_renderer_startup(
        _settings(), config_path=Path("render-node.yaml"),
        windows=True, socket_probe=lambda: sockets, owner_probe=_audit,
    )
    assert report.owner_audit_status == "observed_stable"
    assert report.status == "service_ports_blocked"
    assert not report.service_ports_coherent
    assert report.actual_owned_listener_pid is None
    assert report.blockers and report.next_actions
    assert report.launch_authorized is False
    assert report.restart_authorized is False
    assert report.reattach_authorized is False
    assert report.production_qualified is False


def test_different_ipv4_ipv6_service_listeners_one_pid_are_coherent() -> None:
    sockets = (
        _socket(8188),
        _socket(8190, 102, "0.0.0.0"),
        _socket(8190, 102, "::"),
        _socket(8191, 102, "127.0.0.1"),
    )
    report = inspect_renderer_startup(
        _settings(), config_path=Path("render-node.yaml"),
        windows=True, socket_probe=lambda: sockets, owner_probe=_audit,
    )
    assert report.status == "owned_observed"
    assert report.service_ports_coherent


@pytest.mark.parametrize("args", [
    ("main.py", "--listen", "127.0.0.1", "--listen", "0.0.0.0", "--port", "8188"),
    ("main.py", "--listen", "127.0.0.1", "--port", "8188", "--port", "9999"),
    ("main.py", "--listen", "127.0.0.1", "--port", "8189"),
    ("main.py", "--listen", "127.0.0.1"),
])
def test_invalid_managed_command_line_is_not_protected(args: tuple[str, ...]) -> None:
    settings = _settings()
    settings.render_agent.comfyui_process.arguments = args
    sockets = (
        _socket(8188),
        _socket(8190, 102, "0.0.0.0"),
        _socket(8191, 102, "0.0.0.0"),
    )
    report = inspect_renderer_startup(
        settings, config_path=Path("render-node.yaml"),
        windows=True, socket_probe=lambda: sockets, owner_probe=_audit,
    )
    assert not report.config_protected
    assert report.status != "owned_observed"
    assert report.launch_authorized is False


@pytest.mark.parametrize("args", [
    ("main.py", "--listen=127.0.0.1", "--port", "8188"),
    ("main.py", "--listen", "127.0.0.1", "--port=8188"),
])
def test_valid_equals_style_launch_args_match_actual_manager(args: tuple[str, ...]) -> None:
    settings = _settings()
    settings.render_agent.comfyui_process.arguments = args
    sockets = (
        _socket(8188),
        _socket(8190, 102, "0.0.0.0"),
        _socket(8191, 102, "0.0.0.0"),
    )
    report = inspect_renderer_startup(
        settings, config_path=Path("render-node.yaml"),
        windows=True, socket_probe=lambda: sockets, owner_probe=_audit,
    )
    assert report.config_protected
    assert report.status == "owned_observed"
    assert report.service_ports_coherent


def test_windows_tcp_timeout_returns_blocked_json_not_uncaught_exception() -> None:
    def timeout() -> tuple[WindowsTcpSocket, ...]:
        raise subprocess.TimeoutExpired(cmd="powershell.exe", timeout=15)

    report = inspect_renderer_startup(
        _settings(), config_path=Path("render-node.yaml"),
        windows=True, socket_probe=timeout,
    )
    assert report.status == "probe_failed"
    assert report.blockers and not report.snapshots_consistent
    assert "powershell.exe" not in report.model_dump_json()
    assert report.launch_authorized is False


def test_free_port_does_not_authorize_new_gpu_child() -> None:
    def forbidden(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pytest.fail("Owner lookup should not run with no listener")

    result = inspect_renderer_startup(
        _settings(), config_path=Path("config/render-node.yaml"),
        windows=True, socket_probe=lambda: (), owner_probe=forbidden,
    )
    assert result.status == "no_listener"
    assert result.blockers and result.next_actions
    assert not result.launch_authorized
    assert not result.snapshots_consistent is False


@pytest.mark.parametrize(
    ("sockets", "expected"),
    [
        ((_socket(8188, 202),), "unverified_listener"),
        ((_socket(8188, 101, "0.0.0.0"),), "unprotected_listener"),
        ((_socket(8188, 101), _socket(8188, 202)), "unverified_listener"),
    ],
)
def test_foreign_or_exposed_listener_is_not_adopted(
    sockets: tuple[WindowsTcpSocket, ...], expected: str,
) -> None:
    result = inspect_renderer_startup(
        _settings(), config_path=Path("render-node.yaml"),
        windows=True, socket_probe=lambda: sockets, owner_probe=_audit,
    )
    assert result.status == expected
    assert result.blockers
    assert result.actual_owned_listener_pid is None
    assert not result.launch_authorized and not result.reattach_authorized


def test_audit_unknown_or_missing_check_cannot_fake_owned_listener() -> None:
    for remove in ["receipt", "tcp_ownership", "scheduler_policy"]:
        def owner(
            settings: ArtifexSettings, *, config: Path, removed: str = remove,
        ) -> dict[str, Any]:
            payload = _audit(settings, config=config)
            del payload["checks"][removed]
            return payload

        result = inspect_renderer_startup(
            _settings(), config_path=Path("render-node.yaml"),
            windows=True,
            socket_probe=lambda: (_socket(8188),),
            owner_probe=owner,
        )
        assert result.status == "unverified_listener"
        assert not result.actual_owned_listener_pid


def test_tcp_drift_fails_closed_and_does_not_trust_owner() -> None:
    snapshots = [(_socket(8188, 101),), (_socket(8188, 202),)]

    def probe() -> tuple[WindowsTcpSocket, ...]:
        return snapshots.pop(0)

    def owner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pytest.fail("Changing PID must not reach stable-owner audit")

    result = inspect_renderer_startup(
        _settings(), config_path=Path("render-node.yaml"),
        windows=True, socket_probe=probe, owner_probe=owner,
    )
    assert result.status == "changing_ports"
    assert not result.snapshots_consistent
    assert result.blockers
    assert not result.launch_authorized


def test_tcp_inventory_error_never_assumes_no_listener() -> None:
    def unavailable() -> tuple[WindowsTcpSocket, ...]:
        raise OSError("Get-NetTCPConnection unavailable")

    report = inspect_renderer_startup(
        _settings(), config_path=Path("render-node.yaml"),
        windows=True, socket_probe=unavailable,
    )
    assert report.status == "probe_failed"
    assert not report.snapshots_consistent
    assert report.blockers
    assert "Get-NetTCPConnection unavailable" not in report.model_dump_json()


def test_invalid_protected_config_blocked_even_when_owner_says_stable() -> None:
    settings = _settings()
    settings.render_agent.comfyui_process.arguments = (
        "main.py", "--listen", "0.0.0.0", "--port", "8188",
    )
    report = inspect_renderer_startup(
        settings, config_path=Path("render-node.yaml"),
        windows=True, socket_probe=lambda: (_socket(8188),), owner_probe=_audit,
    )
    assert not report.config_protected
    assert report.status == "unverified_listener"
    assert report.blockers
    assert not report.launch_authorized


def test_non_windows_skips_all_local_windows_probes() -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("No PowerShell or CIM on non-Windows")

    result = inspect_renderer_startup(
        _settings(), config_path=Path("render-node.yaml"),
        windows=False, socket_probe=forbidden, owner_probe=forbidden,
    )
    assert result.status == "unsupported"
    assert not result.windows_native and result.blockers


def test_duplicate_ports_are_blocked_not_misattributed() -> None:
    settings = _settings()
    settings.render_agent.gateway.port = 8188
    report = inspect_renderer_startup(
        settings, config_path=Path("render-node.yaml"),
        windows=True, socket_probe=lambda: (_socket(8188),),
    )
    assert not report.config_protected
    assert report.status != "owned_observed"
    assert report.blockers
    assert not report.launch_authorized


@pytest.mark.skipif(os.name != "nt", reason="Requires native Windows listener and PowerShell")
def test_real_windows_socket_pid_remains_unverified_without_receipt() -> None:
    """Even a real, live process owning the port is not managed GPU evidence."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as live:
        live.bind(("127.0.0.1", 0))
        live.listen(1)
        port = live.getsockname()[1]
        settings = _settings()
        settings.comfyui.base_url = f"http://127.0.0.1:{port}"
        settings.render_agent.gateway.port = 8191 if port != 8191 else 9191
        settings.render_agent.port = 8190 if port != 8190 else 9190
        if settings.render_agent.gateway.port == settings.render_agent.port:
            settings.render_agent.gateway.port = 9191
        report = inspect_renderer_startup(
            settings, config_path=Path("render-node.yaml"),
            windows=True, socket_probe=_windows_connections,
            owner_probe=lambda *_args, **_kwargs: _audit(settings, config=Path("x"),
                                                         status="blocked"),
        )
        assert report.status == "unverified_listener"
        assert any(x.pid == os.getpid() for x in report.listeners)
        assert report.launch_authorized is False


def test_cli_no_config_and_no_clobber_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.cli as cli_module
    import artifex.render_node.startup_inspection as startup

    cli = CliRunner()
    config = tmp_path / "renderer.yaml"
    missing = cli.invoke(
        app, ["onboard", "renderer-safety", "--config", str(config)],
    )
    assert missing.exit_code == 1
    assert "non-symlinked PC-B configuration" in missing.output
    config.write_text("{}\n", encoding="utf-8")
    report = RendererStartupInspection(
        captured_utc=startup.datetime.now(startup.UTC),
        status="no_listener", renderer_node_id="renderer",
        configured_comfyui_port=8188, configured_attestation_port=8190,
        configured_gateway_port=8191, config_protected=True,
        windows_native=True, snapshots_consistent=True,
        blockers=("No observed owned renderer",),
    )
    called: list[int] = []

    def inspect(*args: Any, **kwargs: Any) -> RendererStartupInspection:
        called.append(1)
        return report

    monkeypatch.setattr(cli_module, "_settings", lambda config: _settings())
    monkeypatch.setattr(startup, "inspect_renderer_startup", inspect)
    path = tmp_path / "evidence" / "report.json"
    args = ["onboard", "renderer-safety", "--config", str(config),
            "--output", str(path), "--json"]
    output = cli.invoke(app, args)
    assert output.exit_code == 1
    assert json.loads(output.output)["launch_authorized"] is False
    assert len(called) == 1
    assert json.loads(path.read_text())["status"] == "no_listener"
    second = cli.invoke(app, args)
    assert second.exit_code == 1
    assert "FileExistsError" in second.output
    assert json.loads(path.read_text())["mutated_services"] is False
