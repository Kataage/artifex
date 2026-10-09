"""Native PC-A/PC-B installation audit is strictly observational."""
from __future__ import annotations

import json
import socket
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.pair_installation import inspect_two_pc_installation
from artifex.render_node.attestation import serve_attestation
from artifex.render_node.client import fetch_remote_installation_audit
from artifex.render_node.observer_heartbeat import publish_observer_heartbeat
from artifex.render_node.installation_audit import (
    RemoteRendererInstallationAudit,
    inspect_registered_task,
    inspect_renderer_installation,
    inspect_spool,
)
from artifex.windows_tasks import StartupTaskStatus

NOW = datetime(2026, 10, 10, tzinfo=UTC)


def _task(role: str, *, installed: bool = True,
          managed: bool = True, state: str = "Running") -> StartupTaskStatus:
    return StartupTaskStatus.model_validate({
        "role": role, "task_name": "Artifex-" + role,
        "installed": installed, "managed": managed, "state": state,
        "execute": "C:/private/venv/python.exe",
        "arguments": "C:/private/path/with/token",
        "working_directory": "C:/private",
        "action_count": 1, "allow_hard_terminate": False,
        "multiple_instances": "IgnoreNew",
        "execution_time_limit_seconds": 0,
    })


def _local_ok(role: str, *, config: Path, status: StartupTaskStatus):
    return True, "test-only verdict"


def _settings(tmp_path: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_agent.node_id = "gpu-b"
    settings.render_agent.comfyui_process.enabled = True
    settings.render_agent.comfyui_process.executable = tmp_path / "comfy-python.exe"
    settings.render_agent.comfyui_process.working_directory = tmp_path
    settings.render_agent.survival_evidence_dir = tmp_path / "spool"
    settings.render_nodes.primary = "gpu-b"
    settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://127.0.0.1:8188",
        attestation_url="http://127.0.0.1:8190",
    )
    return settings


def _remote(
    settings: ArtifexSettings,
    config: Path,
    *,
    renderer: str = "Running",
    observer: str = "Running",
    time: datetime = NOW,
) -> RemoteRendererInstallationAudit:
    publish_observer_heartbeat(
        settings, observed_utc=time,
        poll_seconds=15, sample_count=1, state="verified",
    )
    report = inspect_renderer_installation(
        settings, owner_config=config, native_windows=True, now=time,
        probe=lambda role: _task(
            role, state=renderer if role == "renderer" else observer
        ),
        verify=_local_ok,
    )
    return RemoteRendererInstallationAudit(
        node_id="gpu-b", audit=report,
    )


@pytest.mark.parametrize(("change", "expected"), [
    ("ok", "running"), ("missing", "missing"),
    ("unowned", "unsafe"), ("stopped", "registered_not_running"),
    ("wrong_policy", "unsafe"), ("offline", "unavailable"),
    ("not_windows", "unavailable"),
])
def test_task_inspection_does_not_disclose_command_or_mutate(
    tmp_path: Path, change: str, expected: str,
) -> None:
    called: list[str] = []
    def probe(role: str):
        called.append("read:" + role)
        if change == "offline":
            raise OSError("private Windows path and password")
        return _task(
            role, installed=change != "missing",
            managed=change != "unowned",
            state="Ready" if change == "stopped" else "Running",
        )
    def verify(role: str, *, config: Path, status: StartupTaskStatus):
        called.append("verify:" + role)
        return change != "wrong_policy", "private task path"
    result = inspect_registered_task(
        "survival-observer", config=tmp_path / "renderer.yaml",
        native_windows=change != "not_windows", probe=probe, verify=verify,
    )
    assert result.status == expected
    assert "private" not in result.model_dump_json().lower()
    assert all(x.startswith(("read:", "verify:")) for x in called)


def test_pc_b_readiness_uses_both_tasks_and_spool_without_gpu(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    conf = tmp_path / "render-node.yaml"
    conf.write_text("{}")
    (tmp_path / "spool").mkdir()
    yes = _remote(settings, conf)
    assert yes.audit.safe_for_passive_observation
    assert yes.audit.observer_heartbeat == "fresh"
    assert yes.audit.evidence_spool == "empty"
    assert yes.audit.actual_survival_observed is False
    assert yes.audit.production_qualified is False
    stopped = _remote(settings, conf, observer="Ready")
    assert not stopped.audit.safe_for_passive_observation
    assert stopped.audit.survival_observer_task.status == "registered_not_running"
    assert "python.exe" not in yes.model_dump_json()
    assert "C:/private" not in yes.model_dump_json()
    assert inspect_spool(tmp_path / "spool") == "empty"


def test_spool_rejects_symlink_and_excessive_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "spool"
    root.mkdir()
    name = "survival-20261010T000000Z-" + "1" * 32 + ".json"
    (root / name).write_text("{}")
    assert inspect_spool(root) == "has_evidence"
    link = tmp_path / "linked"
    try:
        link.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Windows CI may deny developer-mode symlink")
    assert inspect_spool(link) == "unsafe"


def test_pc_a_aggregates_native_state_without_any_task_actions(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    controller_config = tmp_path / "local.yaml"
    controller_config.write_text("{}")
    renderer_config = tmp_path / "renderer.yaml"
    renderer_config.write_text("{}")
    (tmp_path / "spool").mkdir()
    remote = _remote(settings, renderer_config)
    actions: list[str] = []
    def fetch(node: str, cfg: RenderNodeConfig):
        actions.append("GET")
        assert node == "gpu-b"
        return remote
    result = inspect_two_pc_installation(
        settings, controller_config=controller_config, now=NOW,
        native_windows=True,
        local_probe=lambda role: _task(role),
        local_verify=_local_ok,
        remote_probe=fetch,
    )
    assert result.status == "ready_for_passive_monitoring"
    assert result.blockers == ()
    assert actions == ["GET"]
    assert result.commands_executed is False
    assert result.gpu_jobs_submitted is False
    assert result.production_qualified is False
    assert result.real_machine_qualification_complete is False

    stopped = inspect_two_pc_installation(
        settings, controller_config=controller_config, now=NOW,
        native_windows=True,
        local_probe=lambda role: _task(role, state="Ready"),
        local_verify=_local_ok, remote_probe=fetch,
    )
    assert stopped.status == "pc_a_needs_setup"
    assert "pc_a_controller_task_registered_not_running" in stopped.blockers


@pytest.mark.parametrize(("problem", "expected"), [
    ("no_node", "pc_b_unconfigured"),
    ("missing_url", "pc_b_unconfigured"),
    ("offline", "pc_b_unreachable"),
    ("old_evidence", "pc_b_needs_setup"),
    ("wrong_node", "pc_b_needs_setup"),
    ("observer_stopped", "pc_b_needs_setup"),
    ("no_windows", "unsupported_platform"),
])
def test_pc_a_fail_closed_missing_prerequisites(
    tmp_path: Path, problem: str, expected: str,
) -> None:
    settings = _settings(tmp_path)
    path = tmp_path / "local.yaml"
    path.write_text("{}")
    renderer_config = tmp_path / "renderer.yaml"
    renderer_config.write_text("{}")
    (tmp_path / "spool").mkdir()
    if problem == "no_node":
        settings.render_nodes.nodes.clear()
        settings.render_nodes.primary = None
    if problem == "missing_url":
        settings.render_nodes.nodes["gpu-b"].attestation_url = None
    def remote(node: str, cfg: RenderNodeConfig):
        if problem == "offline":
            raise httpx.ConnectError("offline")
        result = _remote(
            settings, renderer_config,
            observer="Ready" if problem == "observer_stopped" else "Running",
            time=NOW - timedelta(minutes=20)
            if problem == "old_evidence" else NOW,
        )
        if problem == "wrong_node":
            return result.model_copy(update={"node_id": "foreign"})
        return result
    report = inspect_two_pc_installation(
        settings, controller_config=path, now=NOW,
        native_windows=problem != "no_windows",
        local_probe=lambda role: _task(role),
        local_verify=_local_ok,
        remote_probe=remote,
    )
    assert report.status == expected
    assert report.blockers
    assert report.production_qualified is False
    assert report.remote_task_actions_executed is False


def test_bearer_endpoint_fixed_path_no_query_and_no_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.render_node.attestation as att

    settings = _settings(tmp_path)
    settings.render_agent.require_token = True
    settings.render_agent.token_env = "ARTIFEX_RENDER_NODE_TOKEN"
    settings.render_agent.bind_host = "127.0.0.1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        settings.render_agent.port = sock.getsockname()[1]
    cfg = tmp_path / "renderer.yaml"
    cfg.write_text("{}")
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "unit-secret")
    calls: list[str] = []
    mocked = _remote(settings, cfg).audit
    def read_only(s: ArtifexSettings, *, owner_config: Path):
        calls.append("inspect_only")
        return mocked
    monkeypatch.setattr(att, "inspect_renderer_installation", read_only)
    stop = threading.Event()
    server = threading.Thread(
        target=serve_attestation, args=(settings,),
        kwargs={"stop_event": stop, "owner_config": cfg}, daemon=True,
    )
    server.start()
    addr = f"http://127.0.0.1:{settings.render_agent.port}"
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            for _ in range(30):
                try:
                    if client.get(addr + "/health").status_code == 200:
                        break
                except httpx.ConnectError:
                    stop.wait(0.03)
            assert client.get(addr + "/v1/installation-audit").status_code == 401
            header = {"Authorization": "Bearer unit-secret"}
            q = client.get(addr + "/v1/installation-audit?restart=1", headers=header)
            assert q.status_code == 503
            r = client.get(addr + "/v1/installation-audit", headers=header)
            assert r.status_code == 200
            assert r.json()["node_id"] == "gpu-b"
            assert r.json()["audit"]["production_qualified"] is False
            assert "unit-secret" not in r.text
            assert "C:/private" not in r.text
            assert calls == ["inspect_only"]
    finally:
        stop.set()
        server.join(timeout=5)


def test_remote_client_no_redirect_wrong_node_or_leaked_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    cfg = tmp_path / "renderer.yaml"
    cfg.write_text("{}")
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "unit-secret")
    payload = _remote(settings, cfg).model_dump(mode="json")
    config = settings.render_nodes.nodes["gpu-b"]
    def transport(req: httpx.Request):
        assert req.headers["authorization"] == "Bearer unit-secret"
        assert req.url.path == "/v1/installation-audit"
        return httpx.Response(200, json=payload)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        got = fetch_remote_installation_audit("gpu-b", config, client=client)
    assert got.node_id == "gpu-b"

    with (
        httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(302, headers={"location": "https://other.invalid"})
        )) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        fetch_remote_installation_audit("gpu-b", config, client=client)
    wrong = {**payload, "node_id": "foreign"}
    with (
        httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=wrong)
        )) as client,
        pytest.raises(ValueError, match="node identity"),
    ):
        fetch_remote_installation_audit("gpu-b", config, client=client)
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN")
    with pytest.raises(ValueError, match="token is missing"):
        fetch_remote_installation_audit("gpu-b", config)


def test_cli_reports_status_and_never_executes_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex import cli
    from artifex.qualification import pair_installation

    cfg = tmp_path / "local.yaml"
    cfg.write_text("{}")
    monkeypatch.setattr(cli, "_settings", lambda path: _settings(tmp_path))
    reports: list[Path] = []
    class Fake:
        status = "pc_b_needs_setup"
        def model_dump(self, *, mode: str):
            return {"status": self.status, "production_qualified": False}
    def inspect(settings, *, controller_config):
        reports.append(controller_config)
        return Fake()
    monkeypatch.setattr(pair_installation, "inspect_two_pc_installation", inspect)
    res = CliRunner().invoke(app, [
        "qualify", "pair-install-audit", "--config", str(cfg), "--json",
    ])
    assert res.exit_code == 1, res.output
    assert json.loads(res.stdout)["production_qualified"] is False
    assert reports == [cfg]
