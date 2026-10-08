from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.render_node.owner_audit import (
    observe_renderer_owner,
    save_owner_observation,
)
from artifex.render_node.process_identity import (
    ComfyReceiptStore,
    WindowsProcessIdentity,
    expected_receipt,
)
from artifex.render_node.socket_audit import RendererSocketAudit


def _settings(tmp_path: Path) -> tuple[ArtifexSettings, WindowsProcessIdentity]:
    settings = ArtifexSettings()
    exe = tmp_path / "Scripts" / "python.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"test exe")
    workdir = tmp_path / "ComfyUI"
    workdir.mkdir()
    config = settings.render_agent.comfyui_process
    config.enabled = True
    config.executable = exe
    config.working_directory = workdir
    config.arguments = ("main.py", "--listen", "127.0.0.1", "--port", "8188")
    config.ownership_receipt_path = tmp_path / "owned.json"
    settings.render_agent.gateway.enabled = True
    settings.comfyui.base_url = "http://127.0.0.1:8188"
    identity = WindowsProcessIdentity(
        ProcessId=4242, ParentProcessId=123,
        CreationDate="2026-10-09T00:00:00Z", ExecutablePath=str(exe),
        CommandLine=subprocess.list2cmdline([str(exe), *config.arguments]),
    )
    ComfyReceiptStore(config.ownership_receipt_path).save(
        expected_receipt(settings, identity)
    )
    return settings, identity


def _socket(status: str = "owned_loopback_observed") -> RendererSocketAudit:
    return RendererSocketAudit(
        status=status, upstream_port=8188, expected_owned_pid=4242,
        listener_pids=(4242,) if status != "missing_listener" else (),
        listener_addresses=("127.0.0.1",) if status != "missing_listener" else (),
        owner_verified=status == "owned_loopback_observed",
    )


def _mock_probes(
    monkeypatch: pytest.MonkeyPatch, identity: WindowsProcessIdentity,
    *,
    state: str = "Running", socket_state: str = "owned_loopback_observed",
    task_matches: bool = True,
) -> dict[str, Any]:
    import artifex.render_node.owner_audit as module

    counts: dict[str, Any] = {"socket": 0, "identity": 0}
    monkeypatch.setattr(module, "_is_windows", lambda: True)
    monkeypatch.setattr(
        module, "task_status",
        lambda role: SimpleNamespace(state=state),
    )
    monkeypatch.setattr(
        module, "task_configuration_matches",
        lambda role, config, status: (task_matches, "test"),
    )

    def inspect(pid: int) -> WindowsProcessIdentity | None:
        counts["identity"] += 1
        return identity if pid == identity.pid else None

    def sockets(settings: ArtifexSettings, **kwargs: Any) -> RendererSocketAudit:
        counts["socket"] += 1
        return _socket(socket_state)

    monkeypatch.setattr(module, "windows_process_identity", inspect)
    monkeypatch.setattr(module, "audit_renderer_sockets", sockets)
    return counts


def test_owner_audit_observes_verified_process_and_rechecks_tcp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings, identity = _settings(tmp_path)
    counts = _mock_probes(monkeypatch, identity)
    report = observe_renderer_owner(settings, config=tmp_path / "render-node.yaml")
    assert report["status"] == "observed_stable"
    assert report["process_observation_verified"] is True
    assert report["actual_listener_pid"] == identity.pid
    assert report["receipt_schema"] == 1
    assert counts["socket"] == 2
    assert counts["identity"] == 2
    assert all(v["status"] == "pass" for v in report["checks"].values())
    assert report["restart_authorized"] is False
    assert report["child_survival_qualified"] is False
    assert report["production_qualified"] is False
    assert report["mutated_services"] is False


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ("missing_receipt", "blocked"),
        ("foreign_socket", "blocked"),
        ("unmanaged", "blocked"),
        ("unsafe_task", "blocked"),
        ("task_ready", "inconclusive"),
        ("recycled_pid", "blocked"),
        ("non_windows", "unsupported"),
    ],
)
def test_owner_audit_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    change: str, status: str,
) -> None:
    settings, identity = _settings(tmp_path)
    _mock_probes(
        monkeypatch, identity,
        socket_state="foreign_listener" if change == "foreign_socket"
        else "owned_loopback_observed",
        state="Ready" if change == "task_ready" else "Running",
        task_matches=change != "unsafe_task",
    )
    if change == "missing_receipt":
        settings.render_agent.comfyui_process.ownership_receipt_path.unlink()
    if change == "unmanaged":
        settings.render_agent.gateway.enabled = False
    if change == "recycled_pid":
        import artifex.render_node.owner_audit as module

        monkeypatch.setattr(
            module, "windows_process_identity",
            lambda pid: identity.model_copy(update={"started_utc": "2026-10-09T01:00:00Z"}),
        )
    if change == "non_windows":
        import artifex.render_node.owner_audit as module

        monkeypatch.setattr(module, "_is_windows", lambda: False)
    report = observe_renderer_owner(settings, config=tmp_path / "render-node.yaml")
    assert report["status"] == status
    assert report["restart_authorized"] is False
    assert report["production_qualified"] is False
    assert report["mutated_services"] is False


def test_owner_audit_ignores_dead_launcher_but_blocks_reused_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.render_node.owner_audit as module

    settings, identity = _settings(tmp_path)
    launcher = identity.model_copy(update={
        "pid": 101, "parent_pid": 17,
        "started_utc": "2026-10-08T23:59:59Z",
    })
    child = identity.model_copy(update={
        "parent_pid": 101,
        "executable": r"C:\Python312\python.exe",
        "command_line": subprocess.list2cmdline([
            r"C:\Python312\python.exe",
            *settings.render_agent.comfyui_process.arguments,
        ]),
    })
    # The receipt is valid only when the launcher launch argv are exact.
    cfg = settings.render_agent.comfyui_process
    assert cfg.executable is not None
    launcher = launcher.model_copy(update={
        "executable": str(cfg.executable),
        "command_line": subprocess.list2cmdline([str(cfg.executable), *cfg.arguments]),
    })
    ComfyReceiptStore(cfg.ownership_receipt_path).save(
        expected_receipt(settings, child, launcher_identity=launcher)
    )
    _mock_probes(monkeypatch, identity)
    monkeypatch.setattr(
        module, "windows_process_identity",
        lambda pid: child if pid == child.pid else None,
    )
    stable = observe_renderer_owner(settings, config=tmp_path / "render-node.yaml")
    assert stable["status"] == "observed_stable"
    assert stable["launcher_pid"] == 101
    assert stable["receipt_schema"] == 2

    monkeypatch.setattr(
        module, "windows_process_identity",
        lambda pid: child if pid == child.pid else (
            launcher.model_copy(update={"started_utc": "2026-10-09T02:00:00Z"})
            if pid == launcher.pid else None
        ),
    )
    reused = observe_renderer_owner(settings, config=tmp_path / "render-node.yaml")
    assert reused["status"] == "blocked"
    assert reused["checks"]["launcher_identity"]["status"] == "fail"


def test_snapshot_process_drift_is_never_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.render_node.owner_audit as module

    settings, identity = _settings(tmp_path)
    _mock_probes(monkeypatch, identity)
    seen = 0

    def replaced(pid: int) -> WindowsProcessIdentity:
        nonlocal seen
        seen += 1
        return identity if seen == 1 else identity.model_copy(
            update={"started_utc": "2026-10-09T01:00:00Z"}
        )

    monkeypatch.setattr(module, "windows_process_identity", replaced)
    report = observe_renderer_owner(settings, config=tmp_path / "render-node.yaml")
    assert report["checks"]["snapshot_consistency"]["status"] == "fail"
    assert report["status"] == "blocked"


def test_save_report_requires_explicit_new_file_and_refuses_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "evidence" / "observation.json"
    payload = {"status": "blocked", "production_qualified": False}
    assert not target.exists()
    save_owner_observation(payload, target)
    assert json.loads(target.read_text()) == payload
    with pytest.raises(FileExistsError):
        save_owner_observation(payload, target)
    link = tmp_path / "alias.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlinked"):
        save_owner_observation(payload, link)


def test_owner_audit_cli_never_writes_without_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, identity = _settings(tmp_path)
    _mock_probes(monkeypatch, identity)
    import artifex.cli as cli_module

    config = tmp_path / "render-node.yaml"
    config.write_text("render_agent: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "_settings", lambda path: settings)
    runner = CliRunner()
    result = runner.invoke(
        app, ["render-node", "owner-audit", "--config", str(config), "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "observed_stable"
    assert not (tmp_path / "observation.json").exists()

    output = tmp_path / "observation.json"
    saved = runner.invoke(
        app, [
            "render-node", "owner-audit", "--config", str(config),
            "--output", str(output), "--json",
        ],
    )
    assert saved.exit_code == 0, saved.output
    assert output.exists()
    assert json.loads(output.read_text())["production_qualified"] is False
    duplicate = runner.invoke(
        app, [
            "render-node", "owner-audit", "--config", str(config),
            "--output", str(output),
        ],
    )
    assert duplicate.exit_code == 1
    assert "overwrite" in duplicate.output


def test_owner_audit_cli_missing_config_does_not_touch_host(
    tmp_path: Path,
) -> None:
    response = CliRunner().invoke(
        app, ["render-node", "owner-audit", "--config", str(tmp_path / "missing.yaml")],
    )
    assert response.exit_code == 1
    assert "config required" in response.output
