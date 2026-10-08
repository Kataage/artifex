from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.windows_task_lifecycle_probe import (
    _atomic_json,
    _mock_child,
    _read_json,
    _register_script,
    _script_task_check,
    run_lifecycle_probe,
)


@pytest.mark.parametrize("mode", ["protected", "baseline"])
def test_preview_never_registers_or_writes(tmp_path: Path, mode: str) -> None:
    import artifex.windows_task_lifecycle_probe as lifecycle

    report_dir = tmp_path / "not-created"
    preview = run_lifecycle_probe(
        mode=mode, report_dir=report_dir,  # type: ignore[arg-type]
    )
    assert preview["status"] == "preview"
    assert preview["performed"] is False
    assert preview["temporary_task_only"] is True
    assert preview["production_qualified"] is False
    assert preview["restart_authorized"] is False
    assert not report_dir.exists()
    assert lifecycle._check_preview("protected")["child_survival_qualified"] is False


def test_cli_defaults_to_read_only_preview() -> None:
    response = CliRunner().invoke(app, ["startup", "lifecycle-probe"])
    assert response.exit_code == 0, response.output
    report = json.loads(response.output)
    assert report["status"] == "preview"
    assert report["performed"] is False
    assert report["production_qualified"] is False


def test_cli_rejects_unknown_mode() -> None:
    response = CliRunner().invoke(
        app, ["startup", "lifecycle-probe", "--mode", "production"],
    )
    assert response.exit_code == 1
    assert "mode must be protected or baseline" in response.output


@pytest.mark.parametrize("mode", ["protected", "baseline"])
def test_isolated_scripts_can_only_act_on_owned_temporary_task(mode: str) -> None:
    name = "Artifex-LifecycleProbe-abcdef"
    marker = "Artifex isolated lifecycle probe/abcdef"
    code = _register_script(name, marker, "C:/python.exe", "args", "C:/tmp", mode)  # type: ignore[arg-type]
    check = _script_task_check(name, marker, "C:/python.exe", "args")
    assert "Register-ScheduledTask" in code
    assert "Get-ScheduledTask" in code
    assert "Refusing to overwrite existing scheduled task" in code
    assert "Stop-ScheduledTask" not in code
    assert "DisallowHardTerminate = $true" in code if mode == "protected" else "DisallowHardTerminate" not in code
    assert "Artifex-Renderer" not in (code + check)
    assert "Artifex-Controller" not in (code + check)
    assert "Description -cne" in check
    assert "Actions" in check
    assert "modified action" in check


def test_heartbeat_is_atomic_and_nonce_bound(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    _atomic_json(path, {"nonce": "abc", "ticks": 3})
    assert _read_json(path) == {"nonce": "abc", "ticks": 3}
    assert not path.with_suffix(".tmp").exists()
    (tmp_path / "release").write_text("abc", encoding="utf-8")
    _mock_child(tmp_path, "abc", 1)
    exited = _read_json(tmp_path / "child-exited.json")
    assert exited is not None and exited["nonce"] == "abc"


def test_blocked_register_keeps_real_tasks_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.windows_task_lifecycle_probe as lifecycle

    monkeypatch.setattr(lifecycle, "_require_windows", lambda: None)
    scripts: list[str] = []

    def denied(script: str) -> str:
        scripts.append(script)
        raise RuntimeError("Register-ScheduledTask denied")

    monkeypatch.setattr(lifecycle, "_run_powershell", denied)
    result = run_lifecycle_probe(
        apply=True, report_dir=tmp_path,
    )
    assert result["status"] == "blocked"
    assert result["cleanup_complete"] is False
    assert result["restart_authorized"] is False
    assert len(scripts) == 1
    assert "Register-ScheduledTask" in scripts[0]
    assert "Stop-ScheduledTask" not in scripts[0]
    assert Path(result["report_path"]).exists()


@pytest.mark.parametrize(
    ("supervisor_stopped", "child_alive", "advance", "expected"),
    [
        (True, True, True, "child_survived_scheduler_stop"),
        (True, False, False, "child_did_not_survive_scheduler_stop"),
        (False, True, True, "inconclusive_supervisor_still_running"),
    ],
)
def test_status_requires_supervisor_exit_and_new_child_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    supervisor_stopped: bool, child_alive: bool, advance: bool, expected: str,
) -> None:
    import artifex.windows_task_lifecycle_probe as lifecycle

    monkeypatch.setattr(lifecycle, "_require_windows", lambda: None)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda value: None)
    launched: list[str] = []
    def fake_shell(script: str) -> str:
        launched.append(script)
        if "ConvertTo-Json -Compress" in script:
            return '{"allow_hard_terminate":false,"multiple_instances":"IgnoreNew"}'
        return ""

    monkeypatch.setattr(lifecycle, "_run_powershell", fake_shell)
    monkeypatch.setattr(
        lifecycle, "_poll_ready",
        lambda folder, nonce, seconds: (
            {"nonce": nonce, "supervisor_pid": 10, "child_pid": 11},
            {"nonce": nonce, "pid": 11, "ticks": 1},
        ),
    )
    monkeypatch.setattr(
        lifecycle, "_identity",
        lambda pid, nonce: {
            "pid": pid, "started_utc": "2026-01-01T00:00:00Z",
            "executable": "mock.exe", "command_line": nonce,
        },
    )
    calls: list[int] = []
    def unchanged(before: dict[str, Any], nonce: str) -> bool:
        calls.append(int(before["pid"]))
        return (not supervisor_stopped) if before["pid"] == 10 else child_alive
    monkeypatch.setattr(lifecycle, "_unchanged_identity", unchanged)
    original_read = lifecycle._read_json
    def read(path: Path) -> dict[str, Any] | None:
        if path.name == "heartbeat.json":
            return {"nonce": path.parent.name, "pid": 11, "ticks": 2 if advance else 1}
        return original_read(path)
    monkeypatch.setattr(lifecycle, "_read_json", read)
    report = run_lifecycle_probe(apply=True, report_dir=tmp_path)
    assert report["status"] == expected
    assert report["child_survival_qualified"] is (
        expected == "child_survived_scheduler_stop"
    )
    assert report["production_qualified"] is False
    assert report["restart_authorized"] is False
    assert report["cleanup_complete"] is True
    assert calls == [10, 11]
    assert len([s for s in launched if "Register-ScheduledTask" in s]) == 1
    assert len([s for s in launched if "Stop-ScheduledTask" in s]) == 1
    assert len([s for s in launched if "Unregister-ScheduledTask" in s]) == 1
