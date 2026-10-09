"""No-GPU Python launcher identity probe and exact native Windows evidence."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.render_node.native_launcher_probe import (
    LauncherEvidence,
    inspect_native_python_listener,
)


@pytest.mark.skipif(os.name != "nt", reason="Requires Windows CIM and Get-NetTCPConnection")
def test_native_venv_fixture_listener_records_exact_unmodified_process_chain() -> None:
    evidence = inspect_native_python_listener(timeout_seconds=16)
    assert evidence.status == "observed", evidence.model_dump()
    assert evidence.reason == "isolated_mock_python_listener_verified"
    assert evidence.original_launcher_verified
    assert evidence.listener_identity_stable
    assert evidence.tcp_owner_stable
    assert evidence.verified_runtime_provenance
    assert evidence.launcher_pid is not None
    assert evidence.listener_pid is not None
    assert evidence.ephemeral_port is not None
    assert evidence.launcher_identity is not None
    assert evidence.listener_identity is not None
    assert evidence.launcher_identity.pid == evidence.launcher_pid
    assert evidence.listener_identity.pid == evidence.listener_pid
    assert evidence.listener_identity.executable
    assert evidence.listener_identity.started_utc
    assert evidence.relation in {"direct", "venv_child"}
    if evidence.relation == "venv_child":
        assert evidence.listener_pid != evidence.launcher_pid
        assert evidence.listener_identity.parent_pid == evidence.launcher_pid
    else:
        assert evidence.listener_pid == evidence.launcher_pid
    assert evidence.actual_comfyui_inspected is False
    assert evidence.production_child_survival_qualified is False
    assert evidence.real_machine_gpu_qualified is False
    assert evidence.renderer_restart_authorized is False
    assert evidence.gpu_jobs_submitted is False
    assert evidence.test_listener_only


@pytest.mark.skipif(os.name != "nt", reason="Needs native Windows and virtualenv")
def test_native_cli_probe_writes_only_new_local_report(tmp_path: Path) -> None:
    target = tmp_path / "launcher-evidence.json"
    args = ["render-node", "launcher-probe", "--output", str(target), "--json"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed["status"] == "observed"
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert LauncherEvidence.model_validate(stored).status == "observed"
    assert parsed == stored
    assert stored["production_child_survival_qualified"] is False
    assert stored["renderer_restart_authorized"] is False
    # No report or existing file may be clobbered by a second invocation.
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 1
    assert json.loads(target.read_text(encoding="utf-8")) == stored


@pytest.mark.skipif(os.name != "nt", reason="Native Python executable verification")
def test_probe_rejects_non_python_executable_without_spawning(tmp_path: Path) -> None:
    candidate = tmp_path / "suspicious.exe"
    candidate.write_bytes(b"not a Python executable")
    with pytest.raises(ValueError, match="python.exe"):
        inspect_native_python_listener(python=candidate)


def test_nonwindows_reports_unsupported_without_spawning() -> None:
    if os.name == "nt":
        pytest.skip("Windows performs real fixture in native tests above")
    evidence = inspect_native_python_listener()
    assert evidence.status == "unsupported"
    assert evidence.launcher_pid is None
    assert evidence.listener_pid is None
    assert not evidence.verified_runtime_provenance
    assert evidence.production_child_survival_qualified is False
    assert evidence.gpu_jobs_submitted is False
    assert sys.executable  # no mock child was started
