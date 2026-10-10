"""Native PC-B local field diagnostics: bounded, no production process changes."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.render_node.installation_audit import RendererInstallationAudit
from artifex.render_node.local_field_preflight import (
    LocalRendererFieldPreflight,
    compile_local_renderer_field_preflight,
    inspect_local_renderer_field_preflight,
)

NOW = datetime(2026, 10, 11, tzinfo=UTC)
CONFIG = Path("D:/Artifex PC B/render node.yaml")


def _settings() -> ArtifexSettings:
    s = ArtifexSettings()
    s.render_agent.node_id = "gpu-b"
    return s


def _install(
    *, observer: str = "running",
    heartbeat: str = "fresh",
    ready: bool = True,
    hostname: str = "PC-B",
) -> RendererInstallationAudit:
    return RendererInstallationAudit.model_validate({
        "node_id": "gpu-b", "captured_utc": NOW,
        "hostname": hostname, "native_windows": True,
        "managed_renderer_configured": True,
        "renderer_task": {
            "role": "renderer", "status": "running", "registered": True,
            "managed": True, "configuration_verified": True,
            "reason_code": "verified_running",
        },
        "survival_observer_task": {
            "role": "survival-observer", "status": observer,
            "registered": observer == "running",
            "managed": True, "configuration_verified": observer == "running",
            "reason_code": "verified_running" if observer == "running"
                           else "task_not_registered",
        },
        "evidence_spool": "empty", "observer_heartbeat": heartbeat,
        "safe_for_passive_observation": ready,
    })


def _owner(
    *, started: str = "2026-10-10T21:00:00+09:00",
    pid: int = 4700,
    host: str = "pc-b",
    fixture: str = "pass",
) -> dict[str, Any]:
    return {
        "status": "observed_independently", "host": host,
        "checks": {
            "disposable_launcher_fixture": {"status": fixture, "reason": "fixture"},
            "live_comfyui_owner": {"status": "pass", "reason": "observed"},
        },
        "owner_audit": {
            "status": "observed_stable", "actual_listener_pid": pid,
            "actual_process_started_utc": started,
        },
        # May exist in the original detailed report, but must not be emitted
        # by the shareable reduced-scope PC-B field preflight.
        "configured_comfyui_python": "D:/private user/GPU/python.exe",
        "private_token": "NEVER_COPY_THIS",
    }


def _report(
    *, install: RendererInstallationAudit | None = None,
    owner: dict[str, Any] | None = None,
    native: bool = True,
    errors: tuple[str, ...] = (),
) -> LocalRendererFieldPreflight:
    return compile_local_renderer_field_preflight(
        _settings(), config=CONFIG,
        installation=install if install is not None else _install(),
        owner_readiness=owner if owner is not None else _owner(),
        native_windows=native, probe_errors=errors, observed_utc=NOW,
    )


def test_pc_b_local_combined_ready_does_not_authorize_restart_or_gpu() -> None:
    report = _report()
    assert report.status == "passive_observation_ready"
    assert report.node_id == "gpu-b"
    assert report.listener_pid == 4700
    assert report.listener_started_utc == "2026-10-10T12:00:00+00:00"
    assert report.next_host == "pc_a"
    assert report.next_safe_argv is None
    assert report.disposable_no_gpu_probe_attempted is True
    assert report.survival_spool_status == "empty"
    assert not report.existing_comfyui_started
    assert not report.existing_comfyui_stopped
    assert not report.scheduler_actions_executed
    assert not report.production_gpu_jobs_submitted
    assert not report.historical_supervisor_exit_authenticated
    assert not report.production_reattachment_qualified
    assert not report.fourteen_stage_qualification_verified
    assert not report.eight_hour_soak_verified
    assert not report.issue_93_closure_authorized
    assert not report.issue_40_closure_authorized
    assert not report.production_qualified
    assert "NEVER_COPY_THIS" not in report.model_dump_json()
    assert "python.exe" not in report.model_dump_json()


@pytest.mark.parametrize("case", (
    "observer_missing", "heartbeat_stale", "fixture_failure",
    "pid_reused", "naive_timestamp", "host_mismatch", "unsafe_installation",
    "probe_failure",
))
def test_pc_b_blocks_all_missing_or_conflicting_local_evidence(case: str) -> None:
    install = _install(
        observer="missing" if case == "observer_missing" else "running",
        heartbeat="stale" if case == "heartbeat_stale" else "fresh",
        ready=case != "unsafe_installation",
    )
    owner = _owner(
        fixture="fail" if case == "fixture_failure" else "pass",
        started="2026-10-11T00:01:00Z" if case == "pid_reused" else
                "2026-10-10T12:00:00" if case == "naive_timestamp" else
                "2026-10-10T21:00:00+09:00",
        host="another-host" if case == "host_mismatch" else "pc-b",
    )
    report = _report(
        install=install, owner=owner,
        errors=("owner_readiness:OSError",) if case == "probe_failure" else (),
    )
    assert report.status == "needs_evidence"
    assert report.blocked_reasons
    assert report.listener_pid is None
    assert report.next_host == "pc_b"
    assert report.next_safe_argv is not None
    assert report.next_safe_argv[:5] == (
        "uv", "run", "artifex", "startup", "observer-enable",
    )
    assert "--apply" not in report.next_safe_argv
    assert not report.production_qualified


@pytest.mark.parametrize("problem", [
    "contradictory_renderer_state",
    "stale_observation",
    "future_clock",
    "naive_observation",
])
def test_pc_b_individual_scheduler_truth_and_timestamp_override_summary_pass(
    problem: str,
) -> None:
    original = _install()
    if problem == "contradictory_renderer_state":
        installation = original.model_copy(update={
            "renderer_task": original.renderer_task.model_copy(update={
                "status": "missing",
            }),
        })
    elif problem == "stale_observation":
        installation = original.model_copy(update={
            "captured_utc": NOW - timedelta(minutes=5),
        })
    elif problem == "future_clock":
        installation = original.model_copy(update={
            "captured_utc": NOW + timedelta(minutes=5),
        })
    else:
        installation = original.model_copy(update={
            "captured_utc": NOW.replace(tzinfo=None),
        })
    report = _report(install=installation)
    assert report.status == "needs_evidence"
    assert report.listener_pid is None
    assert report.next_host == "pc_b"
    assert not report.production_qualified


def test_unsupported_platform_does_not_launch_even_disposable_fixture() -> None:
    calls: list[str] = []

    def install(*args: Any, **kwargs: Any) -> RendererInstallationAudit:
        calls.append("install")
        assert kwargs["native_windows"] is False
        return _install()

    def owner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Never start a native Windows fixture on other OS")

    report = inspect_local_renderer_field_preflight(
        _settings(), config=CONFIG, native_windows=False,
        installation_fn=install, owner_fn=owner,
    )
    assert calls == ["install"]
    assert report.status == "unsupported"
    assert not report.disposable_no_gpu_probe_attempted
    assert report.next_safe_argv is None
    assert not report.production_qualified


@pytest.mark.parametrize("failure", ("installation", "owner_readiness"))
def test_independent_probe_errors_are_redacted_and_other_checks_run(
    failure: str,
) -> None:
    calls: list[str] = []

    def install(*args: Any, **kwargs: Any) -> RendererInstallationAudit:
        calls.append("installation")
        if failure == "installation":
            raise OSError("PRIVATE PATH AND NETWORK TOKEN")
        return _install()

    def owner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("owner_readiness")
        if failure == "owner_readiness":
            raise RuntimeError("PRIVATE PATH AND NETWORK TOKEN")
        return _owner()

    result = inspect_local_renderer_field_preflight(
        _settings(), config=CONFIG, native_windows=True,
        installation_fn=install, owner_fn=owner,
    )
    assert calls == ["installation", "owner_readiness"]
    assert result.status == "needs_evidence"
    assert len(result.probe_errors) == 1
    assert result.probe_errors[0].startswith(failure + ":")
    assert "PRIVATE" not in result.model_dump_json()
    assert not result.production_qualified


def test_pc_b_cli_custom_path_exclusive_json_and_nonzero_when_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex import cli
    from artifex.render_node import local_field_preflight as module

    yaml = tmp_path / "PC B custom.yaml"
    yaml.write_text("{}\n", encoding="utf-8")
    path = tmp_path / "field" / "pc-b.json"
    calls: list[Path] = []
    report = _report()

    def inspect(*args: Any, **kwargs: Any) -> LocalRendererFieldPreflight:
        calls.append(kwargs["config"])
        return report

    monkeypatch.setattr(cli, "_settings", lambda _: _settings())
    monkeypatch.setattr(module, "inspect_local_renderer_field_preflight", inspect)
    command = [
        "render-node", "field-preflight", "--config", str(yaml),
        "--output", str(path), "--json",
    ]
    runner = CliRunner()
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert calls == [yaml]
    assert yaml.read_text(encoding="utf-8") == "{}\n"
    again = runner.invoke(app, command)
    assert again.exit_code == 1
    assert json.loads(path.read_text(encoding="utf-8")) == payload

    broken = _report(install=_install(observer="missing", ready=False))
    monkeypatch.setattr(module, "inspect_local_renderer_field_preflight",
                        lambda *a, **kw: broken)
    blocked = runner.invoke(app, [
        "render-node", "field-preflight", "--config", str(yaml), "--json",
    ])
    assert blocked.exit_code == 1
    assert json.loads(blocked.stdout)["status"] == "needs_evidence"
