"""One-call native field diagnostics do not elevate cached observations."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.qualification.native_field_preflight import (
    NativeFieldPreflight,
    compile_native_field_preflight,
    inspect_native_field_preflight,
)

PC_A = Path("C:/Artifex With Spaces/pc a.yaml")
PC_B = Path("D:/Models With Spaces/pc b.yaml")
NOW = datetime(2026, 10, 11, tzinfo=UTC)


def _reports(
    *, started: str = "2026-10-10T20:00:00Z",
    install_status: str = "ready_for_passive_monitoring",
    issue_status: str = "observations_correlated",
    handoff_status: str = "session_start_candidate",
    node: str = "gpu-b",
    owner_node: str = "gpu-b",
    owner_present: bool = True,
) -> tuple[Any, Any, Any]:
    installation = SimpleNamespace(
        status=install_status, pc_b_node_id=node,
        blockers=() if install_status == "ready_for_passive_monitoring"
                 else ("pc_b_observer_heartbeat_missing",),
    )
    states = ("observed",) * 4 if issue_status == "observations_correlated" else (
        "observed", "missing", "missing", "missing"
    )
    issue = SimpleNamespace(
        status=issue_status, node_id=node,
        checks=tuple(SimpleNamespace(name=name, state=state) for name, state in zip(
            ("isolated_launcher_and_live_owner", "two_time_separated_owner_samples",
             "natural_exit_survival_trace", "cross_evidence_listener_identity"),
            states, strict=True,
        )),
        current_comfyui_listener_pid=555,
        current_comfyui_started_utc="2026-10-11T05:00:00+09:00",
        current_pc_b_hostname="pc-b",
    )
    evidence = (
        SimpleNamespace(
            node_id=owner_node, live_listener_pid=555,
            live_listener_started_utc=started, live_host="PC-B",
        )
        if owner_present else None
    )
    handoff = SimpleNamespace(
        state=handoff_status,
        authenticated_owner_evidence_correlated=owner_present,
        overview=SimpleNamespace(
            readiness=SimpleNamespace(primary_node_id=owner_node),
            pc_b_owner_evidence=evidence,
        ),
        blockers=() if handoff_status == "session_start_candidate"
                 else ("deployment:workflow_requirements_missing_or_failed",),
        advisory_argv=(
            "uv", "run", "artifex", "qualify", "start",
            "--config", str(PC_A), "--json",
        ),
    )
    return installation, issue, handoff


def _compile(*, reports: tuple[Any, Any, Any] | None = None,
             errors: tuple[str, ...] = ()) -> NativeFieldPreflight:
    a, b, c = reports or _reports()
    return compile_native_field_preflight(
        controller_config=PC_A, renderer_config=PC_B,
        installation=a, issue93=b, handoff=c,
        probe_errors=errors, observed_utc=NOW,
    )


def test_all_three_gates_aligned_allow_only_advisory_session_start() -> None:
    report = _compile()
    assert report.status == "observational_gates_aligned"
    assert report.cross_probe_identity_consistent
    assert report.node_id == "gpu-b"
    assert report.next_safe_argv is not None
    assert report.next_safe_argv[:5] == (
        "uv", "run", "artifex", "qualify", "start",
    )
    assert report.recommended_field_plan is not None
    assert not report.commands_executed_by_report
    assert not report.service_mutated
    assert not report.gpu_jobs_submitted
    assert not report.natural_supervisor_exit_authenticated
    assert not report.native_comfyui_reattachment_qualified
    assert not report.fourteen_stage_qualification_verified
    assert not report.eight_hour_gpu_soak_verified
    assert not report.issue_93_closure_authorized
    assert not report.issue_40_closure_authorized
    assert not report.production_qualified


@pytest.mark.parametrize(("case", "expected"), [
    ("process_restarted", "identity_conflict"),
    ("different_node", "identity_conflict"),
    ("unavailable_issue", "needs_evidence"),
    ("missing_install", "needs_evidence"),
    ("blocked_handoff", "needs_evidence"),
    ("missing_cached_owner", "needs_evidence"),
    ("probe_error", "needs_evidence"),
])
def test_partial_unavailable_or_conflicting_sources_never_suggest_session_start(
    case: str, expected: str,
) -> None:
    a, b, c = _reports(
        started=("2026-10-10T20:00:01Z" if case == "process_restarted"
                 else "2026-10-10T20:00:00Z"),
        owner_node="foreign" if case == "different_node" else "gpu-b",
        install_status="pc_b_needs_setup" if case == "missing_install"
                       else "ready_for_passive_monitoring",
        handoff_status="blocked" if case == "blocked_handoff"
                       else "session_start_candidate",
        owner_present=case != "missing_cached_owner",
    )
    if case == "unavailable_issue":
        b = None
    report = compile_native_field_preflight(
        controller_config=PC_A, renderer_config=PC_B,
        installation=a, issue93=b, handoff=c,
        probe_errors=("installation:OSError",) if case == "probe_error" else (),
        observed_utc=NOW,
    )
    assert report.status == expected
    assert report.next_safe_argv is not None
    assert "start" not in report.next_safe_argv
    assert not report.production_qualified


@pytest.mark.parametrize("phase", ["installation", "issue93", "handoff"])
def test_exception_from_one_read_only_probe_preserves_other_results_and_no_secrets(
    phase: str,
) -> None:
    a, b, c = _reports()
    calls: list[str] = []

    def install(*args: Any, **kwargs: Any) -> Any:
        calls.append("installation")
        if phase == "installation":
            raise OSError("private network URL with token")
        return a

    def issue(*args: Any, **kwargs: Any) -> Any:
        calls.append("issue93")
        if phase == "issue93":
            raise ValueError("Bearer private secret")
        return b

    async def handoff(*args: Any, **kwargs: Any) -> Any:
        calls.append("handoff")
        assert kwargs["pc_b_owner_live"] is True
        assert kwargs["pc_b_survival_live"] is True
        if phase == "handoff":
            raise RuntimeError("LAN credentials leaked")
        return c

    report = asyncio.run(inspect_native_field_preflight(
        ArtifexSettings(), controller_config=PC_A, renderer_config=PC_B,
        installation_fn=install, issue93_fn=issue, handoff_fn=handoff,
    ))
    assert calls == ["installation", "issue93", "handoff"]
    assert report.status == "needs_evidence"
    assert report.probe_errors == (
        f"{phase}:" + {
            "installation": "OSError",
            "issue93": "ValueError",
            "handoff": "RuntimeError",
        }[phase],
    )
    assert "secret" not in report.model_dump_json().lower()
    assert "credentials" not in report.model_dump_json().lower()
    assert report.commands_executed_by_report is False
    assert report.gpu_jobs_submitted is False


def test_field_preflight_cli_writes_exclusive_only_and_keeps_paths_advisory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex import cli
    from artifex.qualification import native_field_preflight as module

    config = tmp_path / "pc a.yaml"
    config.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "results" / "preflight.json"
    invocations: list[Path] = []

    async def inspect(*args: Any, **kwargs: Any) -> NativeFieldPreflight:
        invocations.append(kwargs["controller_config"])
        assert kwargs["renderer_config"] == PC_B
        return _compile()

    monkeypatch.setattr(cli, "_settings", lambda _: ArtifexSettings())
    monkeypatch.setattr(module, "inspect_native_field_preflight", inspect)
    runner = CliRunner()
    command = [
        "qualify", "field-preflight", "--config", str(config),
        "--renderer-config", str(PC_B),
        "--output", str(output), "--json",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed["status"] == "observational_gates_aligned"
    assert parsed["production_qualified"] is False
    assert parsed["gpu_jobs_submitted"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == parsed
    assert invocations == [config]
    assert config.read_text(encoding="utf-8") == "{}\n"
    again = runner.invoke(app, command)
    assert again.exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8")) == parsed


def test_field_preflight_cli_non_green_is_nonzero_but_json_still_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex import cli
    from artifex.qualification import native_field_preflight as module

    config = tmp_path / "pc a.yaml"
    config.write_text("{}\n", encoding="utf-8")
    a, b, c = _reports(started="2026-10-10T20:00:01Z")
    report = _compile(reports=(a, b, c))

    async def inspect(*args: Any, **kwargs: Any) -> NativeFieldPreflight:
        return report

    monkeypatch.setattr(cli, "_settings", lambda _: ArtifexSettings())
    monkeypatch.setattr(module, "inspect_native_field_preflight", inspect)
    output = CliRunner().invoke(app, [
        "qualify", "field-preflight", "--config", str(config), "--json",
    ])
    assert output.exit_code == 1
    parsed = json.loads(output.stdout)
    assert parsed["status"] == "identity_conflict"
    assert parsed["next_safe_argv"] is not None
    assert parsed["production_qualified"] is False
