"""Safe per-machine onboarding advice from a read-only paired installation audit."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.qualification.pair_install_remediation import (
    compile_pair_install_remediation,
)
from artifex.qualification.pair_installation import TwoPCInstallationAudit
from artifex.render_node.installation_audit import (
    RemoteRendererInstallationAudit,
    RendererInstallationAudit,
    TaskInstallationCheck,
)

NOW = datetime(2026, 10, 10, tzinfo=UTC)
A_PATH = Path("D:/Artifex/pc-a/local custom.yaml")
B_PATH = Path("E:/AI/Artifex/pc-b config.yaml")


def _task(role: str, status: str = "running") -> TaskInstallationCheck:
    verified = status in {"running", "registered_not_running"}
    return TaskInstallationCheck.model_validate({
        "role": role,
        "status": status,
        "registered": status != "missing",
        "managed": status not in {"missing", "unsafe"},
        "configuration_verified": verified,
        "state": "Running" if status == "running" else "Ready",
        "reason_code": "test_only",
    })


def _audit(
    *,
    local: str = "running",
    renderer: str = "running",
    observer: str = "missing",
    managed: bool = True,
    windows: bool = True,
    spool: str = "missing",
    remote: bool = True,
    status: str = "pc_b_needs_setup",
    blockers: tuple[str, ...] | None = None,
    heartbeat: str | None = None,
) -> TwoPCInstallationAudit:
    pc_b = None
    if remote:
        pc_b = RemoteRendererInstallationAudit(
            node_id="gpu-b",
            audit=RendererInstallationAudit.model_validate({
                "node_id": "gpu-b",
                "captured_utc": NOW,
                "hostname": "pc-b",
                "native_windows": windows,
                "managed_renderer_configured": managed,
                "renderer_task": _task("renderer", renderer),
                "survival_observer_task": _task("survival-observer", observer),
                "evidence_spool": spool,
                "observer_heartbeat": heartbeat or (
                    "fresh" if observer == "running" else "missing"
                ),
                "safe_for_passive_observation": (
                    windows and managed and renderer == "running"
                    and observer == "running"
                    and heartbeat in {None, "fresh"}
                ),
            }),
        )
    if blockers is None:
        blockers = (
            *(() if local == "running" else ("pc_a_controller_task_" + local,)),
            *(() if renderer == "running" or not remote else ("pc_b_renderer_" + renderer,)),
            *(() if observer == "running" or not remote else ("pc_b_survival_observer_" + observer,)),
        )
    return TwoPCInstallationAudit.model_validate({
        "checked_utc": NOW,
        "status": status,
        "pc_a_controller_task": _task("controller", local),
        "pc_b_node_id": (
            None if "pc_b_primary_node_unconfigured" in blockers else "gpu-b"
        ),
        "pc_b": pc_b,
        "blockers": blockers,
        "next_actions": [],
    })


def _plan(audit: TwoPCInstallationAudit):
    return compile_pair_install_remediation(
        audit,
        controller_config=A_PATH, renderer_config=B_PATH,
    )


def test_missing_observer_only_offers_explicit_observer_approval() -> None:
    result = _plan(_audit())
    observer = next(x for x in result.steps if x.step_id == "pc-b-survival-observer")
    assert observer.kind == "optional_explicit_apply"
    assert observer.role == "pc_b"
    assert observer.read_only_argv == (
        "uv", "run", "artifex", "startup", "observer-enable",
        "--config", str(B_PATH), "--json",
    )
    assert observer.operator_approved_apply_argv == (
        "uv", "run", "artifex", "startup", "observer-enable", "--apply",
        "--config", str(B_PATH), "--json",
    )
    assert result.optional_operator_approval_count == 1
    assert result.read_only_only is False
    assert result.commands_executed is False
    assert result.renderer_task_modified is False
    assert result.comfyui_process_modified is False
    assert result.production_qualified is False
    assert all(
        "--apply" not in (step.read_only_argv or ())
        for step in result.steps
    )
    assert all(
        step.operator_approved_apply_argv is None
        or "observer-enable" in step.operator_approved_apply_argv
        for step in result.steps
    )


@pytest.mark.parametrize(("observer", "renderer", "managed", "expected"), [
    ("registered_not_running", "running", True, True),
    ("missing", "registered_not_running", True, False),
    ("missing", "unsafe", True, False),
    ("missing", "running", False, False),
    ("unsafe", "running", True, False),
    ("unavailable", "running", True, False),
])
def test_only_verified_safe_observer_can_be_operator_approved(
    observer: str, renderer: str, managed: bool, expected: bool,
) -> None:
    result = _plan(_audit(
        observer=observer, renderer=renderer, managed=managed,
    ))
    step = next(x for x in result.steps if x.step_id == "pc-b-survival-observer")
    assert (step.operator_approved_apply_argv is not None) is expected
    assert not result.renderer_task_modified
    assert not result.comfyui_process_modified


def test_foreign_tasks_and_unsafe_renderer_offer_read_only_review_only() -> None:
    result = _plan(_audit(
        local="unsafe", renderer="unsafe", observer="unsafe",
    ))
    assert result.read_only_only
    assert result.optional_operator_approval_count == 0
    assert {x.step_id for x in result.steps} >= {
        "pc-a-task", "pc-a-task-command", "pc-b-renderer-task",
        "pc-b-survival-observer",
    }
    assert all(s.operator_approved_apply_argv is None for s in result.steps)
    assert all(s.services_modified is False for s in result.steps)
    assert all(s.gpu_jobs_submitted is False for s in result.steps)


@pytest.mark.parametrize(("status", "blockers", "wanted"), [
    ("pc_b_unconfigured", ("pc_b_primary_node_unconfigured",), "pc-a-render-node"),
    (
        "pc_b_unconfigured",
        ("pc_b_attestation_or_bearer_unconfigured",), "pc-a-auth",
    ),
    (
        "pc_b_unreachable",
        ("pc_b_authenticated_installation_unavailable",), "pc-b-connection",
    ),
])
def test_offline_or_unconfigured_pc_b_never_leads_to_task_mutations(
    status: str, blockers: tuple[str, ...], wanted: str,
) -> None:
    a = _audit(remote=False, status=status, blockers=blockers)
    result = _plan(a)
    assert result.read_only_only
    assert wanted in {step.step_id for step in result.steps}
    assert all(x.operator_approved_apply_argv is None for x in result.steps)
    assert all(x.read_only_argv is None or "--apply" not in x.read_only_argv
               for x in result.steps)


def test_custom_config_paths_are_advisory_and_never_echo_secrets() -> None:
    result = _plan(_audit())
    serialized = result.model_dump_json()
    parsed = json.loads(serialized)
    assert any(
        str(B_PATH) in step["read_only_argv"]
        for step in parsed["steps"]
        if step["read_only_argv"] is not None
    )
    assert "ARTIFEX_RENDER_NODE_TOKEN=" not in serialized
    assert "http://127.0.0.1:8190" not in serialized
    assert "powershell.exe" not in serialized
    assert '"commands_executed":false' in serialized


def test_evidence_spool_empty_is_only_waiting_not_a_pass() -> None:
    audit = _audit(observer="running", status="ready_for_passive_monitoring",
                   blockers=(), spool="empty")
    report = _plan(audit)
    assert report.read_only_only
    assert report.optional_operator_approval_count == 0
    assert "pc-b-no-event-yet" in {s.step_id for s in report.steps}
    assert report.actual_survival_observed is False
    assert report.production_qualified is False


@pytest.mark.parametrize("heartbeat", ["blocked", "unsupported"])
def test_failed_recent_heartbeat_only_offers_read_only_diagnostics(
    heartbeat: str,
) -> None:
    blocker = "pc_b_observer_heartbeat_" + heartbeat
    audit = _audit(
        observer="running", heartbeat=heartbeat, status="pc_b_needs_setup",
        blockers=(blocker, "pc_b_passive_observer_preconditions_unverified"),
    )
    report = _plan(audit)
    step = next(s for s in report.steps if s.step_id == "pc-b-observer-liveness")
    assert step.matching_blockers == (blocker,)
    assert step.read_only_argv == (
        "uv", "run", "artifex", "startup", "status",
        "--role", "survival-observer", "--json",
    )
    assert step.operator_approved_apply_argv is None
    assert report.read_only_only
    assert not report.renderer_task_modified
    assert not report.comfyui_process_modified
    assert not report.production_qualified


def test_fully_registered_tasks_only_yield_observation_guidance() -> None:
    report = _plan(_audit(
        observer="running", status="ready_for_passive_monitoring",
        blockers=(), spool="has_evidence",
    ))
    assert [s.step_id for s in report.steps] == ["monitor-only"]
    assert report.production_qualified is False


def test_cli_surfaces_typed_plan_without_executing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex import cli
    from artifex.qualification import pair_installation

    cfg = tmp_path / "pc-a.yaml"
    cfg.write_text("{}")
    observed: list[Path] = []

    def fake_inspect(settings: ArtifexSettings, *, controller_config: Path):
        observed.append(controller_config)
        return _audit()

    monkeypatch.setattr(cli, "_settings", lambda _: ArtifexSettings())
    monkeypatch.setattr(pair_installation, "inspect_two_pc_installation", fake_inspect)
    result = CliRunner().invoke(app, [
        "qualify", "pair-install-audit", "--config", str(cfg),
        "--renderer-config", str(B_PATH), "--json",
    ])
    assert result.exit_code == 1, result.output
    data = json.loads(result.stdout)
    assert data["status"] == "pc_b_needs_setup"
    assert data["production_qualified"] is False
    assert data["remediation_plan"]["optional_operator_approval_count"] == 1
    assert data["remediation_plan"]["production_qualified"] is False
    assert observed == [cfg]


def test_all_code_paths_keep_apply_only_in_operator_approval_field() -> None:
    for l in ("running", "missing", "unsafe"):
        for r in ("running", "missing", "unsafe"):
            for o in ("running", "missing", "unsafe"):
                audit = _audit(local=l, renderer=r, observer=o)
                report = _plan(audit)
                assert all(
                    "--apply" not in (step.read_only_argv or ())
                    for step in report.steps
                )
                assert all(
                    step.operator_approved_apply_argv is None or
                    step.step_id == "pc-b-survival-observer"
                    for step in report.steps
                )
