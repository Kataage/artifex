"""PC-A survival-trace review: copied evidence is never production proof."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.survival_review import review_pc_b_survival_evidence
from artifex.render_node.models import RemoteRendererOwnerAudit, RenderNodeAttestation
from artifex.render_node.supervisor_survival import SurvivalSample, assess_survival

NOW = datetime(2026, 10, 11, tzinfo=UTC)
_START = NOW - timedelta(hours=1)
_OWNER_START = "2026-10-10T19:00:00Z"
_CHECKS = (
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
)


def _settings() -> ArtifexSettings:
    s = ArtifexSettings()
    s.render_nodes.primary = "gpu-b"
    s.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://127.0.0.1:8188",
        attestation_url="http://127.0.0.1:8190",
    )
    return s


def _sample(sec: int, *, task_state: str = "Running") -> SurvivalSample:
    running = task_state == "Running"
    return SurvivalSample(
        observed_utc=_START + timedelta(seconds=sec),
        elapsed_seconds=float(sec),
        host="pc-b", node_id="gpu-b", state="verified",
        task_state=task_state, scheduler_policy_verified=True,
        supervisor_identities=((100, "2026-10-10T20:00:00Z"),) if running else (),
        original_supervisor_pids_absent=not running,
        owner_audit_state="observed_stable" if running else "inconclusive",
        owner_checks={
            k: "unknown" if k == "scheduler_running" and not running else "pass"
            for k in _CHECKS
        },
        owner_process_verified=True,
        actual_listener_pid=400,
        actual_listener_started_utc=_OWNER_START,
        launcher_pid=399,
        receipt_schema=2,
    )


def _trace(path: Path) -> None:
    trace = assess_survival((
        _sample(0), _sample(10),
        _sample(20, task_state="Ready"),
        _sample(30, task_state="Ready"),
    ), min_separation_seconds=10)
    assert trace.status == "observed_after_supervisor_absence"
    path.write_text(trace.model_dump_json(), encoding="utf-8")


def _probes(*, host: str = "pc-b", pid: int = 400,
            started: str = _OWNER_START, at: datetime = NOW):
    def owner(node: str, cfg: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        assert node == "gpu-b"
        return RemoteRendererOwnerAudit.model_validate({
            "node_id": node,
            "audit": {
                "schema_version": 1,
                "captured_utc": at,
                "status": "observed_stable",
                "process_observation_verified": True,
                "checks": {
                    k: {"status": "pass", "reason": "native owner audit"}
                    for k in _CHECKS
                },
                "actual_listener_pid": pid,
                "actual_process_started_utc": started,
                "launcher_pid": 399,
                "receipt_schema": 2,
                "scheduler_state": "Running",
                "restart_authorized": False,
                "child_survival_qualified": False,
                "production_qualified": False,
                "mutated_services": False,
            },
        })

    def attest(node: str, cfg: RenderNodeConfig, *,
               fresh: bool, timeout_seconds: float) -> RenderNodeAttestation:
        assert node == "gpu-b" and fresh
        return RenderNodeAttestation(
            node_id=node, hostname=host, created_at=at,
            os={"system": "Windows"},
            comfyui_base_url="http://127.0.0.1:8188",
        )
    return owner, attest


def _review(path: Path, *, host: str = "pc-b", pid: int = 400,
            started: str = _OWNER_START, at: datetime = NOW):
    owner, attest = _probes(host=host, pid=pid, started=started, at=at)
    return review_pc_b_survival_evidence(
        _settings(), path, now=NOW,
        owner_probe=owner, attestation_probe=attest,
    )


@pytest.mark.parametrize("started,expected", (
    ("2026-10-11T04:00:00+09:00", "replayed_and_live_identity_matched"),
    ("2026-10-11T04:00:01+09:00", "replayed_historical_identity"),
))
def test_saved_utc_creation_matches_live_japan_offset_only_for_same_instant(
    tmp_path: Path, started: str, expected: str,
) -> None:
    path = tmp_path / "survival.json"
    _trace(path)
    report = _review(path, started=started)
    assert report.status == expected
    assert report.trace_replayed
    assert report.remote_bearer_checked is False
    assert report.independent_supervisor_survival_qualified is False
    assert report.issue_93_closure_authorized is False
    assert report.production_qualified is False


def test_trace_replayed_against_current_owner_but_never_qualifies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "survival.json"
    _trace(path)
    report = _review(path)
    assert report.status == "replayed_and_live_identity_matched"
    assert report.trace_replayed
    assert report.evidence_sha256 and len(report.evidence_sha256) == 64
    assert report.recorded_host == report.live_host == "pc-b"
    assert report.historical_listener_pid == report.current_listener_pid == 400
    assert report.current_listener_started_utc == _OWNER_START
    assert report.sample_count == 4
    assert report.source_authenticated is False
    assert report.historical_event_authenticated is False
    assert report.independent_supervisor_survival_qualified is False
    assert report.issue_93_closure_authorized is False
    assert report.issue_40_closure_authorized is False
    assert report.stage_pass_registered is False
    assert report.renderer_restart_authorized is False
    assert report.production_qualified is False


@pytest.mark.parametrize("problem,expected", (
    ("malformed", "blocked"),
    ("missing", "blocked"),
    ("oversized", "blocked"),
    ("tampered_verdict", "blocked"),
    ("fake_success", "blocked"),
    ("wrong_node", "blocked"),
    ("future_clock", "blocked"),
    ("false_qualification", "blocked"),
    ("missing_sample", "blocked"),
    ("altered_pid", "blocked"),
    ("wrong_host", "mismatch"),
    ("restarted_process", "replayed_historical_identity"),
    ("stale_live", "blocked"),
))
def test_untrusted_or_inconsistent_trace_never_promotes(
    tmp_path: Path, problem: str, expected: str,
) -> None:
    path = tmp_path / "survival.json"
    _trace(path)
    doc = json.loads(path.read_text())
    if problem == "tampered_verdict":
        doc["reason"] = "invented reason"
    elif problem == "fake_success":
        doc["samples"][1]["supervisor_identities"] = []
    elif problem == "wrong_node":
        doc["node_id"] = "other-b"
    elif problem == "future_clock":
        doc["samples"][3]["observed_utc"] = (NOW + timedelta(hours=2)).isoformat()
    elif problem == "false_qualification":
        doc["production_qualified"] = True
    elif problem == "missing_sample":
        doc["samples"] = doc["samples"][:2]
    elif problem == "altered_pid":
        doc["samples"][2]["actual_listener_pid"] = 999
    if problem in {
        "tampered_verdict", "fake_success", "wrong_node", "future_clock",
        "false_qualification", "missing_sample", "altered_pid",
    }:
        path.write_text(json.dumps(doc))
    elif problem == "malformed":
        path.write_text("not a JSON report")
    elif problem == "missing":
        path.unlink()
    elif problem == "oversized":
        path.write_text("x" * (12 * 1024 * 1024 + 1))
    report = _review(
        path,
        host="another-b" if problem == "wrong_host" else "pc-b",
        pid=401 if problem == "restarted_process" else 400,
        at=NOW - timedelta(hours=1) if problem == "stale_live" else NOW,
    )
    assert report.status == expected
    assert not report.independent_supervisor_survival_qualified
    assert not report.production_qualified


@pytest.mark.parametrize("failure", (
    "blocked_status_only",
    "inconclusive_status_only",
    "missing_native_windows",
    "missing_protected_configuration",
    "scheduler_policy_failed",
    "scheduler_not_running",
    "receipt_failed",
    "missing_snapshot_consistency",
    "unverified_process",
    "naive_creation_time",
    "malformed_creation_time",
))
def test_survival_review_rejects_partially_good_current_owner(
    tmp_path: Path, failure: str,
) -> None:
    """Five passing process checks cannot redeem an unsafe owner audit."""
    path = tmp_path / "survival.json"
    _trace(path)
    real_probe, attest = _probes()

    def degraded_owner(node: str, cfg: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        response = real_probe(node, cfg)
        audit = response.audit
        updates: dict[str, object] = {}
        if failure in {"blocked_status_only", "inconclusive_status_only"}:
            updates["status"] = (
                "blocked" if failure == "blocked_status_only" else "inconclusive"
            )
        elif failure == "unverified_process":
            updates["process_observation_verified"] = False
        elif failure in {"naive_creation_time", "malformed_creation_time"}:
            updates["actual_process_started_utc"] = (
                "2026-10-10T19:00:00" if failure == "naive_creation_time"
                else "not a datetime"
            )
        else:
            checks = dict(audit.checks)
            key = {
                "missing_native_windows": "native_windows",
                "missing_protected_configuration": "protected_configuration",
                "scheduler_policy_failed": "scheduler_policy",
                "scheduler_not_running": "scheduler_running",
                "receipt_failed": "receipt",
                "missing_snapshot_consistency": "snapshot_consistency",
            }[failure]
            if failure in {
                "missing_native_windows", "missing_protected_configuration",
                "missing_snapshot_consistency",
            }:
                checks.pop(key)
            else:
                checks[key] = checks[key].model_copy(
                    update={"status": (
                        "unknown" if failure == "scheduler_not_running" else "fail"
                    )}
                )
            updates["checks"] = checks
        return response.model_copy(update={"audit": audit.model_copy(update=updates)})

    report = review_pc_b_survival_evidence(
        _settings(), path, now=NOW,
        owner_probe=degraded_owner, attestation_probe=attest,
    )
    assert report.trace_replayed
    assert report.status == "blocked"
    assert report.reason == "live_pc_b_identity_cannot_be_authenticated"
    assert not report.independent_supervisor_survival_qualified
    assert not report.historical_event_authenticated
    assert not report.issue_93_closure_authorized
    assert not report.production_qualified


def test_survival_review_requires_live_observation_after_trace_end(
    tmp_path: Path,
) -> None:
    path = tmp_path / "survival.json"
    _trace(path)
    # The trace finishes 30 seconds after START. A fresh authenticated
    # owner snapshot at 15 seconds cannot corroborate an event at 30 seconds,
    # even when clock-skew allowances would accept both timestamps.
    premature = _START + timedelta(seconds=15)
    owner, attest = _probes(at=premature)
    report = review_pc_b_survival_evidence(
        _settings(), path, now=premature,
        owner_probe=owner, attestation_probe=attest,
    )
    assert report.trace_replayed
    assert report.status == "blocked"
    assert report.reason == "live_pc_b_identity_cannot_be_authenticated"
    assert not report.historical_event_authenticated
    assert not report.production_qualified


def test_live_unavailable_keeps_replayed_trace_untrusted(tmp_path: Path) -> None:
    path = tmp_path / "survival.json"
    _trace(path)
    calls: list[str] = []

    def unreachable(*args, **kwargs):
        calls.append("owner")
        raise OSError("PC-B attestation offline")

    def should_not_call(*args, **kwargs):
        raise AssertionError("Must not fetch after owner failure")

    report = review_pc_b_survival_evidence(
        _settings(), path, now=NOW,
        owner_probe=unreachable, attestation_probe=should_not_call,
    )
    assert calls == ["owner"]
    assert report.status == "replayed_live_unavailable"
    assert report.trace_replayed
    assert not report.historical_event_authenticated


def test_invalid_file_never_causes_remote_fetch(tmp_path: Path) -> None:
    def reject(*args, **kwargs):
        raise AssertionError("invalid data must be rejected before fetching")

    r = review_pc_b_survival_evidence(
        _settings(), tmp_path / "missing.json", now=NOW,
        owner_probe=reject, attestation_probe=reject,
    )
    assert r.status == "blocked"


def test_overview_and_handoff_cli_forward_optional_trace_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from artifex import cli
    from artifex.qualification import handoff, overview

    cfg = tmp_path / "controller.yaml"
    cfg.write_text("{}\n", encoding="utf-8")
    survival = tmp_path / "pc-b.json"
    captured: list[tuple[str, object]] = []
    monkeypatch.setattr(cli, "_settings", lambda _: _settings())

    class Overview:
        environment_ready = False
        def model_dump(self, *, mode: str) -> dict[str, object]:
            return {"production_qualified": False, "environment_ready": False}

    def overview_fn(*args, **kwargs):
        captured.append(("overview", kwargs.get("pc_b_survival_report")))
        return Overview()

    async def handoff_fn(*args, **kwargs):
        captured.append(("handoff", kwargs.get("pc_b_survival_report")))
        class Handoff:
            state = "blocked"
            def model_dump(self, *, mode: str) -> dict[str, object]:
                return {"state": "blocked", "production_qualified": False}
        return Handoff()

    monkeypatch.setattr(overview, "compile_qualification_overview", overview_fn)
    monkeypatch.setattr(handoff, "inspect_qualification_handoff", handoff_fn)
    for command in ("overview", "handoff"):
        result = CliRunner().invoke(app, [
            "qualify", command, "--config", str(cfg),
            "--pc-b-survival-report", str(survival), "--json",
        ])
        assert result.exit_code == (0 if command == "overview" else 1), result.output
    assert captured == [("overview", survival), ("handoff", survival)]



def test_authenticated_remote_trace_is_replayed_without_manual_copy(
    tmp_path: Path,
) -> None:
    import hashlib

    from artifex.render_node.models import RemoteSurvivalTrace

    path = tmp_path / "local-pc-b.json"
    _trace(path)
    original = path.read_text(encoding="utf-8")
    path.unlink()
    owner, attestation = _probes()
    requested: list[str] = []

    def get_saved(node: str, cfg: RenderNodeConfig) -> RemoteSurvivalTrace:
        requested.append(node)
        return RemoteSurvivalTrace(
            node_id=node, content=original,
            sha256=hashlib.sha256(original.encode()).hexdigest(),
        )

    reviewed = review_pc_b_survival_evidence(
        _settings(), None, remote=True, now=NOW,
        remote_probe=get_saved, owner_probe=owner,
        attestation_probe=attestation,
    )
    assert requested == ["gpu-b"]
    assert reviewed.status == "replayed_and_live_identity_matched"
    assert reviewed.source_mode == "bearer_remote"
    assert reviewed.remote_bearer_checked is True
    assert reviewed.source_authenticated is False
    assert reviewed.historical_event_authenticated is False
    assert reviewed.independent_supervisor_survival_qualified is False
    assert reviewed.production_qualified is False


@pytest.mark.parametrize("problem,expected", (
    ("unreachable", "remote_unavailable"),
    ("wrong_node", "blocked"),
    ("bad_sha", "blocked"),
    ("future_sample", "blocked"),
    ("historical_pid_changed", "replayed_historical_identity"),
))
def test_remote_trace_unavailable_or_corrupt_never_qualifies(
    tmp_path: Path, problem: str, expected: str,
) -> None:
    import hashlib

    from artifex.render_node.models import RemoteSurvivalTrace

    path = tmp_path / "local-pc-b.json"
    _trace(path)
    original = path.read_text(encoding="utf-8")
    owner, attestation = _probes(pid=401 if problem == "historical_pid_changed" else 400)

    def get_saved(node: str, cfg: RenderNodeConfig) -> RemoteSurvivalTrace:
        if problem == "unreachable":
            raise OSError("PC-B not up yet")
        content = original
        if problem == "future_sample":
            doc = json.loads(content)
            doc["samples"][-1]["observed_utc"] = (
                NOW + timedelta(hours=3)
            ).isoformat()
            content = json.dumps(doc)
        return RemoteSurvivalTrace(
            node_id="foreign" if problem == "wrong_node" else node,
            content=content,
            sha256=("0" * 64 if problem == "bad_sha" else
                    hashlib.sha256(content.encode()).hexdigest()),
        )

    result = review_pc_b_survival_evidence(
        _settings(), None, remote=True, now=NOW,
        remote_probe=get_saved, owner_probe=owner,
        attestation_probe=attestation,
    )
    assert result.status == expected
    assert result.production_qualified is False
    assert result.issue_93_closure_authorized is False


def test_pc_a_cli_accepts_remote_survival_flag_without_local_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from artifex import cli
    from artifex.qualification import handoff, overview

    cfg = tmp_path / "controller.yaml"
    cfg.write_text("{}")
    monkeypatch.setattr(cli, "_settings", lambda _: _settings())
    params: list[tuple[str, bool, object]] = []

    class FakeOverview:
        environment_ready = False

        def model_dump(self, *, mode: str) -> dict[str, object]:
            return {"production_qualified": False}

    def overview_stub(*args, **kwargs):
        params.append(("overview", kwargs["pc_b_survival_live"],
                       kwargs["pc_b_survival_report"]))
        return FakeOverview()

    async def handoff_stub(*args, **kwargs):
        params.append(("handoff", kwargs["pc_b_survival_live"],
                       kwargs["pc_b_survival_report"]))

        class FakeHandoff:
            state = "blocked"

            def model_dump(self, *, mode: str) -> dict[str, object]:
                return {"production_qualified": False, "state": "blocked"}

        return FakeHandoff()

    monkeypatch.setattr(overview, "compile_qualification_overview", overview_stub)
    monkeypatch.setattr(handoff, "inspect_qualification_handoff", handoff_stub)
    for cmd in ("overview", "handoff"):
        result = CliRunner().invoke(app, [
            "qualify", cmd, "--config", str(cfg),
            "--pc-b-survival-live", "--json",
        ])
        assert result.exit_code == (0 if cmd == "overview" else 1), result.output
    assert params == [("overview", True, None), ("handoff", True, None)]
