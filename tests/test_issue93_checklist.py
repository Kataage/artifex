"""Focused Issue #93 read-only checklist: never certify host or GPU."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.issue93_checklist import compile_issue93_checklist
from artifex.qualification.owner_handoff import PCBOwnerCorrelation
from artifex.qualification.owner_pair_review import OwnerPairReview
from artifex.qualification.survival_review import PCBSurvivalEvidenceReview

NOW = datetime(2026, 10, 10, 9, tzinfo=UTC)


def _settings() -> ArtifexSettings:
    s = ArtifexSettings()
    s.render_nodes.primary = "gpu-b"
    s.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://127.0.0.1:8188",
        attestation_url="http://127.0.0.1:8190",
        attestation_token_env="ARTIFEX_RENDER_NODE_TOKEN",
    )
    return s


def _owner(pid: int = 441) -> PCBOwnerCorrelation:
    return PCBOwnerCorrelation(
        checked_utc=NOW, status="correlated_read_only",
        reason="authenticated_snapshot_correlated",
        node_id="gpu-b", live_listener_pid=pid,
        source_mode="authenticated_remote", remote_bearer_checked=True,
        remaining_real_machine_evidence=("supervisor_exit",),
    )


def _pair(pid: int = 441) -> OwnerPairReview:
    return OwnerPairReview(
        checked_utc=NOW, node_id="gpu-b",
        status="correlated_read_only", reason="saved_owner_identity_matches_live_pc_b",
        current_listener_pid=pid, saved_listener_pid=pid,
        saved_required_checks_consistent=True,
        saved_identity_consistent=True, live_pc_b_owner_correlated=True,
        remote_bearer_checked=True,
    )


def _survival(pid: int = 441) -> PCBSurvivalEvidenceReview:
    return PCBSurvivalEvidenceReview(
        reviewed_utc=NOW, status="replayed_and_live_identity_matched",
        reason="copied_trace_replayed_and_current_pc_b_process_matches",
        node_id="gpu-b", current_listener_pid=pid, historical_listener_pid=pid,
        source_mode="bearer_remote", remote_bearer_checked=True,
        trace_replayed=True,
    )


def _run(monkeypatch: pytest.MonkeyPatch, *, owner: PCBOwnerCorrelation | None = None,
         pair: OwnerPairReview | None = None,
         survival: PCBSurvivalEvidenceReview | None = None):
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "fixture-only")
    calls: list[str] = []

    def probe(k: str, result: object | None):
        def fn(s: ArtifexSettings):
            calls.append(k)
            if result is None:
                raise OSError("secret hostname and private Bearer")
            return result
        return fn

    report = compile_issue93_checklist(
        _settings(), now=lambda: NOW,
        owner_fetch=probe("owner", owner),
        pair_fetch=probe("pair", pair),
        survival_fetch=probe("survival", survival),
    )
    return report, calls


def test_all_three_correlated_remains_unqualified(monkeypatch: pytest.MonkeyPatch) -> None:
    report, calls = _run(monkeypatch, owner=_owner(), pair=_pair(), survival=_survival())
    assert calls == ["owner", "pair", "survival"]
    assert report.status == "observations_correlated"
    assert report.current_comfyui_listener_pid == 441
    assert report.observed_checks == 4
    assert report.missing_checks == 0
    assert report.conflicting_checks == 0
    assert all(x.state == "observed" for x in report.checks)
    assert report.next_safe_actions
    assert not report.historical_supervisor_exit_authenticated
    assert not report.native_pc_b_launcher_accepted
    assert not report.uninterrupted_comfyui_survival_qualified
    assert not report.reattachment_qualified
    assert not report.real_gpu_qualified
    assert not report.fourteen_stage_qualification_proven
    assert not report.eight_hour_soak_proven
    assert not report.issue_93_closure_authorized
    assert not report.issue_40_closure_authorized
    assert not report.production_qualified
    assert not report.services_mutated
    assert not report.task_actions_executed
    assert not report.gpu_jobs_submitted


@pytest.mark.parametrize(("kind", "status", "observed"), [
    ("owner_unavailable", "needs_evidence", 2),
    ("pair_unavailable", "needs_evidence", 2),
    ("survival_unavailable", "needs_evidence", 2),
    ("all_unavailable", "needs_evidence", 0),
    ("pid_conflict", "conflict", 3),
    ("survival_bad_transport", "needs_evidence", 2),
    ("pair_bad_auth", "needs_evidence", 2),
])
def test_partial_failure_or_pid_conflict_never_claims_issue93_ready(
    monkeypatch: pytest.MonkeyPatch, kind: str, status: str, observed: int,
) -> None:
    owner = None if kind in {"owner_unavailable", "all_unavailable"} else _owner()
    pair = None if kind in {"pair_unavailable", "all_unavailable"} else _pair(
        442 if kind == "pid_conflict" else 441
    )
    survival = None if kind in {"survival_unavailable", "all_unavailable"} else _survival()
    if kind == "survival_bad_transport" and survival is not None:
        survival = survival.model_copy(update={"remote_bearer_checked": False})
    if kind == "pair_bad_auth" and pair is not None:
        pair = pair.model_copy(update={"remote_bearer_checked": False})
    report, calls = _run(monkeypatch, owner=owner, pair=pair, survival=survival)
    assert calls == ["owner", "pair", "survival"]
    assert report.status == status
    assert report.observed_checks == observed
    assert len(report.next_safe_actions) >= 4
    assert not report.issue_93_closure_authorized
    assert not report.production_qualified
    assert "secret" not in report.model_dump_json().lower()


def test_missing_token_does_not_perform_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    def reject(s: ArtifexSettings):
        pytest.fail("Missing credentials must prevent any fetch")
    report = compile_issue93_checklist(
        _settings(), owner_fetch=reject, pair_fetch=reject, survival_fetch=reject,
        now=lambda: NOW,
    )
    assert report.status == "unconfigured"
    assert report.observed_checks == 0
    assert report.missing_checks == 4
    assert not report.production_qualified


def test_cli_returns_json_even_if_missing_evidence_and_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.qualification.issue93_checklist as module
    from artifex import cli

    selected = tmp_path / "local.yaml"
    selected.write_text("{}")
    monkeypatch.setattr(cli, "_settings", lambda _: _settings())
    monkeypatch.setattr(module, "compile_issue93_checklist", lambda s: (
        compile_issue93_checklist(
            s, owner_fetch=lambda _: _owner(),
            pair_fetch=lambda _: _pair(),
            survival_fetch=lambda _: (_ for _ in ()).throw(OSError("private")),
            now=lambda: NOW,
        )
    ))
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "fixture")
    result = CliRunner().invoke(app, [
        "qualify", "issue93-status", "--config", str(selected), "--json",
    ])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "needs_evidence"
    assert payload["missing_checks"] == 2
    assert payload["issue_93_closure_authorized"] is False
    assert payload["gpu_jobs_submitted"] is False


def test_cross_source_current_pid_mismatch_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r, _ = _run(monkeypatch, owner=_owner(), pair=_pair(), survival=_survival(999))
    assert r.status == "conflict"
    assert r.current_comfyui_listener_pid is None
    assert r.conflicting_checks == 1
    assert r.checks[-1].state == "conflict"
