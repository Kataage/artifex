"""Two remote owner samples must not be promoted to real-host qualification."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.owner_pair import inspect_remote_owner_pair
from artifex.qualification.renderer_owner_evidence import _REQUIRED_CHECKS
from artifex.render_node.models import RemoteRendererOwnerAudit

BASE = datetime(2026, 10, 10, 5, tzinfo=UTC)


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_nodes.primary = "gpu-b"
    settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://127.0.0.1:8188",
        attestation_url="http://127.0.0.1:8190",
        attestation_token_env="ARTIFEX_RENDER_NODE_TOKEN",
    )
    return settings


def _owner(at: datetime, *, pid: int = 4433, started: str = "2026-10-09T00:00:00Z",
           state: str = "observed_stable", node: str = "gpu-b",
           launcher: int = 4400, checks_ok: bool = True) -> RemoteRendererOwnerAudit:
    checks = {k: {"status": "pass", "reason": "observed"} for k in _REQUIRED_CHECKS}
    if not checks_ok:
        checks["tcp_ownership"]["status"] = "fail"
    return RemoteRendererOwnerAudit.model_validate({
        "node_id": node,
        "audit": {
            "schema_version": 1,
            "captured_utc": at.isoformat(),
            "status": state,
            "checks": checks,
            "actual_listener_pid": pid,
            "actual_process_started_utc": started,
            "launcher_pid": launcher,
            "receipt_schema": 2,
            "scheduler_state": "Running",
            "process_observation_verified": state == "observed_stable",
            "restart_authorized": False,
            "child_survival_qualified": False,
            "production_qualified": False,
            "mutated_services": False,
        },
    })


def _sample_pair(monkeypatch: pytest.MonkeyPatch, *, issue: str = "ok"):
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "test-only")
    clock = [0.0]
    calls: list[str] = []
    def tick() -> datetime:
        return BASE + timedelta(seconds=clock[0])
    def advance(seconds: float) -> None:
        if issue != "no_wait":
            clock[0] += seconds
    def probe(node: str, config: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        calls.append(node)
        assert config.attestation_url == "http://127.0.0.1:8190"
        if issue == "first_unavailable" and len(calls) == 1:
            raise httpx.ConnectError("private auth details")
        if issue == "second_unavailable" and len(calls) == 2:
            raise OSError("secret path")
        at = tick()
        if issue == "cached" and len(calls) == 2:
            at = BASE
        if issue == "stale" and len(calls) == 1:
            at -= timedelta(minutes=5)
        if issue == "future" and len(calls) == 1:
            at += timedelta(minutes=5)
        return _owner(
            at,
            pid=4434 if issue == "pid_swap" and len(calls) == 2 else 4433,
            started="2026-10-10T05:00:02Z" if issue == "creation_swap"
            and len(calls) == 2 else "2026-10-09T00:00:00Z",
            launcher=4401 if issue == "launcher_swap" and len(calls) == 2 else 4400,
            node="foreign" if issue == "node_swap" and len(calls) == 2 else "gpu-b",
            state="blocked" if issue == "blocked_second" and len(calls) == 2
            else "observed_stable",
            checks_ok=not (issue == "bad_tcp" and len(calls) == 1),
        )
    result = inspect_remote_owner_pair(
        _settings(), gap_seconds=5.0, owner_probe=probe,
        monotonic=lambda: clock[0], sleep=advance, now=tick,
    )
    return result, calls


def test_two_authenticated_owner_samples_remain_observational(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, calls = _sample_pair(monkeypatch)
    assert report.status == "consistent_samples"
    assert calls == ["gpu-b", "gpu-b"]
    assert report.sample_count == 2
    assert report.first_listener_pid == report.second_listener_pid == 4433
    assert report.elapsed_between_samples_seconds == 5.0
    assert report.first_process_started_utc == "2026-10-09T00:00:00Z"
    assert report.second_process_started_utc == "2026-10-09T00:00:00Z"
    assert report.first_launcher_pid == report.second_launcher_pid == 4400
    assert report.first_receipt_schema == report.second_receipt_schema == 2
    assert set(report.first_required_checks) == _REQUIRED_CHECKS
    assert set(report.second_required_checks) == _REQUIRED_CHECKS
    assert all(value == "pass" for value in report.first_required_checks.values())
    assert "CommandLine" not in report.model_dump_json()
    assert "ExecutablePath" not in report.model_dump_json()
    assert "reason" in report.model_dump()
    assert not report.production_qualified
    assert not report.uninterrupted_child_survival_proven
    assert not report.issue_93_closure_authorized
    assert not report.issue_40_closure_authorized
    assert not report.gpu_jobs_submitted
    assert not report.renderer_restart_authorized
    assert not report.task_actions_executed


@pytest.mark.parametrize(("issue", "status", "reason", "count"), [
    ("first_unavailable", "unavailable", "first_authenticated_owner_probe_unavailable", 1),
    ("second_unavailable", "unavailable", "second_authenticated_owner_probe_unavailable", 2),
    ("stale", "blocked", "first_owner_identity_unverified", 1),
    ("future", "blocked", "first_owner_identity_unverified", 1),
    ("bad_tcp", "blocked", "first_owner_identity_unverified", 1),
    ("blocked_second", "blocked", "second_owner_identity_unverified", 2),
    ("node_swap", "blocked", "second_owner_identity_unverified", 2),
    ("cached", "blocked", "remote_owner_snapshot_not_advanced", 2),
    ("no_wait", "blocked", "independent_sampling_gap_not_observed", 1),
    ("pid_swap", "blocked", "renderer_identity_changed_between_snapshots", 2),
    ("creation_swap", "blocked", "renderer_identity_changed_between_snapshots", 2),
    ("launcher_swap", "blocked", "renderer_identity_changed_between_snapshots", 2),
])
def test_fail_closed_changes_or_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    issue: str, status: str, reason: str, count: int,
) -> None:
    report, calls = _sample_pair(monkeypatch, issue=issue)
    assert report.status == status
    assert report.reason == reason
    assert len(calls) == count
    assert report.sample_count == (
        0 if issue == "first_unavailable" else
        1 if issue == "second_unavailable" or count == 1 else 2
    )
    assert bool(report.first_required_checks) == (issue != "first_unavailable")
    assert bool(report.second_required_checks) == (
        count == 2 and issue != "second_unavailable"
    )
    if issue == "bad_tcp":
        assert report.first_required_checks["tcp_ownership"] == "fail"
    assert "secret" not in report.model_dump_json().lower()
    assert report.production_qualified is False
    assert report.gpu_jobs_submitted is False


def test_missing_token_never_makes_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    def unexpected(node: str, config: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        raise AssertionError("Must not fetch without authentication")
    result = inspect_remote_owner_pair(_settings(), owner_probe=unexpected)
    assert result.status == "unconfigured"
    assert result.sample_count == 0


@pytest.mark.parametrize("gap", [0, -1, 61, float("nan"), float("inf")])
def test_unbounded_or_invalid_gap_refused(gap: float) -> None:
    with pytest.raises(ValueError, match="gap"):
        inspect_remote_owner_pair(_settings(), gap_seconds=gap)


def test_cli_exposes_safe_json_and_never_performs_remote_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.qualification.owner_pair as pair
    from artifex import cli

    config = tmp_path / "pc-a.yaml"
    config.write_text("{}")
    from artifex.qualification.owner_pair import OwnerPairObservation
    monkeypatch.setattr(cli, "_settings", lambda path: _settings())
    monkeypatch.setattr(pair, "inspect_remote_owner_pair", lambda *a, **k: (
        OwnerPairObservation(
            status="consistent_samples", reason="two_point_observation",
            node_id="gpu-b", checked_utc=BASE,
            sample_count=2, requested_gap_seconds=3.0,
            elapsed_between_samples_seconds=3.0,
        )
    ))
    result = CliRunner().invoke(app, [
        "qualify", "owner-pair-check", "--config", str(config),
        "--gap-seconds", "3", "--json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["production_qualified"] is False
    assert payload["sample_count"] == 2
    assert payload["renderer_restart_authorized"] is False


def test_owner_pair_explicit_evidence_save_is_exclusive_and_includes_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.qualification.owner_pair as pair
    from artifex import cli
    from artifex.qualification.owner_pair import OwnerPairObservation

    config = tmp_path / "pc-a.yaml"
    config.write_text("{}")
    target = tmp_path / "evidence" / "owner-pair.json"
    monkeypatch.setattr(cli, "_settings", lambda path: _settings())
    monkeypatch.setattr(pair, "inspect_remote_owner_pair", lambda *a, **k: (
        OwnerPairObservation(
            status="blocked", reason="renderer_identity_changed_between_snapshots",
            node_id="gpu-b", checked_utc=BASE,
            sample_count=2, requested_gap_seconds=5.0,
            elapsed_between_samples_seconds=5.0,
            first_listener_pid=4433, second_listener_pid=4434,
            first_process_started_utc="2026-10-09T00:00:00Z",
            first_launcher_pid=4400, first_receipt_schema=2,
            first_required_checks={"tcp_ownership": "pass"},
        )
    ))
    args = [
        "qualify", "owner-pair-check", "--config", str(config),
        "--output", str(target), "--json",
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 1
    assert target.is_file()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data == json.loads(first.stdout)
    assert data["reason"] == "renderer_identity_changed_between_snapshots"
    assert data["first_process_started_utc"] == "2026-10-09T00:00:00Z"
    assert data["first_required_checks"] == {"tcp_ownership": "pass"}
    assert data["renderer_restart_authorized"] is False
    assert data["production_qualified"] is False
    assert "ARTIFEX_RENDER_NODE_TOKEN" not in target.read_text()

    again = CliRunner().invoke(app, args)
    assert again.exit_code == 1
    assert json.loads(target.read_text(encoding="utf-8")) == data

    link = tmp_path / "symlink.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        return
    unsafe = CliRunner().invoke(app, [
        *args[:args.index("--output")],
        "--output", str(link), "--json",
    ])
    assert unsafe.exit_code == 1
    assert json.loads(target.read_text(encoding="utf-8")) == data


def test_owner_pair_save_uses_configurable_evidence_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.qualification.owner_pair as pair
    from artifex import cli
    from artifex.qualification.owner_pair import OwnerPairObservation

    settings = _settings()
    settings.qualification.evidence_dir = tmp_path / "custom-evidence"
    config = tmp_path / "pc-a.yaml"
    config.write_text("{}")
    monkeypatch.setattr(cli, "_settings", lambda path: settings)
    monkeypatch.setattr(pair, "inspect_remote_owner_pair", lambda *a, **k: (
        OwnerPairObservation(
            status="consistent_samples", reason="same_renderer_identity_at_two_observations",
            node_id="gpu-b", checked_utc=BASE,
            sample_count=2, requested_gap_seconds=5.0,
            elapsed_between_samples_seconds=5.0,
        )
    ))
    command = ["qualify", "owner-pair-check", "--config", str(config), "--save", "--json"]
    first = CliRunner().invoke(app, command)
    assert first.exit_code == 0, first.output
    second = CliRunner().invoke(app, command)
    assert second.exit_code == 0, second.output
    results = list((settings.qualification.evidence_dir / "owner-pair").glob("*.json"))
    assert len(results) == 2
    assert all(json.loads(path.read_text())["production_qualified"] is False for path in results)
    assert all(json.loads(path.read_text())["sample_count"] == 2 for path in results)
