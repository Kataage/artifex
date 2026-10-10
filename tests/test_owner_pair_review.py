"""Offline owner-pair acceptance review remains observational and fail-closed."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.owner_pair import OwnerPairObservation
from artifex.qualification.owner_pair_review import review_owner_pair_evidence
from artifex.qualification.renderer_owner_evidence import _REQUIRED_CHECKS
from artifex.render_node.models import RemoteRendererOwnerAudit

NOW = datetime(2026, 10, 10, 6, tzinfo=UTC)


def _settings(root: Path) -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_nodes.primary = "gpu-b"
    settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://127.0.0.1:8188", attestation_url="http://127.0.0.1:8190",
        attestation_token_env="ARTIFEX_RENDER_NODE_TOKEN",
    )
    settings.qualification.evidence_dir = root / "evidence"
    return settings


def _pair(*, state: str = "consistent_samples", **changes: object) -> OwnerPairObservation:
    check = dict.fromkeys(sorted(_REQUIRED_CHECKS), "pass")
    payload = {
        "status": state, "reason": "same_renderer_identity_at_two_observations",
        "node_id": "gpu-b", "checked_utc": NOW,
        "sample_count": 2,
        "first_captured_utc": NOW - timedelta(seconds=9),
        "second_captured_utc": NOW - timedelta(seconds=4),
        "first_listener_pid": 4433, "second_listener_pid": 4433,
        "first_process_started_utc": "2026-10-09T00:00:00Z",
        "second_process_started_utc": "2026-10-09T00:00:00Z",
        "first_launcher_pid": 4400, "second_launcher_pid": 4400,
        "first_receipt_schema": 2, "second_receipt_schema": 2,
        "first_required_checks": check, "second_required_checks": check,
        "elapsed_between_samples_seconds": 5, "requested_gap_seconds": 5,
    }
    payload.update(changes)
    return OwnerPairObservation.model_validate(payload)


def _save(settings: ArtifexSettings, report: OwnerPairObservation, *,
          filename: str = "20261010T060000Z-abcd.json") -> Path:
    path = settings.qualification.evidence_dir / "owner-pair" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(), encoding="utf-8")
    return path


def _remote(node: str, cfg: RenderNodeConfig, *,
            at: datetime = NOW, pid: int = 4433,
            started: str = "2026-10-09T00:00:00Z",
            launcher: int = 4400) -> RemoteRendererOwnerAudit:
    assert node == "gpu-b"
    return RemoteRendererOwnerAudit.model_validate({
        "node_id": node, "audit": {
            "schema_version": 1, "captured_utc": at,
            "status": "observed_stable",
            "checks": {k: {"status": "pass", "reason": "observed"} for k in _REQUIRED_CHECKS},
            "actual_listener_pid": pid,
            "actual_process_started_utc": started,
            "launcher_pid": launcher, "receipt_schema": 2,
            "scheduler_state": "Running", "process_observation_verified": True,
            "restart_authorized": False, "child_survival_qualified": False,
            "production_qualified": False, "mutated_services": False,
        }
    })


@pytest.mark.parametrize(("saved_second", "live_started", "expected"), [
    ("2026-10-09T09:00:00+09:00", "2026-10-09T00:00:00Z",
     "correlated_read_only"),
    ("2026-10-09T09:00:00+09:00", "2026-10-09T09:00:00+09:00",
     "correlated_read_only"),
    ("2026-10-09T09:00:00+09:00", "2026-10-09T09:00:01+09:00",
     "blocked"),
    ("2026-10-09T09:00:01+09:00", "2026-10-09T00:00:00Z",
     "blocked"),
    ("2026-10-09T00:00:00", "2026-10-09T00:00:00Z",
     "blocked"),
])
def test_saved_owner_pair_and_current_pc_b_use_exact_creation_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    saved_second: str, live_started: str, expected: str,
) -> None:
    settings = _settings(tmp_path)
    report_path = _save(settings, _pair(
        second_process_started_utc=saved_second,
    ))
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "test")
    calls: list[str] = []

    def probe(node: str, config: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        calls.append(node)
        return _remote(node, config, started=live_started)

    review = review_owner_pair_evidence(
        settings, report_path=report_path, owner_probe=probe,
        now=lambda: NOW + timedelta(seconds=1),
    )
    assert review.status == expected
    assert bool(calls) is (expected == "correlated_read_only"
                           or saved_second == "2026-10-09T09:00:00+09:00")
    assert not review.supervisor_loss_survival_proven
    assert not review.issue_93_closure_authorized
    assert not review.production_qualified


def test_latest_report_correlates_but_never_qualifies_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    _save(settings, _pair(), filename="20261010T055900Z-old.json")
    _save(settings, _pair(), filename="20261010T060000Z-latest.json")
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "dummy")
    calls: list[str] = []
    def fetch(node: str, cfg: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        calls.append(node)
        return _remote(node, cfg)
    report = review_owner_pair_evidence(
        settings, owner_probe=fetch, now=lambda: NOW + timedelta(seconds=1),
    )
    assert calls == ["gpu-b"]
    assert report.status == "correlated_read_only"
    assert report.saved_report_sha256 is not None
    assert report.saved_required_checks_consistent
    assert report.saved_identity_consistent
    assert report.remote_bearer_checked
    assert report.live_pc_b_owner_correlated
    assert report.saved_sample_count == 2
    assert report.saved_listener_pid == report.current_listener_pid == 4433
    assert report.current_listener_started_utc == "2026-10-09T00:00:00Z"
    assert not report.file_source_authenticated
    assert not report.supervisor_loss_survival_proven
    assert not report.real_pc_b_launcher_fixture_proven
    assert not report.issue_93_closure_authorized
    assert not report.issue_40_closure_authorized
    assert not report.production_qualified
    assert not report.task_actions_executed
    assert not report.gpu_jobs_submitted


@pytest.mark.parametrize(("changes", "expect"), [
    ({"sample_count": 1}, "blocked"),
    ({"second_listener_pid": 9999}, "blocked"),
    ({"second_process_started_utc": "2026-10-10T00:00:00Z"}, "blocked"),
    ({"second_launcher_pid": 8888}, "blocked"),
    ({"first_receipt_schema": 1}, "blocked"),
    ({"first_required_checks": {"tcp_ownership": "pass"}}, "blocked"),
    ({"first_required_checks": {"tcp_ownership": "fail"}}, "blocked"),
    ({"second_captured_utc": NOW - timedelta(seconds=9)}, "blocked"),
    ({"first_process_started_utc": "nonsense",
      "second_process_started_utc": "nonsense"}, "blocked"),
    ({"first_process_started_utc": "2026-10-09T00:00:00",
      "second_process_started_utc": "2026-10-09T00:00:00"}, "blocked"),
    ({"checked_utc": NOW - timedelta(hours=1)}, "stale"),
    ({"checked_utc": NOW + timedelta(minutes=3)}, "stale"),
    ({"node_id": "foreign"}, "blocked"),
])
def test_invalid_or_stale_saved_data_fails_before_network(
    tmp_path: Path, changes: dict[str, object], expect: str,
) -> None:
    settings = _settings(tmp_path)
    path = _save(settings, _pair(**changes))
    def never(node: str, cfg: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        raise AssertionError("Untrusted evidence must fail before network")
    result = review_owner_pair_evidence(
        settings, report_path=path, owner_probe=never, now=lambda: NOW,
    )
    assert result.status == expect
    assert not result.live_pc_b_owner_correlated


@pytest.mark.parametrize(("issue", "expected"), [
    ("wrong_pid", "blocked"), ("wrong_creation", "blocked"),
    ("wrong_launcher", "blocked"), ("replayed", "blocked"),
    ("wrong_node", "blocked"), ("unavailable", "unavailable"),
])
def test_live_correlation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, issue: str, expected: str,
) -> None:
    settings = _settings(tmp_path)
    path = _save(settings, _pair())
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "dummy")
    def fetch(node: str, cfg: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        if issue == "unavailable":
            raise OSError("secret token/host")
        item = _remote(
            node, cfg, pid=9999 if issue == "wrong_pid" else 4433,
            started="2026-10-10T00:00:00Z" if issue == "wrong_creation"
            else "2026-10-09T00:00:00Z",
            launcher=4411 if issue == "wrong_launcher" else 4400,
            at=NOW - timedelta(seconds=4) if issue == "replayed" else NOW,
        )
        return item.model_copy(update={"node_id": "foreign"}) if issue == "wrong_node" else item
    result = review_owner_pair_evidence(
        settings, report_path=path, owner_probe=fetch, now=lambda: NOW,
    )
    assert result.status == expected
    assert not result.live_pc_b_owner_correlated
    assert not result.production_qualified
    assert "secret" not in result.model_dump_json()


def test_offline_never_makes_network_or_claims_auth(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = _save(settings, _pair())
    def never(*args: object) -> RemoteRendererOwnerAudit:
        raise AssertionError("no network for offline review")
    result = review_owner_pair_evidence(
        settings, report_path=path, offline=True, owner_probe=never, now=lambda: NOW,
    )
    assert result.status == "saved_only"
    assert not result.remote_bearer_checked
    assert not result.live_pc_b_owner_correlated


def test_missing_token_refuses_live_correlation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    path = _save(settings, _pair())
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN", raising=False)
    report = review_owner_pair_evidence(settings, report_path=path, now=lambda: NOW)
    assert report.status == "unconfigured"
    assert not report.remote_bearer_checked


@pytest.mark.parametrize("problem", ["missing", "symlink", "oversized", "invalid", "unsafe_directory"])
def test_bad_file_never_triggers_probe(tmp_path: Path, problem: str) -> None:
    settings = _settings(tmp_path)
    path = _save(settings, _pair())
    if problem == "missing":
        path.unlink()
    elif problem == "symlink":
        alternate = tmp_path / "link.json"
        try:
            alternate.symlink_to(path)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        path = alternate
    elif problem == "oversized":
        path.write_bytes(b"a" * (32769))
    elif problem == "invalid":
        path.write_text("not a JSON payload")
    else:
        from unittest.mock import patch

        import artifex.qualification.owner_pair_review as module
        with patch.object(module, "_safe_path", return_value=False):
            report = review_owner_pair_evidence(settings, now=lambda: NOW)
            assert report.status == "blocked"
        return
    def never(*args: object) -> RemoteRendererOwnerAudit:
        raise AssertionError("must not contact PC-B on invalid local evidence")
    result = review_owner_pair_evidence(
        settings, report_path=path, owner_probe=never, now=lambda: NOW,
    )
    assert result.status == "blocked"


def test_cli_review_json_and_failure_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.qualification.owner_pair_review as module
    from artifex import cli

    settings = _settings(tmp_path)
    path = _save(settings, _pair())
    config = tmp_path / "local.yaml"
    config.write_text("{}")
    monkeypatch.setattr(cli, "_settings", lambda path: settings)
    monkeypatch.setattr(
        module, "review_owner_pair_evidence",
        lambda *args, **kwargs: review_owner_pair_evidence(
            settings, report_path=path, offline=True, now=lambda: NOW,
        ),
    )
    result = CliRunner().invoke(app, [
        "qualify", "owner-pair-review", "--config", str(config), "--offline",
        "--report", str(path), "--json",
    ])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["status"] == "saved_only"
    assert not data["issue_93_closure_authorized"]


def test_newest_owner_evidence_uses_file_write_time_not_random_filename_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from artifex.qualification.owner_pair_review import _latest_evidence

    settings = _settings(tmp_path)
    previous = _save(
        settings, _pair(state="blocked"), filename="20261010T060000Z-zzzz.json",
    )
    newest = _save(
        settings, _pair(), filename="20261010T060000Z-aaaa.json",
    )
    # Same-second filenames sort in the wrong order lexicographically.
    os.utime(previous, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newest, ns=(2_000_000_000, 2_000_000_000))
    selected = _latest_evidence(settings.qualification.evidence_dir / "owner-pair")
    assert selected == newest


def test_owner_pair_same_write_timestamp_fails_closed_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    settings = _settings(tmp_path)
    old = _save(
        settings, _pair(), filename="20261010T060000Z-zzzz.json",
    )
    newer = _save(
        settings, _pair(state="blocked"), filename="20261010T060000Z-aaaa.json",
    )
    # With filesystem ties, lexical/random suffixes cannot tell which was
    # actually written last. No older PASS must be silently selected.
    os.utime(old, ns=(3_000_000_000, 3_000_000_000))
    os.utime(newer, ns=(3_000_000_000, 3_000_000_000))
    def no_network(*args: object) -> RemoteRendererOwnerAudit:
        pytest.fail("Ambiguous local evidence must not contact PC-B")
    report = review_owner_pair_evidence(
        settings, owner_probe=no_network, now=lambda: NOW,
    )
    assert report.status == "blocked"
    assert report.reason == "invalid_or_unreadable_saved_evidence"
    assert not report.live_pc_b_owner_correlated
    assert not report.remote_bearer_checked


def test_owner_pair_read_time_change_fails_closed_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    saved = _save(settings, _pair())
    original = Path.read_bytes

    def mutate_after_read(path: Path) -> bytes:
        raw = original(path)
        if path == saved:
            with path.open("ab") as handle:
                handle.write(b"later")
        return raw

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    def no_network(*args: object) -> RemoteRendererOwnerAudit:
        pytest.fail("Incomplete local evidence must not contact PC-B")
    report = review_owner_pair_evidence(
        settings, report_path=saved, owner_probe=no_network, now=lambda: NOW,
    )
    assert report.status == "blocked"
    assert report.reason == "saved_evidence_changed_during_read"
    assert not report.remote_bearer_checked
    assert not report.production_qualified


def test_owner_pair_bad_newest_does_not_fall_back_to_older_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    settings = _settings(tmp_path)
    good = _save(settings, _pair(), filename="20261010T060000Z-zzzz.json")
    bad = _save(settings, _pair(state="blocked"), filename="20261010T060000Z-aaaa.json")
    os.utime(good, ns=(1_000_000_000, 1_000_000_000))
    os.utime(bad, ns=(2_000_000_000, 2_000_000_000))
    def no_network(*args: object) -> RemoteRendererOwnerAudit:
        pytest.fail("Blocked newest evidence must not fall back to old PASS")
    report = review_owner_pair_evidence(
        settings, owner_probe=no_network, now=lambda: NOW,
    )
    assert report.status == "blocked"
    assert report.saved_report_sha256 is not None
    assert report.saved_sample_count == 2
    assert not report.live_pc_b_owner_correlated
