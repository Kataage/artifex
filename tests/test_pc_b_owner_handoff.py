"""Offline PC-B owner report correlation against live authenticated PC-B facts."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.owner_handoff import correlate_pc_b_owner_report
from artifex.qualification.renderer_owner_evidence import _REQUIRED_CHECKS
from artifex.render_node.models import (
    RemoteRendererOwnerAudit,
    RenderNodeAttestation,
)
from artifex.render_node.native_launcher_probe import LauncherEvidence

NOW = datetime(2026, 10, 9, 14, 0, tzinfo=UTC)


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_nodes.primary = "renderer-b"
    settings.render_nodes.nodes["renderer-b"] = RenderNodeConfig(
        base_url="http://127.0.0.1:8188",
        attestation_url="http://127.0.0.1:8190",
    )
    return settings


def _owner(*, pid: int = 4433, captured: datetime = NOW) -> dict[str, object]:
    return {
        "schema_version": 1,
        "captured_utc": captured.isoformat(),
        "status": "observed_stable",
        "checks": {
            key: {"status": "pass", "reason": "observed"}
            for key in _REQUIRED_CHECKS
        },
        "actual_listener_pid": pid,
        "actual_process_started_utc": "2026-10-09T12:30:00Z",
        "launcher_pid": 4411,
        "receipt_schema": 2,
        "scheduler_state": "Running",
        "process_observation_verified": True,
        "restart_authorized": False,
        "child_survival_qualified": False,
        "production_qualified": False,
        "mutated_services": False,
    }


def _save(path: Path, *, captured: datetime = NOW, host: str = "pc-b") -> None:
    fixture = LauncherEvidence(
        observed_at_utc=captured - timedelta(seconds=12),
        host=host, status="observed",
        reason="isolated_mock_python_listener_verified",
        original_launcher_verified=True, listener_identity_stable=True,
        tcp_owner_stable=True, verified_runtime_provenance=True,
    )
    payload = {
        "schema_version": 1,
        "collected_utc": captured.isoformat(),
        "host": host,
        "status": "observed_independently",
        "configured_comfyui_python": "C:/ComfyUI/venv/Scripts/python.exe",
        "checks": {
            "disposable_launcher_fixture": {
                "status": "pass", "reason": "fixture_verified",
            },
            "live_comfyui_owner": {
                "status": "pass", "reason": "owner_verified",
            },
        },
        "launcher_fixture": fixture.model_dump(mode="json"),
        "owner_audit": _owner(captured=captured - timedelta(seconds=3)),
        "remaining_real_machine_evidence": [
            "real_comfyui_child_survival_after_supervisor_loss",
            "eight_hour_unattended_gpu_soak",
        ],
        "observations_are_independent": True,
        "actual_comfyui_inspected_by_owner_audit": True,
        "production_child_survival_qualified": False,
        "real_machine_gpu_qualified": False,
        "issue_93_closure_authorized": False,
        "issue_40_closure_authorized": False,
        "renderer_restart_authorized": False,
        "gpu_jobs_submitted": False,
        "production_qualified": False,
        "mutated_services": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _probes(*, pid: int = 4433, hostname: str = "pc-b", at: datetime = NOW):
    def owner(node: str, config: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        assert node == "renderer-b"
        return RemoteRendererOwnerAudit.model_validate({
            "node_id": node, "audit": _owner(pid=pid, captured=at),
        })

    def attest(node: str, config: RenderNodeConfig, *,
               fresh: bool, timeout_seconds: float) -> RenderNodeAttestation:
        assert node == "renderer-b" and fresh
        return RenderNodeAttestation(
            node_id=node, created_at=at, hostname=hostname,
            os={"system": "Windows"}, comfyui_base_url="http://127.0.0.1:8188",
        )

    return owner, attest


def _compare(settings: ArtifexSettings, path: Path, *,
             now: datetime = NOW, pid: int = 4433,
             host: str = "pc-b"):
    probe, attest = _probes(pid=pid, hostname=host, at=now)
    return correlate_pc_b_owner_report(
        settings, path, now=now, owner_probe=probe, attestation_probe=attest,
    )


def test_correlated_report_still_never_authenticates_copied_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "copied-pc-b.json"
    _save(path)
    observed = _compare(_settings(), path)
    assert observed.status == "correlated_read_only"
    assert observed.saved_host == observed.live_host == "pc-b"
    assert observed.saved_listener_pid == observed.live_listener_pid == 4433
    assert observed.report_sha256 is not None
    assert "eight_hour_unattended_gpu_soak" in observed.remaining_real_machine_evidence
    assert "all_fourteen_real_machine_qualification_stages" in (
        observed.remaining_real_machine_evidence
    )
    assert observed.file_source_authenticated is False
    assert observed.production_qualified is False
    assert observed.renderer_restart_authorized is False
    assert observed.stage_pass_registered is False
    assert observed.supervisor_loss_survival_qualified is False
    assert observed.services_mutated is False


@pytest.mark.parametrize("problem,expected", (
    ("wrong_host", "mismatch"),
    ("wrong_pid", "mismatch"),
    ("stale_file", "stale"),
    ("future_file", "stale"),
    ("missing", "blocked"),
    ("malformed", "blocked"),
    ("oversized", "blocked"),
    ("false_production_qualified", "blocked"),
    ("forged_status", "blocked"),
    ("missing_check", "blocked"),
    ("remote_stale", "blocked"),
))
def test_bad_local_or_remote_evidence_fails_closed(
    tmp_path: Path, problem: str, expected: str,
) -> None:
    path = tmp_path / "evidence.json"
    captured = (
        NOW - timedelta(hours=1) if problem == "stale_file"
        else NOW + timedelta(hours=1) if problem == "future_file"
        else NOW
    )
    _save(path, captured=captured)
    if problem in {
        "false_production_qualified", "forged_status", "missing_check",
    }:
        data = json.loads(path.read_text())
        if problem == "false_production_qualified":
            data["production_qualified"] = True
        elif problem == "forged_status":
            data["launcher_fixture"]["status"] = "blocked"
        else:
            del data["owner_audit"]["checks"]["tcp_ownership"]
        path.write_text(json.dumps(data))
    if problem == "missing":
        path.unlink()
    if problem == "malformed":
        path.write_text("{not json")
    if problem == "oversized":
        path.write_text("0" * (129 * 1024))

    now = NOW - timedelta(hours=1) if problem == "remote_stale" else NOW
    report = _compare(
        _settings(), path,
        now=NOW,
        pid=4434 if problem == "wrong_pid" else 4433,
        host="other-pc" if problem == "wrong_host" else "pc-b",
    )
    if problem == "remote_stale":
        probe, attest = _probes(at=now)
        report = correlate_pc_b_owner_report(
            _settings(), path, now=NOW,
            owner_probe=probe, attestation_probe=attest,
        )
    assert report.status == expected
    assert report.remaining_real_machine_evidence
    assert report.production_qualified is False
    assert report.file_source_authenticated is False


def test_network_failure_is_unavailable_not_a_pass(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    _save(path)
    counts: list[str] = []

    def fail(node: str, cfg: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        counts.append(node)
        raise OSError("no authenticated PC-B")

    def never(*args, **kwargs):
        raise AssertionError("Never call attestation after owner probe failure")

    result = correlate_pc_b_owner_report(
        _settings(), path, now=NOW,
        owner_probe=fail, attestation_probe=never,
    )
    assert result.status == "unavailable"
    assert counts == ["renderer-b"]
    assert result.production_qualified is False


def test_no_external_fetch_on_bad_file(tmp_path: Path) -> None:
    attempts: list[str] = []

    def fail(*args, **kwargs):
        attempts.append("network")
        raise AssertionError("should not reach network")

    result = correlate_pc_b_owner_report(
        _settings(), tmp_path / "missing.json", now=NOW,
        owner_probe=fail, attestation_probe=fail,
    )
    assert result.status == "blocked"
    assert attempts == []


def test_overview_cli_accepts_copied_report_without_executing_pc_b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from artifex import cli

    chosen = tmp_path / "local.yaml"
    chosen.write_text("render_nodes: {}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeReport:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            return {
                "production_qualified": False,
                "pc_b_owner_evidence": {
                    "status": "correlated_read_only",
                    "file_source_authenticated": False,
                },
            }

    def compile(*args, **kwargs):
        captured.update(kwargs)
        return FakeReport()

    monkeypatch.setattr(
        "artifex.qualification.overview.compile_qualification_overview", compile,
    )
    monkeypatch.setattr(cli, "_settings", lambda path: _settings())
    result = CliRunner().invoke(app, [
        "qualify", "overview", "--config", str(chosen),
        "--pc-b-owner-report", str(tmp_path / "handoff.json"), "--json",
    ])
    assert result.exit_code == 0, result.output
    assert captured["pc_b_owner_report"] == tmp_path / "handoff.json"
    assert json.loads(result.output)["production_qualified"] is False
