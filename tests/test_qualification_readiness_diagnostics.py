from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig, RenderNodesConfig
from artifex.controller_preflight import ControllerPreflight, PreflightCheck
from artifex.qualification.models import (
    QualificationSession,
    QualificationStage,
    QualificationStageEvidence,
)
from artifex.qualification.readiness_diagnostics import (
    QualificationReadiness,
    _read_session,
    diagnose_qualification_readiness,
)
from artifex.render_node.models import RemoteRendererOwnerAudit
from artifex.render_node.startup_inspection import (
    RemoteRendererSafetyInspection,
    RendererStartupInspection,
)

REQUIRED = (
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
)
NOW = datetime(2026, 10, 9, 6, 0, tzinfo=UTC)


def _settings(tmp_path: Path) -> ArtifexSettings:
    config = ArtifexSettings()
    config.qualification.evidence_dir = tmp_path / "qualification"
    config.llm.bootstrap.enabled = False
    config.render_nodes = RenderNodesConfig(
        primary="gpu-b",
        nodes={
            "gpu-b": RenderNodeConfig(
                base_url="http://pc-b:8191", output_mode="api",
                attestation_url="http://pc-b:8190",
                attestation_token_env="ARTIFEX_RENDER_NODE_TOKEN",
            ),
        },
    )
    return config


def _remote(
    *, status: str = "observed_stable", age_seconds: int = 0,
    check: str | None = None,
) -> RemoteRendererOwnerAudit:
    checks = {name: {"status": "pass", "reason": "verified"} for name in REQUIRED}
    if check:
        checks[check] = {"status": "fail", "reason": "unsafe"}
    return RemoteRendererOwnerAudit.model_validate({
        "node_id": "gpu-b",
        "audit": {
            "schema_version": 1,
            "captured_utc": (NOW - timedelta(seconds=age_seconds)).isoformat(),
            "status": status,
            "checks": checks,
            "actual_listener_pid": 404,
            "actual_process_started_utc": "2026-10-09T00:00:00Z",
            "launcher_pid": 100,
            "receipt_schema": 2,
            "scheduler_state": "Running",
            "process_observation_verified": status == "observed_stable",
            "restart_authorized": False, "child_survival_qualified": False,
            "production_qualified": False, "mutated_services": False,
        },
    })


def _safety(
    *, status: str = "owned_observed", pid: int = 404,
    age_seconds: int = 0, service_coherent: bool = True,
    checks_pass: bool = True,
) -> RemoteRendererSafetyInspection:
    return RemoteRendererSafetyInspection(
        node_id="gpu-b",
        inspection=RendererStartupInspection(
            captured_utc=NOW - timedelta(seconds=age_seconds),
            status=status, renderer_node_id="gpu-b",
            configured_comfyui_port=8188, configured_attestation_port=8190,
            configured_gateway_port=8191,
            config_protected=True, windows_native=True,
            snapshots_consistent=True,
            service_ports_coherent=service_coherent,
            owner_audit_status="observed_stable",
            actual_owned_listener_pid=pid,
            owner_checks={
                name: "pass" if checks_pass else "unknown" for name in REQUIRED
            },
        ),
    )


def _preflight(_: ArtifexSettings) -> ControllerPreflight:
    return ControllerPreflight(
        ready=True, renderer_id="gpu-b",
        checks=(
            PreflightCheck(name="render_attestation", ready=True, detail="verified"),
            PreflightCheck(name="comfyui_lan", ready=True, detail="verified"),
            PreflightCheck(name="llm_lan", ready=True, detail="verified"),
        ),
    )


def _prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    import artifex.qualification.readiness_diagnostics as module

    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "private-do-not-print")
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module.shutil, "which", lambda arg: "/usr/bin/uv")


def test_readiness_reports_device_and_never_claims_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _prepare(monkeypatch)
    config = _settings(tmp_path)
    observed: list[str] = []

    def owner(node_id: str, node: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        observed.append(node_id)
        return _remote()

    report = diagnose_qualification_readiness(
        config, now=NOW, controller_probe=_preflight, owner_probe=owner,
        safety_probe=lambda node, cfg: _safety(),
    )
    assert report.environment_ready
    assert report.primary_node_id == "gpu-b"
    assert observed == ["gpu-b"]
    assert report.actual_machine_qualification_complete is False
    assert report.actual_gpu_soak_verified is False
    assert report.mutated_services is False
    assert all(x.status == "pass" for x in report.checks if x.target != "qualification")
    assert next(x for x in report.checks if x.name == "session_records").status == "unknown"
    assert "private-do-not-print" not in report.model_dump_json()


@pytest.mark.parametrize(
    ("mode", "name"),
    [
        ("bad_owner", "owner:overall"),
        ("socket_fail", "owner:tcp_ownership"),
        ("old", "owner:freshness"),
        ("future", "owner:freshness"),
        ("unavailable", "owner:remote_probe"),
        ("bad_preflight", "preflight:comfyui_lan"),
    ],
)
def test_readiness_fails_closed_for_pc_b_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, name: str,
) -> None:
    _prepare(monkeypatch)
    settings = _settings(tmp_path)

    def owner(node_id: str, node: RenderNodeConfig) -> RemoteRendererOwnerAudit:
        if mode == "unavailable":
            raise httpx.ConnectError("disconnected")
        return _remote(
            status="blocked" if mode == "bad_owner" else "observed_stable",
            check="tcp_ownership" if mode == "socket_fail" else None,
            age_seconds=200 if mode == "old" else -120 if mode == "future" else 0,
        )

    def preflight(s: ArtifexSettings) -> ControllerPreflight:
        if mode != "bad_preflight":
            return _preflight(s)
        return ControllerPreflight(
            ready=False, renderer_id="gpu-b",
            checks=(PreflightCheck(name="comfyui_lan", ready=False, detail="offline"),),
        )

    report = diagnose_qualification_readiness(
        settings, now=NOW, controller_probe=preflight, owner_probe=owner,
        safety_probe=lambda node, cfg: _safety(),
    )
    assert report.environment_ready is False
    check = next(x for x in report.checks if x.name == name)
    assert check.status in ("fail", "unknown")
    assert check.target == "pc_b"
    assert check.next_action
    assert report.next_actions
    assert report.actual_machine_qualification_complete is False


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("missing_services", "safety:three_ports"),
        ("not_owned", "safety:overall"),
        ("unknown_owner_check", "safety:overall"),
        ("changed_pid", "safety:pid_consistency"),
        ("stale", "safety:freshness"),
        ("future", "safety:freshness"),
        ("remote_down", "safety:remote_probe"),
        ("wrong_node", "safety:remote_probe"),
    ],
)
def test_unsafe_remote_three_port_evidence_never_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    mode: str, expected: str,
) -> None:
    _prepare(monkeypatch)

    def safety_probe(
        node_id: str, config: RenderNodeConfig,
    ) -> RemoteRendererSafetyInspection:
        if mode == "remote_down":
            raise httpx.ConnectError("unreachable")
        if mode == "wrong_node":
            raise ValueError("remote safety node ID mismatch")
        return _safety(
            status="service_ports_blocked" if mode == "not_owned" else "owned_observed",
            service_coherent=mode != "missing_services",
            checks_pass=mode != "unknown_owner_check",
            pid=405 if mode == "changed_pid" else 404,
            age_seconds=200 if mode == "stale" else -120 if mode == "future" else 0,
        )

    report = diagnose_qualification_readiness(
        _settings(tmp_path), now=NOW,
        controller_probe=_preflight,
        owner_probe=lambda *args: _remote(),
        safety_probe=safety_probe,
    )
    assert report.environment_ready is False
    bad_check = next(x for x in report.checks if x.name == expected)
    assert bad_check.status in {"fail", "unknown"}
    assert bad_check.next_action
    assert report.actual_machine_qualification_complete is False
    assert report.actual_gpu_soak_verified is False
    assert report.mutated_services is False


def test_absent_auth_never_calls_remote_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _prepare(monkeypatch)
    monkeypatch.delenv("ARTIFEX_RENDER_NODE_TOKEN")
    calls = []

    def owner(*args: Any) -> RemoteRendererOwnerAudit:
        calls.append(args)
        pytest.fail("No credentials: remote probe must not run")

    report = diagnose_qualification_readiness(
        _settings(tmp_path), now=NOW, controller_probe=_preflight, owner_probe=owner,
        safety_probe=lambda *args: pytest.fail("No credentials: safety probe must not run"),
    )
    assert not report.environment_ready
    assert not calls
    assert next(x for x in report.checks if x.name == "owner:remote_probe").status == "unknown"
    assert next(x for x in report.checks if x.name == "remote_owner_auth").status == "fail"


def test_missing_primary_does_not_spoof_pc_b_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _prepare(monkeypatch)
    settings = _settings(tmp_path)
    settings.render_nodes = RenderNodesConfig()
    report = diagnose_qualification_readiness(
        settings, now=NOW, controller_probe=_preflight,
        owner_probe=lambda *args: pytest.fail("No primary renderer"),
        safety_probe=lambda *args: pytest.fail("No primary renderer safety"),
    )
    assert not report.environment_ready
    assert report.primary_node_id is None
    assert not any(c.name.startswith("owner:") for c in report.checks)


def test_session_records_are_not_revalidated_or_marked_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _prepare(monkeypatch)
    settings = _settings(tmp_path)
    folder = settings.qualification.evidence_dir / "trial-01"
    folder.mkdir(parents=True)
    session = QualificationSession(
        session_id="trial-01", created_at=NOW, updated_at=NOW,
        hostname="pc-a", environment={}, configuration={}, workflow={},
        assets=(), loras=(), doctor_ready=False, doctor={},
        stages={
            stage.value: QualificationStageEvidence(
                stage=stage,
                status="pass" if stage is QualificationStage.DOCTOR else "pending",
            )
            for stage in QualificationStage
        },
    )
    (folder / "qualification.json").write_text(
        session.model_dump_json(), encoding="utf-8",
    )
    report = diagnose_qualification_readiness(
        settings, session_id="trial-01", now=NOW,
        controller_probe=_preflight, owner_probe=lambda *args: _remote(),
        safety_probe=lambda *args: _safety(),
    )
    assert report.environment_ready
    assert report.recorded_stages["doctor"] == "pass"
    assert "overnight_soak" in report.pending_stages
    assert len(report.pending_stages) == 13
    assert report.actual_machine_qualification_complete is False
    assert report.mutated_services is False
    assert _read_session(settings.qualification.evidence_dir, "trial-01") == session
    with pytest.raises(ValueError, match="unsafe"):
        _read_session(settings.qualification.evidence_dir, "../escape")


def test_session_symlink_and_oversize_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    folder = root / "trial-02"
    folder.mkdir(parents=True)
    file = folder / "qualification.json"
    file.write_bytes(b"z" * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded"):
        _read_session(root, "trial-02")
    file.unlink()
    elsewhere = tmp_path / "outside.json"
    elsewhere.write_text("{}", encoding="utf-8")
    try:
        file.symlink_to(elsewhere)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink privilege unavailable")
    with pytest.raises(ValueError, match="symlink"):
        _read_session(root, "trial-02")


def test_readiness_cli_requires_real_config_and_refuses_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import artifex.cli as cli_module

    runner = CliRunner()
    config = tmp_path / "local.yaml"
    absent = runner.invoke(
        app, ["qualify", "readiness", "--config", str(config)],
    )
    assert absent.exit_code == 1
    assert "configuration is required" in absent.output
    config.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        cli_module, "diagnose_qualification_readiness",
        None, raising=False,
    )
    import artifex.qualification.readiness_diagnostics as module

    def diagnosed(settings: ArtifexSettings, **kwargs: Any) -> QualificationReadiness:
        return QualificationReadiness(
            captured_utc=NOW, primary_node_id=None, environment_ready=False,
            checks=(), next_actions=("Configure PC-B.",),
        )

    monkeypatch.setattr(module, "diagnose_qualification_readiness", diagnosed)
    monkeypatch.setattr(cli_module, "_settings", lambda path: _settings(tmp_path))
    output = tmp_path / "report.json"
    result = runner.invoke(
        app, ["qualify", "readiness", "--config", str(config), "--output", str(output)],
    )
    assert result.exit_code == 1
    assert output.is_file()
    assert json.loads(result.output)["environment_ready"] is False
    assert json.loads(output.read_text())["actual_machine_qualification_complete"] is False
    replacement = runner.invoke(
        app, ["qualify", "readiness", "--config", str(config), "--output", str(output)],
    )
    assert replacement.exit_code == 1
    assert "FileExistsError" in replacement.output
    assert json.loads(output.read_text())["mutated_services"] is False
