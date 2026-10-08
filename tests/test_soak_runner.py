from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings
from artifex.qualification import REQUIRED_STAGES, QualificationStage, QualificationStatus
from artifex.qualification.soak_observer import SoakAssessment, SoakSample


def _run_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    doctor_ready: bool = True,
    observed_ready: bool = True,
    record_error: str | None = None,
) -> tuple[Path, Any, Any]:
    """Mock only live backends and the clock; never claim real hardware proof."""
    settings = ArtifexSettings()
    settings.qualification.evidence_dir = tmp_path / "evidence"
    config = tmp_path / "local.yaml"
    config.write_text("{}\n", encoding="utf-8")
    calls: list[tuple[str, Any]] = []

    class Core:
        async def close(self) -> None:
            calls.append(("close", None))

    class Doctor:
        async def run(self) -> Any:
            calls.append(("doctor", None))
            return SimpleNamespace(ready=doctor_ready)

    class Session:
        session_id = "session-test"

        def stage(self, stage: QualificationStage) -> Any:
            return SimpleNamespace(
                status=(
                    QualificationStatus.PASS
                    if stage in (QualificationStage.DOCTOR, QualificationStage.OVERNIGHT_SOAK)
                    else QualificationStatus.SKIPPED
                    if stage is QualificationStage.DISCORD_CONTROLS
                    else QualificationStatus.PENDING
                )
            )

    class Service:
        def start(self, doctor: Any) -> Any:
            calls.append(("start", doctor.ready))
            return Session()

        def load(self, session_id: str) -> Any:
            calls.append(("load", session_id))
            return Session()

        def require_soak_candidate(self, session: Any) -> None:
            calls.append(("preflight", session.session_id))

        def record(self, session_id: str, stage: Any, **kwargs: Any) -> Any:
            calls.append(("record", (session_id, stage, kwargs)))
            if record_error is not None:
                raise ValueError(record_error)
            return Session()

    def observer(settings: ArtifexSettings, **kwargs: Any) -> SoakAssessment:
        path = kwargs["output"]
        calls.append(("observe", (path, kwargs["qualification_session_id"])))
        calls.append(("duration_hours", kwargs["duration_hours"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic test data")
        if kwargs["on_sample"] is not None:
            kwargs["on_sample"](SoakSample(
                observed_at=datetime.now(UTC),
                elapsed_seconds=0.0,
                llm_ok=True, comfyui_ok=True, renderer_ok=True,
                finalized_packs=10,
            ))
        return SoakAssessment(
            run_id="synthetic",
            ready_for_soak_review=observed_ready,
            elapsed_hours=8 if observed_ready else 0,
            observed_samples=97 if observed_ready else 1,
            verified_completed_packs=3 if observed_ready else 0,
            observed_telemetry_rows=50 if observed_ready else 0,
            fatal_errors=0,
            peak_vram_mib=8000, peak_ram_mib=12000,
            minimum_free_disk_gib=50,
            evidence_sha256="a" * 64,
            issues=() if observed_ready else ("LLM disconnected",),
        )

    monkeypatch.setattr("artifex.cli._settings", lambda path: settings)
    monkeypatch.setattr("artifex.cli.build_core", lambda _: Core())
    monkeypatch.setattr("artifex.cli.build_doctor", lambda _: Doctor())
    monkeypatch.setattr("artifex.cli._qualification_service", lambda _: Service())
    monkeypatch.setattr("artifex.cli.observe_soak", observer)
    return config, calls, settings


def _invoke(config: Path, *args: str) -> Any:
    return CliRunner().invoke(
        app, ["qualify", "soak-run", "--config", str(config), *args]
    )


def test_soak_run_uses_one_command_and_persists_stage_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calls, settings = _run_setup(tmp_path, monkeypatch)
    result = _invoke(config)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["session_id"] == "session-test"
    assert payload["overnight_soak"] == "pass"
    assert payload["production_qualified"] is False
    assert "unattended_multi_pack" in payload["remaining_stages"]
    assert len(payload["remaining_stages"]) == len(REQUIRED_STAGES) - 3
    assert Path(payload["evidence_path"]).is_file()
    assert Path(payload["evidence_path"]).is_relative_to(
        settings.qualification.evidence_dir
    )
    assert "soak sample:" in result.stderr
    assert len([x for x in calls if x[0] == "doctor"]) == 1
    assert len([x for x in calls if x[0] == "start"]) == 1
    assert len([x for x in calls if x[0] == "record"]) == 1
    assert len([x for x in calls if x[0] == "close"]) == 2
    _, (_, stage, params) = next(x for x in calls if x[0] == "record")
    assert stage is QualificationStage.OVERNIGHT_SOAK
    assert params["details"]["evidence_sha256"] == "a" * 64


def test_soak_run_reuses_existing_session_without_new_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calls, _ = _run_setup(tmp_path, monkeypatch)
    result = _invoke(config, "--session-id", "session-test", "--no-progress")
    assert result.exit_code == 0, result.output
    assert ("load", "session-test") in calls
    assert not any(action == "start" for action, _ in calls)
    assert "soak sample:" not in result.stderr


def test_soak_run_rejects_unhealthy_doctor_before_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calls, _ = _run_setup(tmp_path, monkeypatch, doctor_ready=False)
    result = _invoke(config)
    assert result.exit_code == 1
    assert "doctor checks are not ready" in result.output
    assert not any(name in ("start", "observe", "record") for name, _ in calls)


def test_soak_run_does_not_register_failed_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calls, _ = _run_setup(tmp_path, monkeypatch, observed_ready=False)
    result = _invoke(config)
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["overnight_soak"] == "fail"
    assert payload["production_qualified"] is False
    assert "LLM disconnected" in payload["observation"]["issues"]
    assert Path(payload["evidence_path"]).is_file()
    assert not any(action == "record" for action, _ in calls)


def test_soak_run_rejects_too_short_and_existing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calls, settings = _run_setup(tmp_path, monkeypatch)
    short = _invoke(config, "--hours", "0.1")
    assert short.exit_code == 1
    assert "at least 8 hours" in short.output
    assert not any(action == "doctor" for action, _ in calls)
    existing = settings.qualification.evidence_dir / "existing.jsonl"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"preexisting contents")
    denied = _invoke(config, "--output", str(existing))
    assert denied.exit_code == 1
    assert "refusing to overwrite" in denied.output
    assert existing.read_bytes() == b"preexisting contents"
    assert not any(action == "doctor" for action, _ in calls)


def test_soak_run_failed_binding_retains_trace_and_does_not_claim_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calls, settings = _run_setup(
        tmp_path, monkeypatch, record_error="configuration drift"
    )
    result = _invoke(config)
    assert result.exit_code == 1
    assert "configuration drift" in result.output
    observed_paths = [data[0] for name, data in calls if name == "observe"]
    assert len(observed_paths) == 1
    assert observed_paths[0].read_bytes() == b"synthetic test data"
    assert observed_paths[0].is_relative_to(settings.qualification.evidence_dir)


def test_soak_run_enforces_eight_hour_floor_even_if_configured_lower(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calls, settings = _run_setup(tmp_path, monkeypatch)
    settings.qualification.minimum_soak_hours = 1
    rejected = _invoke(config, "--hours", "1")
    assert rejected.exit_code == 1
    assert "at least 8 hours" in rejected.output
    assert not any(name == "doctor" for name, _ in calls)

    accepted = _invoke(config)
    assert accepted.exit_code == 0, accepted.output
    assert json.loads(accepted.stdout)["overnight_soak"] == "pass"
    assert ("duration_hours", 8.0) in calls


def test_soak_run_refuses_evidence_outside_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calls, _ = _run_setup(tmp_path, monkeypatch)
    result = _invoke(config, "--output", str(tmp_path / "outside.jsonl"))
    assert result.exit_code == 1
    assert "inside qualification.evidence_dir" in result.output
    assert not any(action == "doctor" for action, _ in calls)
