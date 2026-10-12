"""PC-A fileless two-PC live preflight, never GPU or Task Scheduler mutations."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.deployment import DeploymentCheck, DeploymentReport
from artifex.deployment_pair_live import (
    PairLiveReadiness,
    collect_pair_live_readiness,
    compile_pair_live_readiness,
)
from artifex.qualification.owner_handoff import PCBOwnerCorrelation
from artifex.qualification.pair_installation import TwoPCInstallationAudit
from artifex.qualification.pc_b_remote_field_gates import RemotePCBPassiveFieldGates

NOW = datetime(2026, 10, 12, 6, tzinfo=UTC)


def _settings() -> ArtifexSettings:
    settings = ArtifexSettings()
    settings.render_nodes.primary = "gpu-b"
    settings.render_nodes.nodes["gpu-b"] = RenderNodeConfig(
        base_url="http://192.0.2.55:8188",
        attestation_url="http://192.0.2.55:8190",
        attestation_token_env="ARTIFEX_SECRET_TOKEN_NAME",
    )
    return settings


def _installation(
    *, status: str = "ready_for_passive_monitoring",
    node_id: str = "gpu-b", task: str = "running",
    heartbeat: str = "fresh",
) -> TwoPCInstallationAudit:
    def task_entry(role: str, state: str) -> dict[str, object]:
        return {
            "role": role, "status": state,
            "registered": True, "managed": True,
            "configuration_verified": state == "running",
            "reason_code": "verified_running" if state == "running"
            else "verified_but_not_running",
        }

    return TwoPCInstallationAudit.model_validate({
        "checked_utc": NOW,
        "status": status,
        "pc_a_controller_task": task_entry("controller", task),
        "pc_b_node_id": node_id,
        "pc_b": {
            "node_id": node_id,
            "audit": {
                "node_id": node_id, "hostname": "SECRET-HOSTNAME",
                "captured_utc": NOW.isoformat(),
                "native_windows": True, "managed_renderer_configured": True,
                "renderer_task": task_entry("renderer", "running"),
                "survival_observer_task": task_entry("survival-observer", "running"),
                "evidence_spool": "empty",
                "observer_heartbeat": heartbeat,
                "safe_for_passive_observation": heartbeat == "fresh",
            },
        },
        "blockers": [] if status == "ready_for_passive_monitoring"
        else ["pc_a_controller_task_registered_not_running"]
        if status == "pc_a_needs_setup" else ["pc_b_observer_heartbeat_stale"],
        "next_actions": ["Do not restart GPU process"],
    })


def _controller(*, ready: bool = True, bad_name: str = "workflow:SECRET_TOKEN") -> DeploymentReport:
    return DeploymentReport(
        role="controller", ready=ready, checks=(
            DeploymentCheck(
                name="controller:llama_conn", ready=True, blocking=True,
                detail="private laptop path C:/VERY_SECRET_LOCATION",
            ),
            DeploymentCheck(
                name=bad_name, ready=ready, blocking=True,
                detail="leaked SECRET-BEARER=should-not-surface",
            ),
        ),
    )


def _cached(
    *, state: str = "observed_prerequisites",
    node_id: str = "gpu-b",
) -> RemotePCBPassiveFieldGates:
    return RemotePCBPassiveFieldGates(
        checked_utc=NOW, node_id=node_id, status=state,
        installation_probe_status="observed",
        cached_owner_probe_status="observed",
        cached_vs_live_owner_identity="match",
        authenticated_read_only_gets_attempted=2,
        blockers=("SECRET-CACHED-REPORT-BLOCKER",)
        if state != "observed_prerequisites" else (),
    )


def _live_owner(
    *, status: str = "correlated_read_only", checked: datetime = NOW,
    bearer: bool = True, node_id: str = "gpu-b",
) -> PCBOwnerCorrelation:
    return PCBOwnerCorrelation.model_validate({
        "checked_utc": checked,
        "status": status,
        "reason": "mock authenticated owner",
        "node_id": node_id,
        "remaining_real_machine_evidence": ["real_supervisor_loss"],
        "source_mode": "authenticated_remote",
        "remote_bearer_checked": bearer,
    })


def _compile(
    *,
    installation: TwoPCInstallationAudit | None = None,
    controller: DeploymentReport | None = None,
    cached: RemotePCBPassiveFieldGates | None = None,
    current_owner: PCBOwnerCorrelation | None = None,
) -> PairLiveReadiness:
    return compile_pair_live_readiness(
        node_id="gpu-b",
        installation=_installation() if installation is None else installation,
        controller=_controller() if controller is None else controller,
        cached=_cached() if cached is None else cached,
        current_owner=_live_owner() if current_owner is None else current_owner,
        checked_utc=NOW,
    )


def test_full_live_read_only_prerequisites_are_not_gpu_qualification() -> None:
    result = _compile()
    assert result.mode == "read_only"
    assert result.status == "observed_prerequisites"
    assert result.read_only_prerequisites_observed
    assert result.native_pair_status == "ready_for_passive_monitoring"
    assert result.pc_a_controller_task_status == "running"
    assert result.pc_b_renderer_task_status == "running"
    assert result.pc_b_survival_observer_task_status == "running"
    assert result.pc_b_heartbeat_status == "fresh"
    assert result.controller_services_ready
    assert result.cached_launcher_status == "observed_prerequisites"
    assert result.live_owner_status == "correlated_read_only"
    assert result.cached_vs_live_owner_identity == "match"
    assert result.next_check_role == "pc_a"
    assert result.next_read_only_argv == (
        "uv", "run", "artifex", "qualify", "field-brief", "--json",
    )
    assert result.blockers == ()
    assert result.next_action == "review_remaining_real_machine_qualification"
    assert result.commands_executed is False
    assert result.task_or_process_mutated is False
    assert result.gpu_jobs_submitted is False
    assert result.real_supervisor_loss_proven is False
    assert result.fourteen_stages_verified is False
    assert result.eight_hour_soak_verified is False
    assert result.production_qualified is False
    payload = result.model_dump_json()
    for forbidden in (
        "SECRET-HOSTNAME", "SECRET-CACHED-REPORT-BLOCKER",
        "SECRET_TOKEN", "SECRET-BEARER", "VERY_SECRET_LOCATION", "192.0.2.55",
    ):
        assert forbidden not in payload


@pytest.mark.parametrize(
    ("part", "expected_action"),
    [
        ("native_stopped", "inspect_pc_a_config_and_tasks"),
        ("observer_stale", "inspect_pc_b_native_tasks_and_observer"),
        ("bad_controller", "inspect_controller_or_workflow_dependencies"),
        ("cache_missing", "inspect_pc_b_live_owner_identity"),
        ("foreign_node", "inspect_pc_b_live_owner_identity"),
    ],
)
def test_partial_pair_native_and_workflow_states_fail_closed(
    part: str, expected_action: str,
) -> None:
    installation = _installation(
        status="pc_a_needs_setup" if part == "native_stopped"
        else "pc_b_needs_setup" if part == "observer_stale"
        else "ready_for_passive_monitoring",
        task="registered_not_running" if part == "native_stopped" else "running",
        heartbeat="stale" if part == "observer_stale" else "fresh",
    )
    controller = _controller(ready=part != "bad_controller")
    cached = _cached(
        state="needs_evidence" if part == "cache_missing" else "observed_prerequisites",
        node_id="other" if part == "foreign_node" else "gpu-b",
    )
    result = _compile(
        installation=installation, controller=controller, cached=cached,
    )
    assert result.status == "needs_setup"
    assert result.blockers
    assert result.next_action == expected_action
    assert not result.read_only_prerequisites_observed
    assert not result.production_qualified
    assert "SECRET" not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    "installation", "controller", "cache", "owner",
])
async def test_independent_live_probes_preserve_other_results_and_hide_errors(
    tmp_path: Path, failure: str,
) -> None:
    settings = _settings()
    config = tmp_path / "controller.yaml"
    config.write_text("{}")
    calls: list[str] = []

    def install(
        actual_settings: ArtifexSettings, *, controller_config: Path,
    ) -> TwoPCInstallationAudit:
        assert actual_settings is settings
        assert controller_config == config
        calls.append("install")
        if failure == "installation":
            raise ValueError("SECRET-BEARER C:/private unsafe")
        return _installation()

    async def controller(
        actual_settings: ArtifexSettings, **kwargs: Any,
    ) -> DeploymentReport:
        assert actual_settings is settings
        assert kwargs == {
            "role": "controller", "require_autostart": True,
            "render_smoke": False,
        }
        calls.append("controller")
        if failure == "controller":
            raise OSError("SECRET-BEARER C:/private unsafe")
        return _controller()

    def owner(actual_settings: ArtifexSettings) -> PCBOwnerCorrelation:
        assert actual_settings is settings
        calls.append("owner")
        if failure == "owner":
            raise RuntimeError("SECRET-BEARER C:/private unsafe")
        return _live_owner()

    def cache(
        actual_settings: ArtifexSettings, *, current_owner: PCBOwnerCorrelation | None,
    ) -> RemotePCBPassiveFieldGates:
        assert actual_settings is settings
        calls.append("cache")
        if failure == "cache":
            raise RuntimeError("SECRET-BEARER C:/private unsafe")
        return _cached()

    result = await collect_pair_live_readiness(
        settings, controller_config=config,
        installation_fn=install, controller_fn=controller,
        owner_fn=owner, cached_fn=cache,
    )
    assert calls == ["install", "controller", "owner", "cache"]
    assert result.status == "probe_unavailable"
    assert not result.read_only_prerequisites_observed
    assert result.commands_executed is False
    assert result.gpu_jobs_submitted is False
    assert result.production_qualified is False
    assert "SECRET-BEARER" not in result.model_dump_json()
    assert "C:/private" not in result.model_dump_json()
    for name, observed in (
        ("installation_probe", failure != "installation"),
        ("controller_probe", failure != "controller"),
        ("cached_launcher_probe", failure != "cache"),
        ("live_owner_probe", failure != "owner"),
    ):
        assert getattr(result, name) == (
            "observed" if observed else "unavailable"
        )


@pytest.mark.parametrize(
    ("status", "bearer", "age", "node", "expected_code"),
    [
        ("correlated_read_only", False, 0, "gpu-b", "pc_b_live_owner_unverified"),
        ("unavailable", True, 0, "gpu-b", "pc_b_live_owner_unverified"),
        ("correlated_read_only", True, 180, "gpu-b", "pc_b_live_owner_unverified"),
        ("correlated_read_only", True, 0, "foreign", "pc_b_live_owner_unverified"),
    ],
)
def test_live_owner_never_passes_if_not_fresh_bearer_correlated(
    status: str, bearer: bool, age: int, node: str, expected_code: str,
) -> None:
    owner = _live_owner(
        status=status, bearer=bearer,
        checked=NOW - timedelta(seconds=age),
        node_id=node,
    )
    report = _compile(current_owner=owner)
    assert report.status == "needs_setup"
    assert expected_code in report.blockers
    assert not report.read_only_prerequisites_observed
    assert report.next_action == "inspect_pc_b_live_owner_identity"
    assert not report.production_qualified


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ("conflict", "pc_b_cached_owner_identity_conflict"),
        ("not_comparable", "pc_b_cached_owner_identity_not_correlated"),
    ],
)
def test_cached_owner_must_match_live_process_identity(
    identity: str, expected: str,
) -> None:
    cached = _cached().model_copy(update={
        "cached_vs_live_owner_identity": identity,
    })
    report = _compile(cached=cached)
    assert report.status == "needs_setup"
    assert expected in report.blockers
    assert report.next_action == "inspect_pc_b_live_owner_identity"
    assert not report.read_only_prerequisites_observed


def test_next_check_is_real_read_only_cli_on_correct_host() -> None:
    variants = (
        _compile(installation=_installation(
            status="pc_a_needs_setup", task="registered_not_running",
        )),
        _compile(installation=_installation(
            status="pc_b_needs_setup", heartbeat="stale",
        )),
        _compile(controller=_controller(ready=False)),
        _compile(cached=_cached().model_copy(update={
            "cached_vs_live_owner_identity": "not_comparable",
        })),
        _compile(),
    )
    expected_hosts = ("pc_a", "pc_b", "pc_a", "pc_a", "pc_a")
    for report, host in zip(variants, expected_hosts, strict=True):
        assert report.next_check_role == host
        argv = report.next_read_only_argv
        assert argv[:3] == ("uv", "run", "artifex")
        assert argv[-1] == "--json"
        assert not set(argv).intersection({
            "--apply", "--replace", "--confirm-render",
            "--render-smoke", "--save", "--output", "--install-missing",
            "--release",
        })
        assert "--config" not in argv
        assert not report.task_or_process_mutated
        assert not report.gpu_jobs_submitted


def test_controller_failure_codes_are_fixed_and_omit_untrusted_check_names() -> None:
    result = _compile(controller=_controller(ready=False))
    assert result.failed_controller_areas == ("controller_workflow_dependencies",)
    assert "SECRET" not in json.dumps(result.model_dump(mode="json"))


def test_pair_live_cli_never_activates_and_returns_failure_for_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("{}")
    observed: list[tuple[Path, str | None]] = []

    async def collector(
        settings: ArtifexSettings, *, controller_config: Path,
    ) -> PairLiveReadiness:
        observed.append((
            controller_config,
            settings.render_nodes.primary,
        ))
        # Real CLI reads a valid config but test never contacts network.
        return compile_pair_live_readiness(
            node_id=None, installation=None, controller=None, cached=None,
        )

    monkeypatch.setattr("artifex.cli.collect_pair_live_readiness", collector)
    runner = CliRunner()
    result = runner.invoke(
        app, ["deployment", "pair-live", "--config", str(config), "--json"],
    )
    assert result.exit_code == 1, result.output
    assert observed and observed[0][0] == config
    data = json.loads(result.stdout)
    assert data["status"] == "probe_unavailable"
    assert data["commands_executed"] is False
    assert data["gpu_jobs_submitted"] is False
    assert data["production_qualified"] is False

    no_config = runner.invoke(
        app, [
            "deployment", "pair-live",
            "--config", str(tmp_path / "absent.yaml"), "--json",
        ],
    )
    assert no_config.exit_code != 0
    assert len(observed) == 1
