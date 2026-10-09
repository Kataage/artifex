"""Read-only PC-B evidence reconciliation; mock proof is never GPU authority."""
from __future__ import annotations

import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifex.config.models import ArtifexSettings
from artifex.render_node.native_launcher_probe import inspect_native_python_listener
from artifex.render_node.owner_audit import observe_renderer_owner

_REQUIRED_CHECKS = frozenset({
    "native_windows", "protected_configuration", "scheduler_policy",
    "scheduler_running", "receipt", "process_identity", "launcher_identity",
    "tcp_ownership", "snapshot_consistency",
})
_ALWAYS_MISSING = (
    "real_comfyui_child_survival_after_supervisor_loss",
    "real_gpu_production_workflow_result",
    "eight_hour_unattended_gpu_soak",
    "all_fourteen_real_machine_qualification_stages",
)


def collect_owner_readiness(
    settings: ArtifexSettings, *, config: Path,
) -> dict[str, Any]:
    """Observe a disposable launcher and separately observe live ComfyUI.

    This never links the mock PID to production, nor authorizes a restart,
    process adoption, production GPU work, or a qualification-stage PASS.
    """
    exe = settings.render_agent.comfyui_process.executable
    fixture: dict[str, Any] | None = None
    fixture_reason = "configured_comfyui_python_missing"
    if exe is not None:
        try:
            result = inspect_native_python_listener(python=exe)
            fixture = result.model_dump(mode="json")
            fixture_reason = result.reason
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            # Never silently fall back to the Artifex interpreter.
            fixture_reason = f"configured_python_probe_failed:{type(exc).__name__}"

    # Observe the real owner *after* the fixture for a later live sample.
    owner = observe_renderer_owner(settings, config=config)
    host = socket.gethostname()
    fixture_ok = bool(
        fixture is not None
        and fixture.get("status") == "observed"
        and fixture.get("host") == host
        and fixture.get("verified_runtime_provenance") is True
        and fixture.get("original_launcher_verified") is True
        and fixture.get("listener_identity_stable") is True
        and fixture.get("tcp_owner_stable") is True
        and fixture.get("test_listener_only") is True
        and fixture.get("actual_comfyui_inspected") is False
        and fixture.get("renderer_restart_authorized") is False
        and fixture.get("real_machine_gpu_qualified") is False
    )
    checks = owner.get("checks")
    owner_ok = bool(
        owner.get("status") == "observed_stable"
        and owner.get("process_observation_verified") is True
        and isinstance(owner.get("actual_listener_pid"), int)
        and isinstance(checks, dict)
        and _REQUIRED_CHECKS.issubset(checks)
        and all(
            isinstance(checks[key], dict) and checks[key].get("status") == "pass"
            for key in _REQUIRED_CHECKS
        )
        and owner.get("restart_authorized") is False
        and owner.get("production_qualified") is False
    )
    missing = []
    if not fixture_ok:
        missing.append("native_launcher_fixture_on_actual_pc_b")
    if not owner_ok:
        missing.append("live_managed_comfyui_receipt_scheduler_tcp_identity")
    missing.extend(_ALWAYS_MISSING)
    status = (
        "unsupported" if owner.get("status") == "unsupported"
        else "observed_independently" if fixture_ok and owner_ok
        else "blocked"
    )
    return {
        "schema_version": 1,
        "collected_utc": datetime.now(UTC).isoformat(),
        "host": host,
        "status": status,
        "configured_comfyui_python": str(exe) if exe is not None else None,
        "checks": {
            "disposable_launcher_fixture": {
                "status": "pass" if fixture_ok else "fail",
                "reason": "configured_python_launcher_fixture_verified"
                if fixture_ok else fixture_reason,
            },
            "live_comfyui_owner": {
                "status": "pass" if owner_ok else "fail",
                "reason": "actual_renderer_owner_observed_stable" if owner_ok
                else f"actual_renderer_owner_{owner.get('status', 'unknown')}",
            },
        },
        "launcher_fixture": fixture,
        "owner_audit": owner,
        "remaining_real_machine_evidence": missing,
        "observations_are_independent": True,
        "actual_comfyui_inspected_by_owner_audit": owner_ok,
        "production_child_survival_qualified": False,
        "real_machine_gpu_qualified": False,
        "issue_93_closure_authorized": False,
        "issue_40_closure_authorized": False,
        "renderer_restart_authorized": False,
        "gpu_jobs_submitted": False,
        "production_qualified": False,
        "mutated_services": False,
    }
