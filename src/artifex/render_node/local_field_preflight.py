"""One PC-B native Windows field inspection with no production GPU actions.

The sole optional subprocess is the *existing* disposable local Python
launcher fixture, not ComfyUI. The report contains no executable paths,
command lines, tokens, LAN URLs or raw exception messages. Even a perfect
observation cannot qualify natural supervisor loss or production GPU work.
"""
from __future__ import annotations

import platform
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.qualification.native_creation_time import native_creation_instant
from artifex.render_node.installation_audit import (
    RendererInstallationAudit,
    inspect_renderer_installation,
)
from artifex.render_node.owner_readiness import collect_owner_readiness

Status = Literal["passive_observation_ready", "needs_evidence", "unsupported"]
NextHost = Literal["pc_a", "pc_b"]


class LocalRendererFieldPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    observed_utc: datetime
    node_id: str
    status: Status
    native_windows: bool
    installation_inspected: bool
    owner_readiness_inspected: bool
    renderer_task_status: str
    observer_task_status: str
    observer_heartbeat_status: str
    survival_spool_status: str
    owner_readiness_status: str
    launcher_fixture_status: str
    real_comfyui_owner_status: str
    listener_pid: int | None = None
    listener_started_utc: str | None = None
    blocked_reasons: tuple[str, ...]
    probe_errors: tuple[str, ...]
    next_host: NextHost
    next_action: str
    next_safe_argv: tuple[str, ...] | None
    # A fixture is a short-lived local Python/TCP process. Never production.
    disposable_no_gpu_probe_attempted: bool
    existing_comfyui_started: Literal[False] = False
    existing_comfyui_stopped: Literal[False] = False
    scheduler_actions_executed: Literal[False] = False
    production_gpu_jobs_submitted: Literal[False] = False
    historical_supervisor_exit_authenticated: Literal[False] = False
    production_reattachment_qualified: Literal[False] = False
    fourteen_stage_qualification_verified: Literal[False] = False
    eight_hour_soak_verified: Literal[False] = False
    issue_93_closure_authorized: Literal[False] = False
    issue_40_closure_authorized: Literal[False] = False
    production_qualified: Literal[False] = False


def compile_local_renderer_field_preflight(
    settings: ArtifexSettings, *, config: Path,
    installation: RendererInstallationAudit | None,
    owner_readiness: dict[str, Any] | None,
    native_windows: bool,
    probe_errors: tuple[str, ...] = (),
    observed_utc: datetime | None = None,
) -> LocalRendererFieldPreflight:
    """Require both independent local observations; never infer stage PASS."""
    renderer_state = (
        installation.renderer_task.status if installation else "unavailable"
    )
    watcher_state = (
        installation.survival_observer_task.status
        if installation else "unavailable"
    )
    heartbeat = installation.observer_heartbeat if installation else "unavailable"
    spool = installation.evidence_spool if installation else "unavailable"
    node = settings.render_agent.node_id
    local_owner: dict[str, Any] = {}
    local_checks: dict[str, Any] = {}
    if isinstance(owner_readiness, dict):
        audit = owner_readiness.get("owner_audit")
        if isinstance(audit, dict):
            local_owner = audit
        checks = owner_readiness.get("checks")
        if isinstance(checks, dict):
            local_checks = checks

    def check_status(name: str) -> str:
        check = local_checks.get(name)
        if not isinstance(check, dict):
            return "not_observed"
        value = check.get("status")
        return value if value in {"pass", "fail"} else "not_observed"

    fixture_status = check_status("disposable_launcher_fixture")
    owner_status = check_status("live_comfyui_owner")
    pid = local_owner.get("actual_listener_pid")
    safe_owner_state = owner_readiness.get("status") if isinstance(
        owner_readiness, dict,
    ) else None
    parsed_start = native_creation_instant(
        local_owner.get("actual_process_started_utc"),
    )
    pid_valid = isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
    owner_verified = bool(
        owner_readiness is not None
        and safe_owner_state == "observed_independently"
        and isinstance(owner_readiness.get("host"), str)
        and installation is not None
        and owner_readiness["host"].casefold() == installation.hostname.casefold()
        and fixture_status == "pass"
        and owner_status == "pass"
        and local_owner.get("status") == "observed_stable"
        and pid_valid
        and parsed_start is not None
        and installation is not None
        and installation.captured_utc.tzinfo is not None
        and parsed_start <= installation.captured_utc
    )
    blockers: list[str] = []
    if not native_windows:
        blockers.append("local_native_windows_required")
    if installation is None:
        blockers.append("pc_b_installation_probe_unavailable")
    else:
        if installation.node_id != node:
            blockers.append("pc_b_node_identity_mismatch")
        if not installation.safe_for_passive_observation:
            blockers.append("pc_b_task_or_observer_preconditions_not_ready")
        if watcher_state != "running" or heartbeat != "fresh":
            blockers.append("pc_b_survival_observer_not_fresh")
        if spool in {"unsafe", "unavailable"}:
            blockers.append("pc_b_survival_spool_unsafe_or_unavailable")
    if owner_readiness is None:
        blockers.append("pc_b_owner_readiness_unavailable")
    elif not owner_verified:
        blockers.append("pc_b_disposable_launcher_or_live_owner_not_verified")
    if probe_errors:
        blockers.append("pc_b_one_or_more_local_diagnostics_failed")

    ready = native_windows and not blockers
    state: Status = (
        "unsupported" if not native_windows
        else "passive_observation_ready" if ready else "needs_evidence"
    )
    if ready:
        next_host: NextHost = "pc_a"
        next_action = (
            "Run the independent authenticated PC-A field-preflight, then "
            "review remaining native supervisor-loss and GPU qualification evidence."
        )
        cmd: tuple[str, ...] | None = None
    else:
        next_host = "pc_b"
        next_action = (
            "Review the reported local PC-B blockers and the observer task dry-run. "
            "Do not force a supervisor exit or restart existing ComfyUI."
        )
        cmd = (
            "uv", "run", "artifex", "startup", "observer-enable",
            "--config", str(config), "--json",
        ) if native_windows and installation is not None else None
    return LocalRendererFieldPreflight(
        observed_utc=observed_utc or datetime.now(UTC),
        node_id=node, status=state, native_windows=native_windows,
        installation_inspected=installation is not None,
        owner_readiness_inspected=owner_readiness is not None,
        renderer_task_status=renderer_state,
        observer_task_status=watcher_state,
        observer_heartbeat_status=heartbeat,
        survival_spool_status=spool,
        owner_readiness_status=(
            safe_owner_state if isinstance(safe_owner_state, str)
            and safe_owner_state in {"observed_independently", "blocked", "unsupported"}
            else "unavailable"
        ),
        launcher_fixture_status=fixture_status,
        real_comfyui_owner_status=owner_status,
        listener_pid=pid if owner_verified else None,
        listener_started_utc=parsed_start.isoformat() if owner_verified else None,
        blocked_reasons=tuple(dict.fromkeys(blockers)),
        probe_errors=probe_errors,
        next_host=next_host, next_action=next_action, next_safe_argv=cmd,
        disposable_no_gpu_probe_attempted=native_windows,
    )


def inspect_local_renderer_field_preflight(
    settings: ArtifexSettings, *, config: Path,
    native_windows: bool | None = None,
    installation_fn: Callable[..., RendererInstallationAudit] = (
        inspect_renderer_installation
    ),
    owner_fn: Callable[..., dict[str, Any]] = collect_owner_readiness,
) -> LocalRendererFieldPreflight:
    """Serially collect local task/heartbeat and disposable Python owner proof.

    A failure in one probe does not prevent the other. Never use a configured
    ComfyUI command as the test fixture, and never start or stop production GPU.
    """
    native = platform.system() == "Windows" if native_windows is None else native_windows
    installation: RendererInstallationAudit | None = None
    owner: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        installation = installation_fn(
            settings, owner_config=config, native_windows=native,
        )
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        errors.append(f"installation:{type(exc).__name__}")
    if native:
        try:
            owner = owner_fn(settings, config=config)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            errors.append(f"owner_readiness:{type(exc).__name__}")
    return compile_local_renderer_field_preflight(
        settings, config=config, installation=installation,
        owner_readiness=owner, native_windows=native,
        probe_errors=tuple(errors),
    )
