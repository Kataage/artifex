"""One PC-A, fileless, GET-only two-PC bring-up diagnosis.

No copied PC-B report is required. This is an observational prerequisite
inventory, not permission to activate tasks, submit a GPU prompt, or mark
qualification stages as passed. Deliberately omit raw remote payloads,
endpoint URLs, exception messages, executable paths and credentials.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.deployment import DeploymentReport, verify_deployment
from artifex.qualification.owner_handoff import (
    PCBOwnerCorrelation,
    correlate_live_pc_b_owner_report,
)
from artifex.qualification.pair_installation import (
    TwoPCInstallationAudit,
    inspect_two_pc_installation,
)
from artifex.qualification.pc_b_remote_field_gates import (
    RemotePCBPassiveFieldGates,
    inspect_remote_pc_b_passive_gates,
)

ProbeState = Literal["observed", "unavailable"]
ReadinessState = Literal[
    "observed_prerequisites", "needs_setup", "probe_unavailable",
]
NextAction = Literal[
    "inspect_pc_a_config_and_tasks",
    "restore_pc_b_authenticated_diagnostics",
    "inspect_pc_b_native_tasks_and_observer",
    "inspect_controller_or_workflow_dependencies",
    "inspect_pc_b_cached_launcher_evidence",
    "review_remaining_real_machine_qualification",
    "inspect_pc_b_live_owner_identity",
]
SafeRole = Literal["pc_a", "pc_b"]
_MAX_OWNER_AGE = timedelta(minutes=2)

# Fixed local argv only: no secrets, configured URLs, PIDs or PC-B paths.
# Arguments are for an operator to run on the indicated machine, NEVER
# executed by pair-live. --config may be supplied locally if non-default.
_NEXT_CHECKS: dict[NextAction, tuple[SafeRole, tuple[str, ...]]] = {
    "inspect_pc_a_config_and_tasks": (
        "pc_a", ("uv", "run", "artifex", "startup", "audit",
                 "--role", "controller", "--json"),
    ),
    "restore_pc_b_authenticated_diagnostics": (
        "pc_a", ("uv", "run", "artifex", "onboard", "inspect",
                 "--role", "controller", "--json"),
    ),
    "inspect_pc_b_native_tasks_and_observer": (
        "pc_b", ("uv", "run", "artifex", "deployment", "activate",
                 "--role", "renderer", "--with-observer", "--json"),
    ),
    "inspect_controller_or_workflow_dependencies": (
        "pc_a", ("uv", "run", "artifex", "deployment", "verify",
                 "--role", "controller", "--require-autostart", "--json"),
    ),
    "inspect_pc_b_cached_launcher_evidence": (
        "pc_a", ("uv", "run", "artifex", "qualify", "field-brief", "--json"),
    ),
    "inspect_pc_b_live_owner_identity": (
        "pc_a", ("uv", "run", "artifex", "qualify", "field-brief", "--json"),
    ),
    "review_remaining_real_machine_qualification": (
        "pc_a", ("uv", "run", "artifex", "qualify", "field-brief", "--json"),
    ),
}



class PairLiveReadiness(BaseModel):
    """Safe output only: no serializable PC-B raw diagnostic objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checked_utc: datetime
    mode: Literal["read_only"] = "read_only"
    status: ReadinessState
    node_id: str | None
    installation_probe: ProbeState
    native_pair_status: str
    pc_a_controller_task_status: str
    pc_b_renderer_task_status: str
    pc_b_survival_observer_task_status: str
    pc_b_heartbeat_status: str
    controller_probe: ProbeState
    controller_services_ready: bool
    failed_controller_areas: tuple[str, ...]
    cached_launcher_probe: ProbeState
    cached_launcher_status: str
    live_owner_probe: ProbeState
    live_owner_status: str
    cached_vs_live_owner_identity: Literal["not_comparable", "match", "conflict"]
    blockers: tuple[str, ...]
    next_action: NextAction
    next_check_role: SafeRole
    next_read_only_argv: tuple[str, ...]
    read_only_prerequisites_observed: bool
    remote_report_copied_by_operator: Literal[False] = False
    commands_executed: Literal[False] = False
    task_or_process_mutated: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    discord_messages_sent: Literal[False] = False
    real_supervisor_loss_proven: Literal[False] = False
    fourteen_stages_verified: Literal[False] = False
    eight_hour_soak_verified: Literal[False] = False
    production_qualified: Literal[False] = False


def _controller_failures(report: DeploymentReport) -> tuple[str, ...]:
    """Convert untrusted check names to bounded, fixed categories."""
    failures: list[str] = []
    for item in report.checks:
        if not item.blocking or item.ready:
            continue
        if item.name == "windows_autostart":
            code = "controller_task_registration_or_state"
        elif item.name.startswith("workflow:"):
            code = "controller_workflow_dependencies"
        elif item.name.startswith("controller:"):
            code = "controller_runtime_preflight"
        else:
            code = "controller_other_required_service"
        if code not in failures:
            failures.append(code)
    if not report.ready and not failures:
        failures.append("controller_unverified_service")
    return tuple(failures)


def compile_pair_live_readiness(
    *,
    node_id: str | None,
    installation: TwoPCInstallationAudit | None,
    controller: DeploymentReport | None,
    cached: RemotePCBPassiveFieldGates | None,
    current_owner: PCBOwnerCorrelation | None = None,
    checked_utc: datetime | None = None,
) -> PairLiveReadiness:
    """Combine independent snapshots conservatively, without side effects."""
    blockers: list[str] = []
    pair_probe: ProbeState = "observed" if installation else "unavailable"
    controller_probe: ProbeState = "observed" if controller else "unavailable"
    cached_probe: ProbeState = "observed" if cached else "unavailable"
    owner_probe: ProbeState = "observed" if current_owner else "unavailable"
    now = checked_utc or datetime.now(UTC)

    pair_status = installation.status if installation else "unavailable"
    task_a = (
        installation.pc_a_controller_task.status if installation else "unavailable"
    )
    remote = installation.pc_b.audit if installation and installation.pc_b else None
    task_b = remote.renderer_task.status if remote else "unavailable"
    watcher = remote.survival_observer_task.status if remote else "unavailable"
    heartbeat = remote.observer_heartbeat if remote else "unavailable"
    if installation is None:
        blockers.append("pc_a_or_pc_b_installation_audit_unavailable")
    else:
        # These are fixed category codes from the native pair audit.
        blockers.extend(installation.blockers)
        if installation.pc_b_node_id != node_id:
            blockers.append("pair_primary_node_mismatch")
        if installation.status != "ready_for_passive_monitoring" and not blockers:
            blockers.append("pair_native_prerequisites_unverified")
    failed_areas = _controller_failures(controller) if controller else ()
    if controller is None:
        blockers.append("controller_live_preflight_unavailable")
    else:
        blockers.extend(failed_areas)
    cached_status = cached.status if cached else "unavailable"
    if cached is None:
        blockers.append("pc_b_cached_evidence_probe_unavailable")
    else:
        # The additional independent protected GETs are not a single atomic
        # PC-B snapshot. A node mismatch must prevent false readiness.
        if cached.node_id != node_id:
            blockers.append("pc_b_cached_evidence_node_mismatch")
        if cached.status != "observed_prerequisites":
            blockers.append("pc_b_cached_evidence_unverified")
        # Do not expose arbitrary report text even if a future GET grows.
        if cached.cached_vs_live_owner_identity == "conflict":
            blockers.append("pc_b_cached_owner_identity_conflict")

    owner_status = current_owner.status if current_owner else "unavailable"
    owner_verified = bool(
        current_owner is not None
        and current_owner.status == "correlated_read_only"
        and current_owner.source_mode == "authenticated_remote"
        and current_owner.remote_bearer_checked
        and current_owner.node_id == node_id
        and current_owner.checked_utc.tzinfo is not None
        and -timedelta(seconds=30) <= (
            now - current_owner.checked_utc
        ) <= _MAX_OWNER_AGE
    )
    agreement = (
        cached.cached_vs_live_owner_identity if cached is not None
        else "not_comparable"
    )
    if current_owner is None:
        blockers.append("pc_b_live_owner_probe_unavailable")
    elif not owner_verified:
        blockers.append("pc_b_live_owner_unverified")
    if agreement == "conflict":
        blockers.append("pc_b_cached_owner_identity_conflict")
    elif agreement != "match":
        blockers.append("pc_b_cached_owner_identity_not_correlated")

    complete = bool(
        installation is not None
        and installation.status == "ready_for_passive_monitoring"
        and installation.pc_b_node_id == node_id
        and controller is not None and controller.ready
        and cached is not None
        and cached.node_id == node_id
        and cached.status == "observed_prerequisites"
        and owner_verified
        and agreement == "match"
        and not blockers
    )
    if complete:
        status: ReadinessState = "observed_prerequisites"
        action: NextAction = "review_remaining_real_machine_qualification"
    elif (
        installation is None or controller is None or cached is None
        or current_owner is None
    ):
        status = "probe_unavailable"
        if installation is None:
            action = "inspect_pc_a_config_and_tasks"
        elif installation.status in {"pc_b_unreachable", "pc_b_unconfigured"}:
            action = "restore_pc_b_authenticated_diagnostics"
        elif controller is None:
            action = "inspect_controller_or_workflow_dependencies"
        elif current_owner is None:
            action = "restore_pc_b_authenticated_diagnostics"
        else:
            action = "inspect_pc_b_cached_launcher_evidence"
    else:
        status = "needs_setup"
        if installation.status in {"pc_b_unreachable", "pc_b_unconfigured"}:
            action = "restore_pc_b_authenticated_diagnostics"
        elif installation.pc_a_controller_task.status != "running":
            action = "inspect_pc_a_config_and_tasks"
        elif installation.status != "ready_for_passive_monitoring":
            action = "inspect_pc_b_native_tasks_and_observer"
        elif not controller.ready:
            action = "inspect_controller_or_workflow_dependencies"
        elif (
            cached.status != "observed_prerequisites"
            or cached.node_id != node_id
        ):
            action = "inspect_pc_b_cached_launcher_evidence"
        elif not owner_verified or agreement != "match":
            action = "inspect_pc_b_live_owner_identity"
        else:
            action = "inspect_pc_b_cached_launcher_evidence"

    return PairLiveReadiness(
        checked_utc=now,
        status=status,
        node_id=node_id,
        installation_probe=pair_probe,
        native_pair_status=pair_status,
        pc_a_controller_task_status=task_a,
        pc_b_renderer_task_status=task_b,
        pc_b_survival_observer_task_status=watcher,
        pc_b_heartbeat_status=heartbeat,
        controller_probe=controller_probe,
        controller_services_ready=bool(controller and controller.ready),
        failed_controller_areas=failed_areas,
        cached_launcher_probe=cached_probe,
        cached_launcher_status=cached_status,
        live_owner_probe=owner_probe,
        live_owner_status=owner_status,
        cached_vs_live_owner_identity=agreement,
        blockers=tuple(dict.fromkeys(blockers)),
        next_action=action,
        next_check_role=_NEXT_CHECKS[action][0],
        next_read_only_argv=_NEXT_CHECKS[action][1],
        read_only_prerequisites_observed=complete,
    )


async def collect_pair_live_readiness(
    settings: ArtifexSettings, *,
    controller_config: Path,
    installation_fn: Callable[..., TwoPCInstallationAudit] = inspect_two_pc_installation,
    controller_fn: Callable[..., Awaitable[DeploymentReport]] = verify_deployment,
    cached_fn: Callable[..., RemotePCBPassiveFieldGates] = inspect_remote_pc_b_passive_gates,
    owner_fn: Callable[..., PCBOwnerCorrelation] = correlate_live_pc_b_owner_report,
) -> PairLiveReadiness:
    """PC-A only: sequential, bounded existing GETs, never launch anything."""
    chosen = settings.render_nodes.primary_node()
    node_id = chosen[0] if chosen else None
    installation: TwoPCInstallationAudit | None = None
    controller: DeploymentReport | None = None
    cached: RemotePCBPassiveFieldGates | None = None
    current_owner: PCBOwnerCorrelation | None = None

    # Each independent observation must still be reported if another one
    # fails. Do not serialize exception strings containing remote URLs/tokens.
    try:
        installation = await asyncio.to_thread(
            installation_fn, settings, controller_config=controller_config,
        )
    except Exception:  # noqa: BLE001 - status-only diagnosis, no secret echo
        installation = None
    try:
        controller = await controller_fn(
            settings, role="controller", require_autostart=True,
            render_smoke=False,
        )
    except Exception:  # noqa: BLE001 - preserve other GET observations
        controller = None
    # The native cache is not live process identity. The current,
    # independently Bearer-correlated owner must be sampled separately.
    try:
        current_owner = await asyncio.to_thread(owner_fn, settings)
    except Exception:  # noqa: BLE001 - omit remote paths and error details
        current_owner = None
    try:
        cached = await asyncio.to_thread(
            cached_fn, settings, current_owner=current_owner,
        )
    except Exception:  # noqa: BLE001 - do not discard valid native/LLM checks
        cached = None
    return compile_pair_live_readiness(
        node_id=node_id,
        installation=installation, controller=controller, cached=cached,
        current_owner=current_owner,
    )
