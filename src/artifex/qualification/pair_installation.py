"""PC-A read-only installation readiness inventory for the two Windows PCs."""
from __future__ import annotations

import platform
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.render_node.client import fetch_remote_installation_audit
from artifex.render_node.installation_audit import (
    RemoteRendererInstallationAudit,
    TaskInstallationCheck,
    inspect_registered_task,
)
from artifex.windows_tasks import (
    StartupRole,
    StartupTaskStatus,
    task_configuration_matches,
    task_status,
)

Overall = Literal[
    "ready_for_passive_monitoring", "pc_a_needs_setup",
    "pc_b_needs_setup", "pc_b_unreachable", "pc_b_unconfigured",
    "unsupported_platform",
]


class TwoPCInstallationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checked_utc: datetime
    status: Overall
    pc_a_controller_task: TaskInstallationCheck
    pc_b_node_id: str | None
    pc_b: RemoteRendererInstallationAudit | None = None
    blockers: tuple[str, ...]
    next_actions: tuple[str, ...]
    commands_executed: Literal[False] = False
    local_files_modified: Literal[False] = False
    remote_task_actions_executed: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    actual_survival_observed: Literal[False] = False
    real_machine_qualification_complete: Literal[False] = False
    production_qualified: Literal[False] = False


def inspect_two_pc_installation(
    settings: ArtifexSettings,
    *,
    controller_config: Path,
    now: datetime | None = None,
    native_windows: bool | None = None,
    local_probe: Callable[[StartupRole], StartupTaskStatus] = task_status,
    local_verify: Callable[..., tuple[bool, str]] = task_configuration_matches,
    remote_probe: Callable[
        [str, RenderNodeConfig], RemoteRendererInstallationAudit
    ] = fetch_remote_installation_audit,
) -> TwoPCInstallationAudit:
    """Probe only configured read-only endpoints, never attempt repairs."""
    tick = now or datetime.now(UTC)
    native = platform.system() == "Windows" if native_windows is None else native_windows
    config_safe = controller_config.is_file() and not any(
        part.is_symlink() for part in (controller_config, *controller_config.parents)
    )
    local = inspect_registered_task(
        "controller", config=controller_config,
        native_windows=native and config_safe,
        probe=local_probe, verify=local_verify,
    )
    blockers: list[str] = []
    advice: list[str] = []
    if not native:
        blockers.append("pc_a_native_windows_required")
        advice.append("Run this command on the actual Windows PC-A.")
    if not config_safe:
        blockers.append("pc_a_controller_config_missing_or_unsafe")
        advice.append("Select an existing non-symlinked PC-A YAML with --config.")
    if local.status != "running":
        blockers.append("pc_a_controller_task_" + local.status)
        advice.append(
            "Audit the existing PC-A Task Scheduler controller registration; "
            "do not replace a running task."
        )
    chosen = settings.render_nodes.primary_node()
    if chosen is None:
        return TwoPCInstallationAudit(
            checked_utc=tick, status="pc_b_unconfigured",
            pc_a_controller_task=local, pc_b_node_id=None,
            blockers=(*blockers, "pc_b_primary_node_unconfigured"),
            next_actions=(*advice, "Configure render_nodes.primary and its attestation URL."),
        )
    node_id, config = chosen
    if not config.attestation_url or not config.attestation_token_env:
        return TwoPCInstallationAudit(
            checked_utc=tick, status="pc_b_unconfigured",
            pc_a_controller_task=local, pc_b_node_id=node_id,
            blockers=(*blockers, "pc_b_attestation_or_bearer_unconfigured"),
            next_actions=(*advice, "Configure PC-B's protected attestation URL and token variable."),
        )
    try:
        response = remote_probe(node_id, config)
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError):
        return TwoPCInstallationAudit(
            checked_utc=tick, status="pc_b_unreachable",
            pc_a_controller_task=local, pc_b_node_id=node_id,
            blockers=(*blockers, "pc_b_authenticated_installation_unavailable"),
            next_actions=(
                *advice,
                (
                    "Check PC-B attestation availability, network and Bearer environment; "
                    "do not start or restart ComfyUI merely to enable this audit."
                ),
            ),
        )
    detail = response.audit
    if (
        response.node_id != node_id or detail.node_id != node_id
        or detail.captured_utc.tzinfo is None
        or not -timedelta(seconds=30) <= tick - detail.captured_utc <= timedelta(minutes=2)
    ):
        return TwoPCInstallationAudit(
            checked_utc=tick, status="pc_b_needs_setup",
            pc_a_controller_task=local, pc_b_node_id=node_id, pc_b=response,
            blockers=(*blockers, "pc_b_node_identity_or_snapshot_stale"),
            next_actions=(
                *advice, "Inspect PC-B node ID and clock; do not trust mismatched/stale reports.",
            ),
        )
    if not detail.native_windows:
        blockers.append("pc_b_not_native_windows")
    if not detail.managed_renderer_configured:
        blockers.append("pc_b_managed_renderer_not_configured")
    for role, snapshot in (
        ("renderer", detail.renderer_task),
        ("survival_observer", detail.survival_observer_task),
    ):
        if snapshot.status != "running" or not snapshot.configuration_verified:
            blockers.append("pc_b_" + role + "_" + snapshot.status)
    if detail.evidence_spool in {"unsafe", "unavailable"}:
        blockers.append("pc_b_survival_spool_" + detail.evidence_spool)
    if not detail.safe_for_passive_observation:
        blockers.append("pc_b_passive_observer_preconditions_unverified")
    if detail.evidence_spool in {"empty", "missing"}:
        # An empty spool isn't a failed installation but is still evidence
        # missing from real-machine qualification.
        advice.append(
            "No native survival event evidence yet; do not terminate the renderer to force one."
        )
    if any(value.startswith("pc_b_") for value in blockers):
        advice.append(
            "On PC-B, inspect 'uv run artifex startup observer-enable "
            "--config config/render-node.yaml --json'; its default is dry-run."
        )
    status: Overall = (
        "unsupported_platform" if not native else
        "pc_a_needs_setup" if local.status != "running" else
        "pc_b_needs_setup" if blockers else
        "ready_for_passive_monitoring"
    )
    return TwoPCInstallationAudit(
        checked_utc=tick, status=status,
        pc_a_controller_task=local, pc_b_node_id=node_id, pc_b=response,
        blockers=tuple(dict.fromkeys(blockers)),
        next_actions=tuple(dict.fromkeys(advice)),
    )
