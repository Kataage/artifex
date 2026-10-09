"""One-command read-only PC-A/PC-B qualification readiness diagnostics.

This is a deployment *preflight*, not the 14-stage production qualification.
It does not start processes, write to the database, or issue GPU requests.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.controller_preflight import ControllerPreflight, check_controller
from artifex.qualification.models import (
    REQUIRED_STAGES,
    QualificationSession,
    QualificationStatus,
)
from artifex.qualification.renderer_owner_evidence import _REQUIRED_CHECKS
from artifex.render_node.client import fetch_renderer_owner_audit
from artifex.render_node.models import RemoteRendererOwnerAudit

Target = Literal["pc_a", "pc_b", "qualification"]
Status = Literal["pass", "fail", "unknown"]

_OWNER_ACTIONS = {
    "native_windows": "Run PC-B on native Windows, not WSL or Linux.",
    "protected_configuration": "Inspect PC-B render_agent.gateway and comfyui_process configuration; never restart a live GPU process to fix it.",
    "scheduler_policy": "On PC-B, use 'artifex startup audit --role renderer --config <PC-B YAML>'; change only in a safe maintenance window.",
    "scheduler_running": "Inspect PC-B scheduled task and supervisor; a Ready task alone is not proof of reattachment.",
    "receipt": "Inspect PC-B managed ComfyUI ownership receipt, without deleting it or starting another ComfyUI.",
    "process_identity": "Audit live PC-B CIM process identity; do not kill or adopt an unidentified listener.",
    "launcher_identity": "Verify original Windows venv launcher ancestry and original execution PID.",
    "tcp_ownership": "Inspect PC-B 'render-node socket-audit' and unexpected clients; do not interrupt existing GPU requests.",
    "snapshot_consistency": "Recheck live PC-B process and ownership receipt; refuse changing state.",
}
_PREFLIGHT_ACTIONS = {
    "renderer_configuration": "Configure PC-A render_nodes.primary, API transport and authenticated PC-B URL.",
    "render_token": "Set the matching PC-A/PC-B renderer token environment variable; never paste its value into diagnostics.",
    "comfyui_lan": "Check PC-B ComfyUI/gateway LAN address, firewall, port, and existing service.",
    "render_attestation": "Check PC-B attestation service, firewall, LAN address and matching Bearer token.",
    "render_attestation_freshness": "Check clocks on both Windows PCs and refresh PC-B attestation.",
    "render_os": "Use native Windows on PC-B.",
    "render_gpu": "Verify PC-B GPU driver with nvidia-smi without restarting ComfyUI.",
    "render_inventory": "Inspect PC-B asset/LoRA inventory with 'artifex render-node preflight'.",
    "llm_lan": "Check the existing PC-A LLM server endpoint and model; do not start another GPU workload.",
}


class ReadinessCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: Target
    name: str
    status: Status
    reason: str
    next_action: str | None = None


class QualificationReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    captured_utc: datetime
    primary_node_id: str | None
    environment_ready: bool
    actual_machine_qualification_complete: Literal[False] = False
    actual_gpu_soak_verified: Literal[False] = False
    mutated_services: Literal[False] = False
    checks: tuple[ReadinessCheck, ...]
    recorded_stages: dict[str, str] = Field(default_factory=dict)
    pending_stages: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()


def _check(
    target: Target, name: str, status: Status, reason: str,
    action: str | None = None,
) -> ReadinessCheck:
    return ReadinessCheck(
        target=target, name=name, status=status, reason=reason,
        next_action=None if status == "pass" else action,
    )


def _read_session(
    root: Path, session_id: str,
) -> QualificationSession:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", session_id):
        raise ValueError("Qualification session ID contains unsafe characters")
    evidence_dir = root.expanduser().absolute()
    path = evidence_dir / session_id / "qualification.json"
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Qualification evidence directory contains a symlink")
    if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Qualification session is absent or exceeds bounded size")
    session = QualificationSession.model_validate_json(path.read_bytes())
    if session.session_id != session_id:
        raise ValueError("Qualification session ID differs from its evidence path")
    return session


def diagnose_qualification_readiness(
    settings: ArtifexSettings,
    *,
    session_id: str | None = None,
    now: datetime | None = None,
    controller_probe: Callable[[ArtifexSettings], ControllerPreflight] = check_controller,
    owner_probe: Callable[
        [str, RenderNodeConfig], RemoteRendererOwnerAudit
    ] = fetch_renderer_owner_audit,
) -> QualificationReadiness:
    """Group non-destructive facts and safe next actions; never mark stage PASS."""
    current = now or datetime.now(UTC)
    checks: list[ReadinessCheck] = []
    primary = settings.render_nodes.primary_node()
    node_id = primary[0] if primary is not None else None
    checks.append(_check(
        "pc_a", "native_windows",
        "pass" if platform.system() == "Windows" else "fail",
        "PC-A must be native Windows for real target-machine qualification",
        "Run on the actual native Windows PC-A.",
    ))
    checks.append(_check(
        "pc_a", "uv_available",
        "pass" if shutil.which("uv") is not None else "fail",
        "uv package runner available" if shutil.which("uv") is not None
        else "uv executable not found on PC-A PATH",
        "Install/configure uv on PC-A without Docker.",
    ))
    checks.append(_check(
        "pc_a", "primary_renderer",
        "pass" if primary is not None else "fail",
        f"PC-A primary renderer: {node_id or '(unconfigured)'}",
        "Configure PC-A render_nodes.primary and its node entry.",
    ))
    if settings.llm.backend == "llama_cpp" and settings.llm.bootstrap.enabled:
        model = settings.llm.bootstrap.model_path().expanduser()
        checks.append(_check(
            "pc_a", "selected_gguf",
            "pass" if model.is_file() else "fail",
            "Selected local GGUF exists" if model.is_file()
            else "Selected local GGUF does not exist",
            "Use 'artifex onboard llama-assets' to inspect the chosen model, then bootstrap if necessary.",
        ))

    if primary is not None:
        node, config = primary
        token_ready = bool(
            config.attestation_token_env
            and os.environ.get(config.attestation_token_env)
        )
        checks.append(_check(
            "pc_a", "remote_owner_auth",
            "pass" if config.attestation_url and token_ready else "fail",
            "Authenticated PC-B owner audit configured" if config.attestation_url and token_ready
            else "Missing PC-B attestation URL or token environment variable",
            "Configure the PC-B attestation URL and shared Bearer token environment variable on PC-A.",
        ))
    try:
        preflight = controller_probe(settings)
        for item in preflight.checks:
            target: Target = (
                "pc_b" if item.name.startswith(
                    ("render_", "asset_", "comfyui_")
                ) and item.name != "render_token"
                else "pc_a"
            )
            action = _PREFLIGHT_ACTIONS.get(
                item.name,
                "Inspect the named PC-A/PC-B preflight check and repair it without interrupting GPU jobs.",
            )
            checks.append(_check(
                target, "preflight:" + item.name,
                "pass" if item.ready else "fail", item.detail, action,
            ))
        if not preflight.ready and not any(not c.ready for c in preflight.checks):
            checks.append(_check(
                "pc_a", "preflight:inconsistent", "fail",
                "Controller preflight is not ready despite no failing check",
                "Inspect PC-A preflight diagnostics.",
            ))
    except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError) as exc:
        checks.append(_check(
            "pc_a", "preflight:probe", "unknown",
            f"Controller preflight failed: {type(exc).__name__}",
            "Run 'artifex preflight --config <PC-A YAML> --json' for details.",
        ))

    if primary is not None:
        node, config = primary
        if config.attestation_url and config.attestation_token_env and os.environ.get(
            config.attestation_token_env
        ):
            try:
                remote = owner_probe(node, config)
                audit = remote.audit
                fresh = (
                    audit.captured_utc.tzinfo is not None
                    and current.tzinfo is not None
                    and -30 <= (current - audit.captured_utc).total_seconds() <= 120
                )
                checks.append(_check(
                    "pc_b", "owner:freshness",
                    "pass" if fresh else "fail",
                    "PC-B owner observation within 120 seconds" if fresh
                    else "PC-B owner observation is stale or clock-skewed",
                    "Check both PC clocks and retry read-only owner audit.",
                ))
                for key in sorted(_REQUIRED_CHECKS):
                    value = audit.checks.get(key)
                    state: Status = value.status if value is not None else "unknown"
                    checks.append(_check(
                        "pc_b", "owner:" + key, state,
                        value.reason if value is not None else f"Missing PC-B owner check: {key}",
                        _OWNER_ACTIONS[key],
                    ))
                checks.append(_check(
                    "pc_b", "owner:overall",
                    "pass" if (
                        audit.status == "observed_stable"
                        and audit.process_observation_verified
                        and remote.node_id == node
                    ) else "fail",
                    f"PC-B owner observation status: {audit.status}",
                    "Inspect actual PC-B 'render-node owner-audit'; never force reattachment or restart.",
                ))
            except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError) as exc:
                checks.append(_check(
                    "pc_b", "owner:remote_probe", "unknown",
                    f"Authenticated PC-B owner probe unavailable: {type(exc).__name__}",
                    "Check the PC-B renderer attestation service and network; retry without restarting ComfyUI.",
                ))
        else:
            checks.append(_check(
                "pc_b", "owner:remote_probe", "unknown",
                "PC-A cannot authenticate to PC-B owner audit",
                "Set PC-A PC-B attestation URL and token environment variable.",
            ))

    recorded: dict[str, str] = {}
    pending: tuple[str, ...] = ()
    if session_id:
        try:
            session = _read_session(settings.qualification.evidence_dir, session_id)
            recorded = {
                stage.value: session.stage(stage).status.value
                for stage in REQUIRED_STAGES
            }
            pending = tuple(
                stage.value for stage in REQUIRED_STAGES
                if session.stage(stage).status not in (
                    QualificationStatus.PASS, QualificationStatus.SKIPPED,
                )
            )
            checks.append(_check(
                "qualification", "session_records", "unknown",
                f"Found {len(recorded)} recorded stages; {len(pending)} not recorded as PASS/SKIPPED; these entries are not revalidated",
                "Run 'artifex qualify verify SESSION_ID --config <PC-A YAML>' for real evidence revalidation.",
            ))
        except (OSError, ValueError, KeyError) as exc:
            checks.append(_check(
                "qualification", "session_records", "fail",
                f"Qualification session cannot be inspected: {type(exc).__name__}",
                "Use an existing qualification session ID, or run 'artifex qualify start' once preflight passes.",
            ))
    else:
        checks.append(_check(
            "qualification", "session_records", "unknown",
            "No session ID selected; 14-stage GPU qualification has not been verified",
            "After fixing environment gaps, run 'artifex qualify start --config <PC-A YAML>'.",
        ))
    environment_ready = all(
        check.status == "pass"
        for check in checks if check.target != "qualification"
    )
    actions = tuple(dict.fromkeys(
        check.next_action for check in checks
        if check.status != "pass" and check.next_action is not None
    ))
    return QualificationReadiness(
        captured_utc=current, primary_node_id=node_id,
        environment_ready=environment_ready, checks=tuple(checks),
        recorded_stages=recorded, pending_stages=pending, next_actions=actions,
    )
