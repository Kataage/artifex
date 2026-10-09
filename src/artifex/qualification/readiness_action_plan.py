"""Generate a deterministic, non-executable remediation plan from readiness evidence.

Every argv vector is advisory. Nothing in this module spawns a process or
changes config, ComfyUI, a Task Scheduler entry, or qualification evidence.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from artifex.qualification.readiness_diagnostics import (
    QualificationReadiness,
    ReadinessCheck,
)

Role = Literal["pc_a", "pc_b"]
Safety = Literal["read_only", "review_required", "real_machine_evidence"]


class RemediationStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    role: Role
    safety: Safety
    blocked_checks: tuple[str, ...]
    description: str
    argv: tuple[str, ...] | None = None
    automatically_executed: Literal[False] = False


class QualificationActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source: Literal["live", "saved"]
    source_captured_utc: datetime
    compiled_utc: datetime
    snapshot_fresh: bool
    environment_ready_observed: bool
    production_qualified: Literal[False] = False
    gpu_soak_qualified: Literal[False] = False
    mutated_services: Literal[False] = False
    steps: tuple[RemediationStep, ...]
    outstanding_stages: tuple[str, ...]
    blocking_checks: tuple[str, ...]
    read_only_steps: int = Field(ge=0)
    review_required_steps: int = Field(ge=0)


def _argv(role: Role, config: Path, *parts: str) -> tuple[str, ...]:
    # Structured argv: never concatenate check text or config paths into a
    # shell string. Operators can use subprocess/PowerShell argument arrays.
    return ("uv", "run", "artifex", *parts, "--config", str(config), "--json")


def _unresolved(check: ReadinessCheck) -> bool:
    return check.status != "pass" and check.target != "qualification"


def compile_qualification_action_plan(
    report: QualificationReadiness,
    *,
    controller_config: Path = Path("config/local.yaml"),
    renderer_config: Path = Path("config/render-node.yaml"),
    source: Literal["live", "saved"] = "live",
    now: datetime | None = None,
) -> QualificationActionPlan:
    """Group environmental gaps into safe PC-specific inspection and review work.

    Never infer that a previously saved snapshot is current, and never convert
    recorded qualification stage names to real production evidence.
    """
    current = now or datetime.now(UTC)
    captured = report.captured_utc
    fresh = bool(
        current.tzinfo is not None
        and captured.tzinfo is not None
        and timedelta(seconds=-30) <= current - captured <= timedelta(minutes=5)
    )
    unresolved = [c for c in report.checks if _unresolved(c)]
    gaps = tuple(f"{c.target}:{c.name}" for c in unresolved)
    steps: list[RemediationStep] = []

    def add(
        ident: str,
        role: Role,
        safety: Safety,
        description: str,
        affected: list[ReadinessCheck],
        argv: tuple[str, ...] | None = None,
    ) -> None:
        if not affected:
            return
        steps.append(RemediationStep(
            id=ident, role=role, safety=safety,
            description=description,
            blocked_checks=tuple(f"{c.target}:{c.name}" for c in affected),
            argv=argv,
        ))

    def group(prefix: str, names: tuple[str, ...]) -> list[ReadinessCheck]:
        return [
            c for c in unresolved
            if c.target == prefix and any(
                c.name == name or c.name.startswith(name + ":") for name in names
            )
        ]

    # PC-A prerequisites can be measured without creating or altering
    # services. File presence, credentials and network require separate help.
    local = group("pc_a", ("native_windows", "uv_available"))
    add(
        "pc-a-platform", "pc_a", "review_required",
        "PC-A must run Windows natively with uv on PATH; install or repair outside active GPU jobs.",
        local,
    )
    controller_settings = group(
        "pc_a", ("primary_renderer", "remote_owner_auth", "preflight:renderer_configuration",
                 "preflight:render_token", "preflight:asset_"),
    )
    add(
        "pc-a-config", "pc_a", "review_required",
        "Review primary renderer, chosen production models and shared token environment variable without showing its value.",
        controller_settings,
    )
    model = group("pc_a", ("selected_gguf",))
    add(
        "pc-a-gguf", "pc_a", "review_required",
        "Select or bootstrap the pinned GGUF model outside active GPU work; never overwrite existing files.",
        model,
    )
    controller_checks = [
        c for c in unresolved if c.target == "pc_a"
        and c not in (*local, *controller_settings, *model)
    ]
    add(
        "pc-a-preflight", "pc_a", "read_only",
        "Recheck PC-A LLM, ComfyUI transport, renderer attestation and model agreement.",
        controller_checks,
        _argv("pc_a", controller_config, "preflight"),
    )
    # PC-B may have live GPU work. All suggested probes are observational:
    # no scripts start or stop a listener, download weights or install tasks.
    pc_b_owner = [c for c in unresolved if c.target == "pc_b" and c.name.startswith("owner:")]
    add(
        "pc-b-owner", "pc_b", "read_only",
        "Inspect current PC-B managed ComfyUI PID, startup task, ownership receipt and TCP listeners without intervention.",
        pc_b_owner,
        _argv("pc_b", renderer_config, "render-node", "owner-audit"),
    )
    pc_b_network = group(
        "pc_b", ("preflight:comfyui_lan", "preflight:render_attestation",
                 "preflight:render_attestation_freshness"),
    )
    add(
        "pc-b-connectivity", "pc_b", "read_only",
        "Check PC-B service endpoints, LAN bind and clock from PC-A; confirm firewall/token settings before any restart.",
        pc_b_network,
        _argv("pc_a", controller_config, "preflight"),
    )
    pc_b_assets = [
        c for c in unresolved if c.target == "pc_b"
        and c not in (*pc_b_owner, *pc_b_network)
        and (
            c.name in {"preflight:render_gpu", "preflight:render_os",
                       "preflight:render_inventory"}
            or c.name.startswith("preflight:asset_")
        )
    ]
    add(
        "pc-b-inventory", "pc_b", "read_only",
        "Inspect PC-B GPU visibility and selected checkpoint, VAE, refiner and LoRA inventory without loading models.",
        pc_b_assets,
        _argv("pc_b", renderer_config, "render-node", "preflight"),
    )

    # Protect against new check names silently disappearing from the plan:
    # unresolved/unknown checks are retained even if a mapping is missing.
    allocated = {name for item in steps for name in item.blocked_checks}
    remaining = [c for c in unresolved if f"{c.target}:{c.name}" not in allocated]
    for c in remaining:
        add(
            "inspect-" + re.sub(r"[^a-z0-9_-]+", "-", c.name.lower())[:64],
            "pc_a" if c.target == "pc_a" else "pc_b",
            "review_required",
            f"Investigate unclassified check {c.name} without modifying running services.",
            [c],
        )

    # Always include a real-machine evidence step. Even a flawless preflight
    # is not a qualification success and cannot authorize production.
    steps.append(RemediationStep(
        id="14-stage-real-machine-evidence",
        role="pc_a",
        safety="real_machine_evidence",
        blocked_checks=("qualification:session_records",),
        description=(
            "Complete and independently verify all 14 target-machine stages, including "
            "actual active GPU recovery, Discord (if enabled) and eight-hour soak. "
            "A recorded PASS or a simulation does not qualify production."
        ),
        argv=None,
    ))
    if not fresh:
        steps.insert(0, RemediationStep(
            id="refresh-readiness",
            role="pc_a",
            safety="read_only",
            blocked_checks=("readiness:snapshot_freshness",),
            description="The readiness snapshot is older than five minutes or clock-skewed; collect a fresh read-only observation.",
            argv=_argv("pc_a", controller_config, "qualify", "readiness"),
        ))
    steps.sort(key=lambda step: (
        0 if step.safety == "read_only" else
        1 if step.safety == "review_required" else 2,
        0 if step.role == "pc_a" else 1,
        step.id,
    ))
    return QualificationActionPlan(
        source=source, source_captured_utc=captured, compiled_utc=current,
        snapshot_fresh=fresh,
        environment_ready_observed=report.environment_ready and fresh and not gaps,
        steps=tuple(steps), outstanding_stages=report.pending_stages,
        blocking_checks=gaps,
        read_only_steps=sum(s.safety == "read_only" for s in steps),
        review_required_steps=sum(s.safety == "review_required" for s in steps),
    )
