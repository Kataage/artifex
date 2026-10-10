"""Operator-ready, non-executing per-PC diagnostics for Issue #93.

The advice is derived from the current PC-A observational checklist. Command
vectors are data, never shell strings. The PC-B config path is supplied by the
operator, not read from an untrusted remote response or used on PC-A.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.qualification.issue93_checklist import Issue93EvidenceChecklist

Role = Literal["pc_a", "pc_b"]
Effect = Literal["read_only", "disposable_no_gpu_probe", "local_evidence_write"]


class Issue93FieldStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    role: Role
    effect: Effect
    description: str
    argv: tuple[str, ...]
    executed: Literal[False] = False
    service_mutated: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False


class Issue93FieldPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source_status: str
    steps: tuple[Issue93FieldStep, ...]
    commands_executed: Literal[False] = False
    pc_b_config_read_remotely: Literal[False] = False
    service_mutated: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    historical_supervisor_loss_authenticated: Literal[False] = False
    physical_pc_b_qualified: Literal[False] = False
    production_qualified: Literal[False] = False


def _cmd(path: Path, *words: str) -> tuple[str, ...]:
    return ("uv", "run", "artifex", *words, "--config", str(path), "--json")


def compile_issue93_field_plan(
    checklist: Issue93EvidenceChecklist, *,
    controller_config: Path,
    renderer_config: Path,
) -> Issue93FieldPlan:
    """Propose only safe already-implemented CLI commands; execute none.

    The actual PC-B config path is advisory, and can differ arbitrarily from
    the PC-A YAML path. Never interpolate a user path into a shell string.
    """
    steps: list[Issue93FieldStep] = []

    def append(
        id: str, role: Role, effect: Effect,
        description: str, argv: tuple[str, ...],
    ) -> None:
        if any(x.id == id for x in steps):
            raise ValueError("Duplicate Issue #93 field diagnostic step")
        if "--apply" in argv or "--confirm" in argv:
            raise ValueError("Issue #93 field plan cannot contain mutation flags")
        steps.append(Issue93FieldStep(
            id=id, role=role, effect=effect, description=description,
            argv=argv,
        ))

    if checklist.status == "unconfigured":
        append(
            "pc-a-first-run", "pc_a", "read_only",
            "Inspect the actual PC-A YAML and missing authenticated PC-B connection settings. "
            "Never guess a LAN address or expose a token value.",
            _cmd(controller_config, "onboard", "first-run", "--role", "controller"),
        )
    else:
        unresolved = {
            item.name: item.state for item in checklist.checks
            if item.state != "observed"
        }
        if "isolated_launcher_and_live_owner" in unresolved:
            append(
                "pc-b-launcher-owner", "pc_b", "disposable_no_gpu_probe",
                "On PC-B, run the local disposable loopback Python launcher fixture "
                "and independently read current ComfyUI ownership. The fixture "
                "does not start, stop, or adopt ComfyUI.",
                _cmd(renderer_config, "render-node", "owner-readiness"),
            )
        if "two_time_separated_owner_samples" in unresolved:
            append(
                "pc-a-fresh-owner-pair", "pc_a", "local_evidence_write",
                "On PC-A, obtain two fresh authenticated read-only owner snapshots "
                "and save an exclusive-create local evidence file. A failed new "
                "snapshot cannot fall back to an earlier PASS.",
                _cmd(
                    controller_config, "qualify", "issue93-status",
                    "--refresh-owner-pair", "--pair-gap-seconds", "5",
                ),
            )
        if "natural_exit_survival_trace" in unresolved:
            append(
                "pc-b-observer-dry-run", "pc_b", "read_only",
                "On PC-B, inspect whether the independent survival observer could "
                "be enrolled; this command omits --apply and changes no task. "
                "Never force a supervisor exit.",
                _cmd(renderer_config, "startup", "observer-enable"),
            )
            append(
                "pc-a-survival-review", "pc_a", "read_only",
                "On PC-A, fetch and replay the latest PC-B saved trace through "
                "the configured authenticated GET endpoint, without asserting "
                "that its historical event is independently authentic.",
                _cmd(
                    controller_config, "qualify", "overview",
                    "--pc-b-survival-live",
                ),
            )
        if unresolved.get("cross_evidence_listener_identity") == "conflict":
            append(
                "pc-b-owner-conflict", "pc_b", "read_only",
                "Inspect exact live ComfyUI listener PID, creation time and "
                "launcher receipt on PC-B. Do not adopt or restart the process.",
                _cmd(renderer_config, "render-node", "owner-audit"),
            )
        if not steps:
            # Observational correlation is not a real-host process ancestry,
            # natural-loss authentication, reattachment or GPU qualification.
            append(
                "pc-a-production-overview", "pc_a", "read_only",
                "Review real-machine stage gaps and current PC-B authenticated "
                "owner/survival evidence. Correlation alone never authorizes a "
                "GPU production run or closes Issues #93/#40.",
                _cmd(
                    controller_config, "qualify", "overview",
                    "--pc-b-owner-live", "--pc-b-survival-live",
                ),
            )
    return Issue93FieldPlan(source_status=checklist.status, steps=tuple(steps))
