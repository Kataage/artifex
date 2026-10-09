"""Deterministic operator-safe install actions from a SINGLE read-only pair audit.

Advice is structured argv, never a shell script. All steps are advisory:
no child execution, Windows Task Scheduler change, config write, or GPU action.
Only the separately existing PC-B observer enrollment has an optional, explicit
operator-approved apply argv. It is NEVER executed here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.qualification.pair_installation import TwoPCInstallationAudit
from artifex.render_node.installation_audit import TaskInstallationCheck

Role = Literal["pc_a", "pc_b"]
Kind = Literal["read_only", "operator_review", "optional_explicit_apply", "wait_for_evidence"]


class PairInstallAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    role: Role
    kind: Kind
    explanation: str
    matching_blockers: tuple[str, ...]
    config_keys: tuple[str, ...] = ()
    read_only_argv: tuple[str, ...] | None = None
    # Only safe PC-B observer enrollment. Never include an apply argv for
    # renderer task drift, unowned tasks, or controller/ComfyUI processes.
    operator_approved_apply_argv: tuple[str, ...] | None = None
    commands_executed: Literal[False] = False
    services_modified: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False


class PairInstallRemediationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source: Literal["read_only_pair_install_audit"] = "read_only_pair_install_audit"
    status: str
    steps: tuple[PairInstallAdvice, ...]
    read_only_only: bool
    optional_operator_approval_count: int
    commands_executed: Literal[False] = False
    config_written: Literal[False] = False
    renderer_task_modified: Literal[False] = False
    comfyui_process_modified: Literal[False] = False
    actual_survival_observed: Literal[False] = False
    production_qualified: Literal[False] = False


def _argv(
    *args: str,
    config: Path | None = None,
) -> tuple[str, ...]:
    if config is None:
        return ("uv", "run", "artifex", *args, "--json")
    return ("uv", "run", "artifex", *args, "--config", str(config), "--json")


def compile_pair_install_remediation(
    audit: TwoPCInstallationAudit,
    *,
    controller_config: Path,
    renderer_config: Path,
) -> PairInstallRemediationPlan:
    """Turn *observed* gaps into safe PC-specific actions with no I/O.

    The renderer_config is solely an operator-supplied PC-B-local path. No
    remote PC-B paths, tokens, or executable command lines are copied back.
    """
    steps: list[PairInstallAdvice] = []

    def add(
        step_id: str, role: Role, kind: Kind, explanation: str,
        blockers: tuple[str, ...],
        *, argv: tuple[str, ...] | None = None,
        apply: tuple[str, ...] | None = None,
        keys: tuple[str, ...] = (),
    ) -> None:
        if any(x.step_id == step_id for x in steps):
            raise ValueError("Duplicate pair installation advice")
        steps.append(PairInstallAdvice(
            step_id=step_id, role=role, kind=kind,
            explanation=explanation, matching_blockers=blockers,
            config_keys=keys, read_only_argv=argv,
            operator_approved_apply_argv=apply,
        ))

    local = audit.pc_a_controller_task
    if local.reason_code == "native_windows_required":
        add(
            "pc-a-windows", "pc_a", "operator_review",
            "Run the installation audit on the real native Windows PC-A.",
            ("pc_a_native_windows_required",),
        )
    if "pc_a_controller_config_missing_or_unsafe" in audit.blockers:
        add(
            "pc-a-config", "pc_a", "read_only",
            "Preview required controller settings before writing any YAML.",
            ("pc_a_controller_config_missing_or_unsafe",),
            argv=_argv("onboard", "first-run", "--role", "controller"),
            keys=("render_nodes.primary", "render_nodes.nodes", "llm"),
        )
    if local.status in {"missing", "registered_not_running", "unsafe", "unavailable"}:
        add(
            "pc-a-task", "pc_a",
            "operator_review" if local.status == "unsafe" else "read_only",
            "Inspect the current controller task. Never auto-register, replace "
            "or start an existing controller process.",
            ("pc_a_controller_task_" + local.status,),
            argv=_argv("startup", "status", "--role", "controller"),
        )
        if local.status in {"registered_not_running", "unsafe"}:
            add(
                "pc-a-task-command", "pc_a", "read_only",
                "Check ownership and the exact configured controller startup "
                "action; a drifted or foreign task must be reviewed manually.",
                ("pc_a_controller_task_" + local.status,),
                argv=_argv(
                    "startup", "audit", "--role", "controller",
                    config=controller_config,
                ),
            )
    if audit.pc_b_node_id is None or "pc_b_primary_node_unconfigured" in audit.blockers:
        add(
            "pc-a-render-node", "pc_a", "read_only",
            "Select the real PC-B node and authenticated URL; no IP, port or token "
            "value is guessed.",
            ("pc_b_primary_node_unconfigured",),
            argv=_argv("onboard", "first-run", "--role", "controller"),
            keys=("render_nodes.primary", "render_nodes.nodes", "render_nodes.nodes.*.attestation_url"),
        )
    elif "pc_b_attestation_or_bearer_unconfigured" in audit.blockers:
        add(
            "pc-a-auth", "pc_a", "operator_review",
            "Configure the actual PC-B attestation URL and shared environment "
            "variable name; store its secret value outside YAML/CLI arguments.",
            ("pc_b_attestation_or_bearer_unconfigured",),
            keys=("render_nodes.nodes.*.attestation_url", "render_nodes.nodes.*.attestation_token_env"),
        )
    if audit.status == "pc_b_unreachable":
        add(
            "pc-b-connection", "pc_a", "read_only",
            "Check trusted PC-B URL, LAN reachability, attestation service and "
            "Bearer variable. Do not restart ComfyUI to make the check pass.",
            ("pc_b_authenticated_installation_unavailable",),
            argv=_argv(
                "onboard", "first-run", "--role", "controller",
                config=controller_config,
            ),
            keys=("render_nodes.nodes.*.attestation_url",),
        )
    remote = audit.pc_b.audit if audit.pc_b else None
    if remote is not None:
        if not remote.native_windows:
            add(
                "pc-b-windows", "pc_b", "operator_review",
                "Real PC-B must use a native Windows setup for this watcher.",
                ("pc_b_not_native_windows",),
            )
        if not remote.managed_renderer_configured:
            add(
                "pc-b-managed-comfyui", "pc_b", "operator_review",
                "Review real ComfyUI ownership and managed launch settings; "
                "never adopt, kill or replace a live unrelated process.",
                ("pc_b_managed_renderer_not_configured",),
                keys=(
                    "render_agent.comfyui_process.enabled",
                    "render_agent.comfyui_process.executable",
                    "render_agent.comfyui_process.working_directory",
                ),
            )

        def task_step(
            task: TaskInstallationCheck, *,
            role: Literal["renderer", "survival-observer"],
        ) -> None:
            if task.status == "running" and task.configuration_verified:
                return
            key = "pc-b-" + role
            if role == "renderer":
                add(
                    "pc-b-renderer-task", "pc_b", "operator_review",
                    "Inspect only the owned renderer task and policy; no "
                    "automatic renderer install, start, stop or replacement.",
                    ("pc_b_renderer_" + task.status,),
                    argv=_argv(
                        "startup", "audit", "--role", "renderer",
                        config=renderer_config,
                    ),
                )
            else:
                # Explicit approval is available ONLY when the task is safe
                # to enroll without changing ComfyUI, and the renderer is
                # already fully verified on the real Windows PC-B.
                can_enroll = (
                    remote.native_windows
                    and remote.managed_renderer_configured
                    and audit.pc_b is not None
                    and audit.pc_b.node_id == audit.pc_b_node_id
                    and remote.node_id == audit.pc_b_node_id
                    and "pc_b_node_identity_or_snapshot_stale" not in audit.blockers
                    and remote.renderer_task.role == "renderer"
                    and remote.renderer_task.status == "running"
                    and remote.renderer_task.configuration_verified
                    and task.role == "survival-observer"
                    and (
                        task.status == "missing"
                        or (
                            task.status == "registered_not_running"
                            and task.managed and task.configuration_verified
                        )
                    )
                )
                readonly = _argv(
                    "startup", "observer-enable", config=renderer_config,
                )
                approval = (
                    _argv(
                        "startup", "observer-enable", "--apply",
                        config=renderer_config,
                    )
                    if can_enroll else None
                )
                add(
                    key, "pc_b",
                    "optional_explicit_apply" if can_enroll else "operator_review",
                    "On PC-B preview the independent survival observer. "
                    "Only if the renderer is verified and the observer task "
                    "is absent or safely stopped may an operator explicitly "
                    "approve observer-only --apply; never touch ComfyUI.",
                    ("pc_b_survival_observer_" + task.status,),
                    argv=readonly, apply=approval,
                )
        task_step(remote.renderer_task, role="renderer")
        task_step(remote.survival_observer_task, role="survival-observer")
        if remote.observer_heartbeat != "fresh":
            add(
                "pc-b-observer-liveness", "pc_b", "read_only",
                "Task Scheduler Running is not enough: inspect recent independent "
                "observer samples without changing renderer state.",
                ("pc_b_observer_heartbeat_" + remote.observer_heartbeat,),
                argv=_argv("startup", "status", "--role", "survival-observer"),
            )
        if remote.evidence_spool in {"unsafe", "unavailable"}:
            add(
                "pc-b-spool", "pc_b", "operator_review",
                "Inspect the configured evidence directory and permissions "
                "without deleting, overwriting or following symlinked files.",
                ("pc_b_survival_spool_" + remote.evidence_spool,),
                keys=("render_agent.survival_evidence_dir",),
            )
        elif remote.evidence_spool in {"missing", "empty"}:
            add(
                "pc-b-no-event-yet", "pc_b", "wait_for_evidence",
                "The absence of an event trace is normal before a naturally "
                "observed exit. Never terminate a working GPU task to create one.",
                (),
            )
        if (
            not remote.safe_for_passive_observation
            or remote.renderer_task.role != "renderer"
            or remote.survival_observer_task.role != "survival-observer"
        ):
            add(
                "pc-b-preconditions", "pc_b", "operator_review",
                "Do not infer live watcher health from scheduler registration. "
                "Check the contradictory/insufficient observer prerequisites.",
                ("pc_b_passive_observer_preconditions_unverified",),
            )
    if not steps:
        add(
            "monitor-only", "pc_a", "wait_for_evidence",
            "Installed task prerequisites look consistent, but no actual "
            "ComfyUI survival, GPU run or eight-hour soak is qualified.",
            (),
        )
    return PairInstallRemediationPlan(
        status=audit.status, steps=tuple(steps),
        read_only_only=all(step.operator_approved_apply_argv is None for step in steps),
        optional_operator_approval_count=sum(
            step.operator_approved_apply_argv is not None for step in steps
        ),
    )
