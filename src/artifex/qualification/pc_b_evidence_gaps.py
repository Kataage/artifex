"""Read-only triage of PC-B renderer ownership and supervisor-survival evidence.

Only a matching *live identity* can be reported as correlated. Neither a saved
launcher fixture nor a replayed historical trace proves real process-loss
survival, GPU operation, or eight-hour production qualification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.qualification.owner_handoff import PCBOwnerCorrelation
from artifex.qualification.survival_review import PCBSurvivalEvidenceReview

ProofKind = Literal["renderer_owner_identity", "supervisor_survival_trace"]
Assessment = Literal[
    "not_requested", "identity_correlated", "historical_only", "needs_review",
]


class PCBEvidenceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proof: ProofKind
    assessment: Assessment
    recorded_status: str | None
    source_mode: str | None
    responsible_role: Literal["pc_a", "pc_b"]
    identity_correlated: bool
    trace_replayed: bool
    next_action: str
    read_only_argv: tuple[str, ...] | None = None
    commands_executed: Literal[False] = False
    real_gpu_qualified: Literal[False] = False
    supervisor_loss_qualified: Literal[False] = False
    issue_93_closed: Literal[False] = False
    stage_pass_registered: Literal[False] = False


def compile_pc_b_evidence_checks(
    *,
    pc_b_configured: bool,
    owner: PCBOwnerCorrelation | None,
    survival: PCBSurvivalEvidenceReview | None,
    controller_config: Path,
) -> tuple[PCBEvidenceCheck, ...]:
    """Describe gaps without fetching data or changing any PC-B process."""
    if not pc_b_configured:
        # The controller has no configured remote renderer. Do not invent a
        # PC-B machine or ask to retrieve authenticated proof from nowhere.
        return ()

    def overview_command(option: str) -> tuple[str, ...]:
        return (
            "uv", "run", "artifex", "qualify", "overview",
            option, "--config", str(controller_config), "--json",
        )

    checks: list[PCBEvidenceCheck] = []
    if owner is None:
        checks.append(PCBEvidenceCheck(
            proof="renderer_owner_identity",
            assessment="not_requested",
            recorded_status=None,
            source_mode=None,
            responsible_role="pc_a",
            identity_correlated=False,
            trace_replayed=False,
            next_action=(
                "No PC-B owner report was requested by this overview. On PC-A, "
                "retrieve the protected PC-B cached ownership report and compare "
                "the listener PID/start time with fresh authenticated PC-B "
                "attestation. Read-only GET; no ComfyUI restart."
            ),
            read_only_argv=overview_command("--pc-b-owner-live"),
        ))
    elif owner.status == "correlated_read_only":
        checks.append(PCBEvidenceCheck(
            proof="renderer_owner_identity",
            assessment="identity_correlated",
            recorded_status=owner.status,
            source_mode=owner.source_mode,
            responsible_role="pc_a",
            identity_correlated=True,
            trace_replayed=False,
            next_action=(
                "Saved PC-B owner evidence matches the freshly observed "
                "renderer identity. The original report or fixture is not "
                "thereby authenticated, and process-loss survival, real GPU "
                "work, and eight-hour soak remain independently unqualified."
            ),
        ))
    else:
        role: Literal["pc_a", "pc_b"] = (
            "pc_b" if owner.status in {"mismatch", "stale"} else "pc_a"
        )
        checks.append(PCBEvidenceCheck(
            proof="renderer_owner_identity",
            assessment="needs_review",
            recorded_status=owner.status,
            source_mode=owner.source_mode,
            responsible_role=role,
            identity_correlated=False,
            trace_replayed=False,
            next_action=(
                "PC-B owner identity is not correlated (" + owner.status + "). "
                "Inspect the existing evidence and native PC-B PID/start time "
                "or PC-A authenticated reachability. A stale/mismatched report "
                "cannot be accepted as proof. Do not kill or restart ComfyUI."
            ),
        ))

    if survival is None:
        checks.append(PCBEvidenceCheck(
            proof="supervisor_survival_trace",
            assessment="not_requested",
            recorded_status=None,
            source_mode=None,
            responsible_role="pc_a",
            identity_correlated=False,
            trace_replayed=False,
            next_action=(
                "No saved PC-B supervisor-survival trace was requested. On PC-A, "
                "read the current PC-B immutable trace and compare its historical "
                "listener identity with the authenticated current owner. "
                "A replay alone never proves a physical supervisor-loss event."
            ),
            read_only_argv=overview_command("--pc-b-survival-live"),
        ))
    elif survival.status == "replayed_and_live_identity_matched":
        checks.append(PCBEvidenceCheck(
            proof="supervisor_survival_trace",
            assessment="identity_correlated",
            recorded_status=survival.status,
            source_mode=survival.source_mode,
            responsible_role="pc_a",
            identity_correlated=True,
            trace_replayed=survival.trace_replayed,
            next_action=(
                "A saved survival trace was replayed against a matching current "
                "renderer identity, but its historical supervisor-loss event "
                "is not authenticated and Issue #93 remains unqualified. "
                "Independent real PC-B process-loss evidence is still required."
            ),
        ))
    elif survival.status == "replayed_historical_identity":
        checks.append(PCBEvidenceCheck(
            proof="supervisor_survival_trace",
            assessment="historical_only",
            recorded_status=survival.status,
            source_mode=survival.source_mode,
            responsible_role="pc_b",
            identity_correlated=False,
            trace_replayed=survival.trace_replayed,
            next_action=(
                "The saved trace was replayed but its historical listener "
                "identity differs from the current process. Inspect PC-B "
                "process identity and timestamps without forcing a restart. "
                "Historical replay is not live survivor qualification."
            ),
        ))
    else:
        role = (
            "pc_b" if survival.status in {"mismatch", "blocked"} else "pc_a"
        )
        checks.append(PCBEvidenceCheck(
            proof="supervisor_survival_trace",
            assessment="needs_review",
            recorded_status=survival.status,
            source_mode=survival.source_mode,
            responsible_role=role,
            identity_correlated=False,
            trace_replayed=survival.trace_replayed,
            next_action=(
                "PC-B survival trace has not been correlated ("
                + survival.status
                + "). Review original source, sample timing and current "
                "authenticated PC-B identity without starting or stopping "
                "the renderer."
            ),
        ))
    return tuple(checks)
