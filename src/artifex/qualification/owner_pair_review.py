"""Read-only, bounded review of saved PC-A dual-snapshot owner evidence.

Saved JSON is not signed. Fresh authenticated PC-B corroboration is needed
to assert any current observational correlation; this never proves uptime,
Task Scheduler survival, active GPU generation, or production qualification.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings, RenderNodeConfig
from artifex.qualification.native_creation_time import (
    native_creation_instant,
    same_native_creation_instant,
)
from artifex.qualification.owner_pair import OwnerPairObservation, _valid_owner
from artifex.qualification.renderer_owner_evidence import _REQUIRED_CHECKS
from artifex.render_node.client import fetch_renderer_owner_audit
from artifex.render_node.models import RemoteRendererOwnerAudit

_MAX_BYTES = 32 * 1024
_MAX_FILES = 1000
_MAX_SAVED_AGE = timedelta(minutes=10)
_MAX_CLOCK_SKEW = timedelta(seconds=30)
ReviewStatus = Literal[
    "correlated_read_only", "saved_only", "blocked", "stale",
    "unavailable", "unconfigured",
]


class OwnerPairReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    status: ReviewStatus
    reason: str
    checked_utc: datetime
    node_id: str | None
    saved_report_sha256: str | None = None
    saved_sample_count: int = 0
    saved_listener_pid: int | None = None
    current_listener_pid: int | None = None
    current_listener_started_utc: str | None = None
    saved_required_checks_consistent: bool = False
    saved_identity_consistent: bool = False
    live_pc_b_owner_correlated: bool = False
    file_source_authenticated: Literal[False] = False
    # Authenticated GET proves the currently retrieved endpoint response,
    # not provenance of an operator-copied or independently edited JSON.
    remote_bearer_checked: bool = False
    real_pc_b_launcher_fixture_proven: Literal[False] = False
    supervisor_loss_survival_proven: Literal[False] = False
    real_gpu_qualified: Literal[False] = False
    fourteen_stage_qualification_proven: Literal[False] = False
    eight_hour_soak_proven: Literal[False] = False
    issue_93_closure_authorized: Literal[False] = False
    issue_40_closure_authorized: Literal[False] = False
    production_qualified: Literal[False] = False
    renderer_restart_authorized: Literal[False] = False
    task_actions_executed: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False


def _safe_path(path: Path) -> bool:
    selected = path.expanduser().absolute()
    return not any(item.is_symlink() for item in (selected, *selected.parents))


def _latest_evidence(directory: Path) -> Path | None:
    """Deterministically choose a saved CLI report without arbitrary traversal."""
    directory = directory.expanduser().absolute()
    if not _safe_path(directory):
        raise ValueError("symlinked_evidence_directory")
    if not directory.is_dir():
        return None
    choices: list[Path] = []
    for count, path in enumerate(directory.iterdir(), start=1):
        if count > _MAX_FILES:
            raise ValueError("evidence_directory_entry_limit")
        if path.suffix == ".json":
            if not _safe_path(path) or not path.is_file():
                raise ValueError("unsafe_evidence_candidate")
            choices.append(path)
    # Even the current timestamp format has a random suffix; filesystem
    # modification times are the actual available ordering evidence.
    # A tie is ambiguous (e.g., coarse-resolution Windows/SMB timestamps).
    # Do NOT use the random name to pick an older apparent PASS.
    if not choices:
        return None
    stamped = [(candidate.lstat().st_mtime_ns, candidate) for candidate in choices]
    newest_time = max(mtime for mtime, _ in stamped)
    newest = [candidate for mtime, candidate in stamped if mtime == newest_time]
    if len(newest) != 1:
        raise ValueError("owner_pair_latest_evidence_ambiguous")
    return newest[0]


def _valid_saved_pair(report: OwnerPairObservation, checked: datetime) -> tuple[bool, bool]:
    a, b = report.first_captured_utc, report.second_captured_utc
    expected = set(_REQUIRED_CHECKS)
    required = (
        set(report.first_required_checks) == expected
        and set(report.second_required_checks) == expected
        and all(value == "pass" for value in report.first_required_checks.values())
        and all(value == "pass" for value in report.second_required_checks.values())
    )
    if not (
        report.status == "consistent_samples"
        and report.reason == "same_renderer_identity_at_two_observations"
        and report.sample_count == 2
        and report.node_id
        and report.checked_utc.tzinfo is not None
        and a is not None and b is not None
        and a.tzinfo is not None and b.tzinfo is not None
        and checked.tzinfo is not None
        and -_MAX_CLOCK_SKEW <= checked - report.checked_utc <= _MAX_SAVED_AGE
        and -_MAX_CLOCK_SKEW <= report.checked_utc - b <= timedelta(minutes=2)
        and a <= b
        and (b - a).total_seconds() >= report.requested_gap_seconds * 0.75
        and report.elapsed_between_samples_seconds >= report.requested_gap_seconds
        and report.first_listener_pid is not None
        and report.first_listener_pid > 0
        and report.first_listener_pid == report.second_listener_pid
        and report.first_process_started_utc
        and same_native_creation_instant(
            report.first_process_started_utc, report.second_process_started_utc,
        )
        and report.first_launcher_pid == report.second_launcher_pid
        and report.first_receipt_schema in {1, 2}
        and report.first_receipt_schema == report.second_receipt_schema
        and (report.first_receipt_schema == 2) == (report.first_launcher_pid is not None)
    ):
        return required, False
    started = native_creation_instant(report.first_process_started_utc)
    return required, (
        started is not None and started <= a and started <= b
    )


def review_owner_pair_evidence(
    settings: ArtifexSettings, *,
    report_path: Path | None = None,
    offline: bool = False,
    owner_probe: Callable[
        [str, RenderNodeConfig], RemoteRendererOwnerAudit
    ] = fetch_renderer_owner_audit,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> OwnerPairReview:
    """Fail closed on malformed evidence and require auth for live correlation."""
    current = now()
    primary = settings.render_nodes.primary_node()
    expected_node = primary[0] if primary else None
    saved: OwnerPairObservation | None = None
    digest: str | None = None
    checked = False
    same_identity = False
    live_pid: int | None = None
    live_started: str | None = None
    bearer = False

    def result(status: ReviewStatus, reason: str) -> OwnerPairReview:
        return OwnerPairReview(
            status=status, reason=reason, checked_utc=current, node_id=expected_node,
            saved_report_sha256=digest,
            saved_sample_count=saved.sample_count if saved else 0,
            saved_listener_pid=saved.second_listener_pid if saved else None,
            current_listener_pid=live_pid,
            current_listener_started_utc=live_started,
            saved_required_checks_consistent=checked,
            saved_identity_consistent=same_identity,
            live_pc_b_owner_correlated=status == "correlated_read_only",
            remote_bearer_checked=bearer,
        )

    try:
        selected = report_path or _latest_evidence(
            settings.qualification.evidence_dir / "owner-pair"
        )
        if selected is None:
            return result("unconfigured", "no_saved_owner_pair_evidence")
        selected = selected.expanduser().absolute()
        if not _safe_path(selected) or not selected.is_file():
            return result("blocked", "unsafe_or_missing_evidence_path")
        before = selected.stat()
        if not 0 < before.st_size <= _MAX_BYTES:
            return result("blocked", "empty_or_oversized_saved_evidence")
        raw = selected.read_bytes()
        after = selected.stat()
        if (
            not _safe_path(selected)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or len(raw) != after.st_size
            or not 0 < len(raw) <= _MAX_BYTES
        ):
            # A writer may have replaced or extended the file while it
            # was being read. Refuse the report before contacting PC-B.
            return result("blocked", "saved_evidence_changed_during_read")
        digest = hashlib.sha256(raw).hexdigest()
        saved = OwnerPairObservation.model_validate_json(raw)
    except (OSError, ValueError, TypeError):
        return result("blocked", "invalid_or_unreadable_saved_evidence")

    if expected_node is None or saved.node_id != expected_node:
        return result("blocked", "saved_evidence_renderer_node_mismatch")
    # Exclude stale or future-dated files *before* any remote request.
    if (
        saved.checked_utc.tzinfo is None
        or not -_MAX_CLOCK_SKEW <= current - saved.checked_utc <= _MAX_SAVED_AGE
    ):
        return result("stale", "saved_evidence_not_recent")
    checked, same_identity = _valid_saved_pair(saved, current)
    if not checked or not same_identity:
        return result("blocked", "saved_owner_evidence_internally_inconsistent")
    if offline:
        return result("saved_only", "file_consistent_but_not_authenticated")
    if primary is None:
        return result("unconfigured", "pc_a_primary_render_node_missing")
    node, cfg = primary
    if (
        not cfg.attestation_url
        or not cfg.attestation_token_env
        or not os.environ.get(cfg.attestation_token_env)
    ):
        return result("unconfigured", "pc_b_owner_bearer_not_configured")
    try:
        remote = owner_probe(node, cfg)
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError):
        return result("unavailable", "pc_b_live_owner_unavailable")
    bearer = True
    live = remote.audit
    live_pid = live.actual_listener_pid
    live_started = live.actual_process_started_utc
    if (
        remote.node_id != node
        or not _valid_owner(live, current=now())
        or saved.second_captured_utc is None
        or live.captured_utc <= saved.second_captured_utc
        or live.actual_listener_pid != saved.second_listener_pid
        or not same_native_creation_instant(
            live.actual_process_started_utc, saved.second_process_started_utc,
        )
        or live.launcher_pid != saved.second_launcher_pid
        or live.receipt_schema != saved.second_receipt_schema
    ):
        return result("blocked", "saved_and_current_owner_identity_do_not_correlate")
    return result("correlated_read_only", "saved_owner_identity_matches_live_pc_b")


