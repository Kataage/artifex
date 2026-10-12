"""Fail-closed, single-sample check before starting an eight-hour soak.

The probe runs only existing GET/database/telemetry reads. It cannot establish
an eight-hour result and never initiates GPU work or writes a session.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict

from artifex.config.models import ArtifexSettings
from artifex.qualification.native_creation_time import native_creation_instant
from artifex.qualification.renderer_owner_evidence import _REQUIRED_CHECKS
from artifex.qualification.soak_observer import (
    _REQUIRED,
    SoakSample,
    sample_soak,
)

StartIssue = Literal[
    "sample_unavailable",
    "primary_renderer_unconfigured",
    "sample_clock_invalid",
    "llm_unavailable",
    "comfyui_unavailable",
    "renderer_attestation_unavailable",
    "sample_reported_errors",
    "resource_metrics_missing",
    "database_metrics_missing",
    "required_asset_hashes_missing",
    "renderer_owner_missing",
    "renderer_owner_unverified",
    "renderer_owner_identity_invalid",
    "renderer_owner_snapshot_stale",
]


class SoakStartPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    mode: Literal["read_only"] = "read_only"
    ready: bool
    issues: tuple[StartIssue, ...]
    sample_checked_utc: datetime
    # This gate is not the session and is not the full soak trace.
    qualification_session_created: Literal[False] = False
    soak_evidence_file_written: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    services_modified: Literal[False] = False
    production_qualified: Literal[False] = False


def check_soak_start_sample(
    settings: ArtifexSettings,
    sample: SoakSample | None,
    *, checked_utc: datetime | None = None,
) -> SoakStartPreflight:
    now = checked_utc or datetime.now(UTC)
    issues: list[StartIssue] = []
    if settings.render_nodes.primary_node() is None:
        issues.append("primary_renderer_unconfigured")
    if sample is None:
        issues.append("sample_unavailable")
    else:
        if (
            sample.observed_at.tzinfo is None
            or sample.observed_at.utcoffset() is None
            or not -timedelta(seconds=30)
            <= now - sample.observed_at <= timedelta(seconds=90)
        ):
            issues.append("sample_clock_invalid")
        if not sample.llm_ok:
            issues.append("llm_unavailable")
        if not sample.comfyui_ok:
            issues.append("comfyui_unavailable")
        if not sample.renderer_ok:
            issues.append("renderer_attestation_unavailable")
        if sample.errors:
            # Never echo arbitrary strings from a failed HTTP/driver/DB probe.
            issues.append("sample_reported_errors")
        if None in (sample.gpu_used_mib, sample.ram_used_mib, sample.free_disk_gib):
            issues.append("resource_metrics_missing")
        if None in (
            sample.telemetry_rows, sample.severe_events, sample.finalized_packs,
        ):
            issues.append("database_metrics_missing")
        if not _REQUIRED <= sample.asset_sha256.keys() or any(
            not isinstance(sample.asset_sha256[label], str)
            or len(sample.asset_sha256[label]) != 64
            or any(c not in "0123456789abcdefABCDEF"
                   for c in sample.asset_sha256[label])
            for label in _REQUIRED if label in sample.asset_sha256
        ):
            issues.append("required_asset_hashes_missing")
        owner_required = bool(
            settings.render_nodes.primary_node() is not None
            and settings.qualification.require_renderer_owner_observation
        )
        if owner_required:
            owner = sample.owner_observation
            primary = settings.render_nodes.primary_node()
            node_id = primary[0] if primary else None
            if owner is None:
                issues.append("renderer_owner_missing")
            else:
                if (
                    owner.node_id != node_id
                    or owner.status != "observed_stable"
                    or not owner.process_observation_verified
                    or not _REQUIRED_CHECKS.issubset(owner.checks)
                    or any(owner.checks[key] != "pass" for key in _REQUIRED_CHECKS)
                ):
                    issues.append("renderer_owner_unverified")
                started = native_creation_instant(
                    owner.actual_process_started_utc,
                )
                if (
                    owner.actual_listener_pid is None
                    or started is None
                    or owner.captured_utc.tzinfo is None
                    or owner.captured_utc.utcoffset() is None
                    or (
                        owner.captured_utc.tzinfo is not None
                        and started is not None
                        and started > owner.captured_utc
                    )
                ):
                    issues.append("renderer_owner_identity_invalid")
                if (
                    sample.observed_at.tzinfo is None
                    or sample.observed_at.utcoffset() is None
                    or owner.captured_utc.tzinfo is None
                    or owner.captured_utc.utcoffset() is None
                    or abs((sample.observed_at - owner.captured_utc).total_seconds())
                    > 90
                ):
                    issues.append("renderer_owner_snapshot_stale")
    return SoakStartPreflight(
        ready=not issues,
        issues=tuple(dict.fromkeys(issues)),
        sample_checked_utc=now,
    )


def preflight_soak_start(settings: ArtifexSettings) -> SoakStartPreflight:
    """Perform one actual read-only sample, no session creation or evidence write."""
    try:
        sample = sample_soak(settings, 0.0)
    except Exception:  # noqa: BLE001 - status-only failure, no secret echo
        sample = None
    return check_soak_start_sample(settings, sample)
