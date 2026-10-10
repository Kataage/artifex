"""Fail-closed, read-only PC-B owner and scheduler evidence collection.

This observes CIM, Task Scheduler and TCP state only. It never owns a
process, acquires an exclusive renderer lease or authorizes GPU restart.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from artifex.config.models import ArtifexSettings
from artifex.render_node.process_identity import (
    ComfyReceiptStore,
    WindowsProcessIdentity,
    matches_owned_process,
    windows_process_identity,
)
from artifex.render_node.socket_audit import audit_renderer_sockets
from artifex.windows_tasks import task_configuration_matches, task_status

CheckResult = Literal["pass", "fail", "unknown"]
AuditStatus = Literal["observed_stable", "blocked", "inconclusive", "unsupported"]


def _is_windows() -> bool:
    return os.name == "nt"


def _check(
    checks: dict[str, dict[str, str]], name: str, status: CheckResult, reason: str,
) -> None:
    checks[name] = {"status": status, "reason": reason}


def _same_process(
    before: WindowsProcessIdentity, after: WindowsProcessIdentity | None,
) -> bool:
    return after is not None and before == after


def observe_renderer_owner(
    settings: ArtifexSettings, *, config: Path,
) -> dict[str, Any]:
    """Observe a running renderer without starting, stopping or adopting it.

    Snapshot evidence is inherently non-atomic. Never use this report for
    restart permission, producer admission, or production PASS.
    """
    checks: dict[str, dict[str, str]] = {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "captured_utc": datetime.now(UTC).isoformat(),
        "status": "inconclusive",
        "checks": checks,
        "actual_listener_pid": None,
        "actual_process_started_utc": None,
        "launcher_pid": None,
        "receipt_schema": None,
        "scheduler_state": None,
        "process_observation_verified": False,
        "restart_authorized": False,
        "child_survival_qualified": False,
        "production_qualified": False,
        "mutated_services": False,
    }
    if not _is_windows():
        _check(checks, "native_windows", "fail", "requires native Windows PC-B")
        report["status"] = "unsupported"
        return report
    _check(checks, "native_windows", "pass", "native Windows process inventory available")

    config_settings = settings.render_agent
    if not config_settings.comfyui_process.enabled or not config_settings.gateway.enabled:
        _check(
            checks, "protected_configuration", "fail",
            "protected gateway and managed ComfyUI must both be enabled",
        )
    else:
        _check(checks, "protected_configuration", "pass", "protected ownership enabled")

    try:
        task = task_status("renderer")
        report["scheduler_state"] = task.state
        matching, _ = task_configuration_matches(
            "renderer", config=config, status=task,
        )
        _check(
            checks, "scheduler_policy",
            "pass" if matching else "fail",
            "installed task action and safety settings match" if matching
            else "task missing, unowned, unsafe or does not match selected configuration",
        )
        _check(
            checks, "scheduler_running",
            "pass" if (task.state or "").casefold() == "running" else "unknown",
            "scheduled renderer is running" if (task.state or "").casefold() == "running"
            else "task is not running; orphan reattachment not proven",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _check(checks, "scheduler_policy", "unknown", f"inspection failed: {type(exc).__name__}")
        _check(checks, "scheduler_running", "unknown", "task state could not be inspected")

    receipt_path = config_settings.comfyui_process.ownership_receipt_path
    try:
        receipt = ComfyReceiptStore(receipt_path).load()
    except (OSError, ValueError) as exc:
        receipt = None
        _check(checks, "receipt", "fail", f"invalid ownership receipt: {type(exc).__name__}")
    else:
        if receipt is None:
            _check(checks, "receipt", "fail", "no persisted Artifex-owned ComfyUI identity")
        elif receipt.schema_version not in {1, 2}:
            _check(checks, "receipt", "fail", "unsupported ownership receipt schema")
        else:
            _check(checks, "receipt", "pass", "bounded, valid persisted ownership receipt")
            report["receipt_schema"] = receipt.schema_version
            report["actual_listener_pid"] = receipt.identity.pid
            report["actual_process_started_utc"] = receipt.identity.started_utc
            if receipt.launcher_identity is not None:
                report["launcher_pid"] = receipt.launcher_identity.pid

    observed: WindowsProcessIdentity | None = None
    if receipt is not None and checks["receipt"]["status"] == "pass":
        try:
            observed = windows_process_identity(receipt.identity.pid)
            match = (
                observed is not None
                and matches_owned_process(settings, receipt, observed)
            )
            _check(
                checks, "process_identity", "pass" if match else "fail",
                "CIM identity and receipt match" if match
                else "process absent, PID recycled, arguments or ancestry drifted",
            )
            if receipt.launcher_identity is not None:
                # A dead launcher is allowed: it may have exited while the
                # original renderer remains alive. A reused PID is not allowed.
                launcher = windows_process_identity(receipt.launcher_identity.pid)
                launch_ok = launcher is None or launcher == receipt.launcher_identity
                _check(
                    checks, "launcher_identity",
                    "pass" if launch_ok else "fail",
                    "launcher original or safely absent" if launch_ok
                    else "recorded launcher PID now belongs to a different process",
                )
            else:
                _check(checks, "launcher_identity", "pass", "direct launch receipt")
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            _check(
                checks, "process_identity", "unknown",
                f"CIM process inspection failed: {type(exc).__name__}",
            )
            _check(checks, "launcher_identity", "unknown", "launcher not proven")
    else:
        _check(checks, "process_identity", "unknown", "valid receipt required")
        _check(checks, "launcher_identity", "unknown", "valid receipt required")

    # Re-sample TCP + CIM to detect changes during observation. Do not trust
    # a single matching PID or an HTTP health response as ownership evidence.
    try:
        first = audit_renderer_sockets(
            settings, owned_pid=receipt.identity.pid if receipt else None,
        )
        second = audit_renderer_sockets(
            settings, owned_pid=receipt.identity.pid if receipt else None,
        )
        stable = (
            first.status == second.status == "owned_loopback_observed"
            and first.listener_pids == second.listener_pids
            and first.listener_addresses == second.listener_addresses
            and first.unexpected_client_pids == second.unexpected_client_pids
        )
        _check(
            checks, "tcp_ownership",
            "pass" if stable else "fail",
            "two matching, exclusive loopback socket inventories" if stable
            else "owner, direct clients, exposure or TCP state unproven/changed",
        )
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        _check(checks, "tcp_ownership", "unknown", f"TCP inventory failed: {type(exc).__name__}")

    try:
        observed_after = (
            windows_process_identity(receipt.identity.pid)
            if receipt is not None and observed is not None
            else None
        )
        stable_identity = observed is not None and _same_process(observed, observed_after)
        stable_receipt = (
            receipt is not None
            and ComfyReceiptStore(receipt_path).load() == receipt
        )
        launcher_stable = True
        if receipt is not None and receipt.launcher_identity is not None:
            # The original venv shim may have exited naturally, but its PID
            # could also have been reused *during* our two TCP inspections.
            # A single earlier CIM result cannot establish final consistency.
            launcher_after = windows_process_identity(receipt.launcher_identity.pid)
            launcher_stable = (
                launcher_after is None or launcher_after == receipt.launcher_identity
            )
            if not launcher_stable:
                _check(
                    checks, "launcher_identity", "fail",
                    "launcher PID was reused during the owner observation",
                )
        consistent = stable_identity and stable_receipt and launcher_stable
        _check(
            checks, "snapshot_consistency",
            "pass" if consistent else "fail",
            "receipt, execution process and launcher unchanged during observation"
            if consistent else
            "receipt, execution process or launcher changed during observation",
        )
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        if (
            receipt is not None and receipt.launcher_identity is not None
            and checks["launcher_identity"]["status"] == "pass"
        ):
            _check(
                checks, "launcher_identity", "unknown",
                "launcher stability could not be rechecked",
            )
        _check(
            checks, "snapshot_consistency", "unknown",
            f"consistency check failed: {type(exc).__name__}",
        )
    required = (
        "native_windows", "protected_configuration", "scheduler_policy",
        "scheduler_running", "receipt", "process_identity",
        "launcher_identity", "tcp_ownership", "snapshot_consistency",
    )
    failures = any(checks[key]["status"] == "fail" for key in required)
    unknowns = any(checks[key]["status"] == "unknown" for key in required)
    report["status"] = "blocked" if failures else "inconclusive" if unknowns else "observed_stable"
    report["process_observation_verified"] = all(
        checks[key]["status"] == "pass"
        for key in ("receipt", "process_identity", "launcher_identity",
                    "tcp_ownership", "snapshot_consistency")
    )
    return report


def save_owner_observation(report: dict[str, Any], output: Path) -> Path:
    """Only explicit evidence output; refuse symlinks and overwrites."""
    target = output.expanduser().absolute()
    if any(part.is_symlink() for part in (target, *target.parents)):
        raise ValueError("Refusing symlinked owner-audit report destination")
    if target.exists():
        raise FileExistsError("Refusing to overwrite existing owner-audit evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(part.is_symlink() for part in (target, *target.parents)):
        raise ValueError("Refusing symlinked owner-audit report directory")
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        if target.exists():
            raise FileExistsError("Owner-audit report target appeared during write")
        # Exclusive creation prevents clobbering an existing report.
        with target.open("x", encoding="utf-8") as output_handle:
            output_handle.write(temp.read_text(encoding="utf-8"))
    finally:
        temp.unlink(missing_ok=True)
    return target
