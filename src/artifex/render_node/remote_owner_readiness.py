"""Local background-only PC-B launcher/owner evidence cache for authenticated GET.

The HTTP handler *never* launches Python, GPU work or changes services. Only
the local renderer-attestation service's bounded worker runs PR #116's
disposable, no-GPU launcher fixture on a protected native Windows node.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Event, Lock
from typing import Any

from artifex.config.models import ArtifexSettings
from artifex.render_node.owner_readiness import collect_owner_readiness

_REFRESH_SECONDS = 240.0
_MAX_CACHE_AGE_SECONDS = 600.0
_MAX_RESPONSE_BYTES = 128 * 1024


def _safe_config(path: Path | None) -> bool:
    return (
        path is not None
        and not any(item.is_symlink() for item in (path, *path.parents))
        and path.is_file()
    )


def can_collect_owner_readiness(
    settings: ArtifexSettings, owner_config: Path | None,
) -> bool:
    """Do not create even disposable child processes on unmanaged/non-PC-B."""
    agent = settings.render_agent
    return (
        os.name == "nt"
        and _safe_config(owner_config)
        and agent.gateway.enabled
        and agent.comfyui_process.enabled
        and agent.comfyui_process.executable is not None
        and agent.require_token
        and bool(agent.token_env and os.environ.get(agent.token_env))
    )


def _redact(report: dict[str, Any]) -> dict[str, Any]:
    """Drop fixture command lines/paths not needed for remote correlation."""
    copy = dict(report)
    copy["configured_comfyui_python"] = None
    fixture = copy.get("launcher_fixture")
    if isinstance(fixture, dict):
        stripped = dict(fixture)
        stripped["launcher_identity"] = None
        stripped["listener_identity"] = None
        copy["launcher_fixture"] = stripped
    return copy


class OwnerReadinessCache:
    """Single writer, bounded lifetime, no file writes and no on-demand probe."""

    def __init__(
        self, settings: ArtifexSettings, *, owner_config: Path | None,
    ) -> None:
        self.settings = settings
        self.owner_config = owner_config
        self._lock = Lock()
        self._snapshot: dict[str, Any] | None = None
        self._updated_monotonic = 0.0

    def snapshot(self) -> dict[str, Any] | None:
        if not can_collect_owner_readiness(self.settings, self.owner_config):
            return None
        with self._lock:
            if (
                self._snapshot is None
                or time.monotonic() - self._updated_monotonic > _MAX_CACHE_AGE_SECONDS
            ):
                return None
            return dict(self._snapshot)

    def _collect_once(self) -> None:
        if not can_collect_owner_readiness(self.settings, self.owner_config):
            with self._lock:
                self._snapshot = None
            return
        assert self.owner_config is not None
        try:
            report = _redact(collect_owner_readiness(
                self.settings, config=self.owner_config,
            ))
            # A bounded HTTP reply; no paths or raw exception strings on failure.
            if len(json.dumps(report, default=str).encode("utf-8")) > _MAX_RESPONSE_BYTES:
                raise ValueError("owner readiness report too large")
        except Exception:  # noqa: BLE001 - do not disclose paths or tokens
            with self._lock:
                self._snapshot = None
            return
        with self._lock:
            self._snapshot = report
            self._updated_monotonic = time.monotonic()

    def run(self, stop: Event) -> None:
        """Worker runs locally at most once per four minutes; never on GET."""
        while not stop.is_set():
            self._collect_once()
            if stop.wait(_REFRESH_SECONDS):
                break
