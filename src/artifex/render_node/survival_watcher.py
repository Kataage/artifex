"""Detached read-only PC-B Windows owner-survival watcher.

MUST be scheduled as its own OS process, NOT as a thread in the renderer
supervisor. Never launches, terminates, restarts or re-parents a GPU job.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from artifex.config.models import ArtifexSettings
from artifex.render_node.owner_audit import save_owner_observation
from artifex.render_node.observer_heartbeat import publish_observer_heartbeat
from artifex.render_node.supervisor_survival import (
    SurvivalAssessment,
    SurvivalSample,
    assess_survival,
    sample_supervisor_survival,
)
from artifex.render_node.survival_spool import automatic_survival_output

_MIN_POLL = 5.0
_MAX_POLL = 300.0
_MAX_WINDOW = 24 * 3600.0
_MAX_SAMPLES = 4096


@dataclass
class PassiveSurvivalWindow:
    """Arm only on actual verified Running; retain exact supervisor identities."""

    interval_seconds: float
    window_seconds: float
    samples: list[SurvivalSample] = field(default_factory=list)
    original_pids: tuple[int, ...] = ()
    first_monotonic: float | None = None
    last_monotonic: float | None = None

    def clear(self) -> None:
        self.samples.clear()
        self.original_pids = ()
        self.first_monotonic = None
        self.last_monotonic = None

    @staticmethod
    def _can_arm(sample: SurvivalSample) -> bool:
        # Source assessor validates ALL Task Scheduler policy/receipt/PID/
        # socket checks, so this cannot arm on "Running" alone.
        return assess_survival(
            (sample.model_copy(update={"elapsed_seconds": 0.0}),),
            min_separation_seconds=5,
        ).status == "inconclusive"

    def ingest(
        self, observation: SurvivalSample, *, monotonic_seconds: float,
    ) -> SurvivalAssessment | None:
        """Return only terminal/incomplete events worth persisting in the spool."""
        if self.first_monotonic is None:
            if not self._can_arm(observation):
                return None  # Do not spam thousands of unverified reports.
            self.first_monotonic = monotonic_seconds
            self.last_monotonic = monotonic_seconds
            self.original_pids = tuple(
                pid for pid, _ in observation.supervisor_identities
            )
            self.samples = [observation.model_copy(update={"elapsed_seconds": 0.0})]
            return None

        assert self.last_monotonic is not None
        elapsed = max(0.0, monotonic_seconds - self.first_monotonic)
        sampled = observation.model_copy(update={"elapsed_seconds": elapsed})

        # The source assessor checks monotonic gaps, so any long wall clock
        # delay must terminate this window WITHOUT claiming a successful exit.
        self.samples.append(sampled)
        result = assess_survival(
            tuple(self.samples), min_separation_seconds=self.interval_seconds,
        )
        if result.status != "inconclusive":
            self.clear()
            return result

        # Only a naturally witnessed transition can become a success. A
        # healthy unchanging renderer never fills the disk with no-op reports.
        # With a lone post-exit sample, persist inconclusive evidence rather
        # than losing or falsely upgrading it on the next window.
        if (
            elapsed >= self.window_seconds
            or len(self.samples) >= _MAX_SAMPLES - 1
        ):
            transition_seen = any(
                item.task_state == "Ready" for item in self.samples[1:]
            )
            self.clear()
            return result if transition_seen else None
        return None


def run_passive_survival_watcher(
    settings: ArtifexSettings,
    *,
    config: Path,
    poll_seconds: float = 15,
    window_seconds: float = 3600,
    max_seconds: float = 0,
    sample: Callable[..., SurvivalSample] = sample_supervisor_survival,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    output: Callable[[ArtifexSettings], Path] = automatic_survival_output,
    persist: Callable[[dict[str, Any], Path], Path] = save_owner_observation,
    publish_heartbeat: Callable[..., None] = publish_observer_heartbeat,
) -> dict[str, object]:
    """Loop independently of the renderer; never cause a task transition.

    A zero max_seconds means indefinite service mode. A nonzero bound exists
    for controlled operational diagnostics and deterministic CI tests.
    """
    if os.name != "nt":
        raise OSError("Survival watcher requires native Windows PC-B")
    if not config.is_file() or any(
        part.is_symlink() for part in (config, *config.parents)
    ):
        raise ValueError("Existing non-symlinked PC-B configuration required")
    if not _MIN_POLL <= poll_seconds <= _MAX_POLL:
        raise ValueError("Poll interval must be 5-300 seconds")
    if (
        not 2 * poll_seconds <= window_seconds <= _MAX_WINDOW
        or window_seconds / poll_seconds > _MAX_SAMPLES - 2
    ):
        raise ValueError("Watch window must permit two samples and fit bounded memory")
    if max_seconds < 0 or max_seconds > _MAX_WINDOW:
        raise ValueError("Max watch duration must be 0-86400 seconds")
    # Validate spool path before polling; never create a file on GET.
    output(settings)
    watcher = PassiveSurvivalWindow(
        interval_seconds=poll_seconds, window_seconds=window_seconds,
    )
    started = monotonic()
    saved = 0
    samples = 0
    last_event: Literal[
        "observed_after_supervisor_absence", "inconclusive", "blocked",
    ] | None = None
    while True:
        current = monotonic()
        original = watcher.original_pids
        reading = sample(
            settings, config=config, elapsed_seconds=0,
            original_supervisor_ids=original, now=now,
        )
        samples += 1
        verdict = watcher.ingest(reading, monotonic_seconds=current)
        publish_heartbeat(
            settings,
            observed_utc=reading.observed_utc,
            poll_seconds=poll_seconds,
            sample_count=samples,
            state=reading.state,
        )
        if verdict is not None:
            # Automatic deterministic naming with exclusive file creation;
            # no task/renderer/receipt mutations, ever.
            persist(verdict.model_dump(mode="json"), output(settings))
            saved += 1
            last_event = verdict.status
        elapsed = max(0.0, monotonic() - started)
        if max_seconds and elapsed >= max_seconds:
            break
        delay = poll_seconds
        if max_seconds:
            delay = min(delay, max_seconds - elapsed)
        sleep(delay)
    return {
        "schema_version": 1,
        "status": "bounded_watch_completed",
        "node_id": settings.render_agent.node_id,
        "samples": samples,
        "saved_traces": saved,
        "last_event": last_event,
        "gpu_jobs_submitted": False,
        "scheduler_actions_executed": False,
        "services_mutated": False,
        "renderer_restart_authorized": False,
        "child_survival_qualified": False,
        "production_qualified": False,
    }
