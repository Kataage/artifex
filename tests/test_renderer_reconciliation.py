from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy.reconciliation import (
    RendererReconcileReport,
    RendererReconcileSample,
    _queue_counts,
    reconcile_renderer,
)
from artifex.comfy.workflow_audit import (
    MissingWorkflowAsset,
    WorkflowAudit,
    WorkflowAuditEntry,
)
from artifex.config.models import ArtifexSettings, RenderNodeConfig


class FakeComfy:
    def __init__(self, queues: list[dict[str, Any] | Exception]) -> None:
        self.queues = queues
        self.calls = 0
        self.closed = False

    async def queue_snapshot(self) -> dict[str, Any]:
        value = self.queues[min(self.calls, len(self.queues) - 1)]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


def _audit(*, ready: bool) -> WorkflowAudit:
    missing = MissingWorkflowAsset(
        label="checkpoint",
        node_class="CheckpointLoaderSimple",
        input_name="ckpt_name",
        requested="missing.safetensors",
        available=(),
    )
    return WorkflowAudit(
        ready=ready, comfyui_url="http://127.0.0.1:8188",
        custom_node_folders=(),
        guidance=(),
        entries=(
            WorkflowAuditEntry(
                template_id="illust_main_v1",
                source_sha256="a" * 64,
                ready=ready,
                required_node_count=1,
                required_asset_count=1,
                missing_node_types=() if ready else ("UninstalledNode",),
                missing_assets=() if ready else (missing,),
                unverifiable_assets=(),
            ),
        ),
    )


def test_comfyui_queue_requires_explicit_valid_running_and_pending_arrays() -> None:
    assert _queue_counts({"queue_running": [], "queue_pending": []}) == ("idle", 0, 0)
    assert _queue_counts({
        "queue_running": [["prompt-A"]],
        "queue_pending": [["prompt-B"], ["prompt-C"]],
    }) == ("busy", 2, 1)
    for value in (
        None, {}, {"queue_running": []},
        {"queue_running": None, "queue_pending": []},
        {"queue_running": [], "queue_pending": "not a list"},
        ["queue_running", "queue_pending"],
    ):
        assert _queue_counts(value) == ("unknown", None, None)


@pytest.mark.asyncio
async def test_renderer_reconciliation_waits_for_real_idle_and_fresh_workflow_readiness() -> None:
    fake = FakeComfy([
        {"queue_running": [1], "queue_pending": []},
        {"queue_running": [], "queue_pending": []},
        {"queue_running": [], "queue_pending": []},
    ])
    passes: list[bool] = []
    states = iter([True, False, True])
    sleeps: list[float] = []

    async def audit(*args: Any, **kwargs: Any) -> WorkflowAudit:
        assert kwargs["client"] is fake
        ready = next(states)
        passes.append(ready)
        return _audit(ready=ready)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = await reconcile_renderer(
        ArtifexSettings(),
        client=fake, audit_fn=audit, sleep_fn=sleep,
        wait_seconds=60, poll_seconds=3,
    )
    assert result.ready and result.status == "ready"
    assert result.samples == 3
    assert result.last.queue_state == "idle"
    assert result.last.workflow_ready is True
    assert len(sleeps) == 2 and all(0 < x <= 3 for x in sleeps)
    assert passes == [True, False, True]
    assert fake.calls == 3
    assert result.restart_performed is False
    assert result.production_qualified is False
    assert result.next_actions == ("run_two_pc_deployment_verify",)


@pytest.mark.asyncio
async def test_renderer_reconciliation_busy_never_claims_ready_or_interrupts_gpu_jobs() -> None:
    fake = FakeComfy([{"queue_running": [1], "queue_pending": []}])
    async def audit(*args: Any, **kwargs: Any) -> WorkflowAudit:
        return _audit(ready=True)

    result = await reconcile_renderer(
        ArtifexSettings(), client=fake, audit_fn=audit,
    )
    assert result.status == "busy" and not result.ready
    assert result.samples == 1
    assert result.last.queued == 0 and result.last.running == 1
    assert "do_not_restart_or_interrupt_gpu_jobs" in result.next_actions
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_renderer_reconciliation_queue_unknown_even_when_workflow_loaded() -> None:
    fake = FakeComfy([{"queue_running": []}])
    async def audit(*args: Any, **kwargs: Any) -> WorkflowAudit:
        return _audit(ready=True)

    result = await reconcile_renderer(
        ArtifexSettings(), client=fake, audit_fn=audit,
    )
    assert result.status == "unverifiable_queue" and not result.ready
    assert result.last.queued is None
    assert "do_not_restart_or_assume_idle" in result.next_actions


@pytest.mark.asyncio
async def test_renderer_reconciliation_after_external_restart_without_touching_process() -> None:
    fake = FakeComfy([{"queue_running": [], "queue_pending": []}])
    async def audit(*args: Any, **kwargs: Any) -> WorkflowAudit:
        return _audit(ready=False)

    result = await reconcile_renderer(
        ArtifexSettings(), client=fake, audit_fn=audit,
    )
    assert result.status == "pending_runtime_refresh" and not result.ready
    assert result.last.missing_node_types == ("UninstalledNode",)
    assert result.last.missing_model_choices == ("checkpoint:missing.safetensors",)
    assert result.next_actions == (
        "review_missing_custom_nodes_before_any_restart",
        "inspect_approved_model_sources_and_existing_verified_files",
        "refresh_or_restart_owned_comfyui_only_after_queue_is_idle",
        "repeat_read_only_runtime_reconciliation",
    )


@pytest.mark.asyncio
async def test_renderer_reconciliation_reports_offline_network_not_ready() -> None:
    fake = FakeComfy([httpx.ConnectError("offline")])
    async def audit(*args: Any, **kwargs: Any) -> WorkflowAudit:
        raise httpx.ConnectError("offline")

    result = await reconcile_renderer(
        ArtifexSettings(), client=fake, audit_fn=audit,
    )
    assert result.status == "unreachable" and not result.ready
    assert "queue" in (result.last.error or "")
    assert "workflow" in (result.last.error or "")
    assert result.production_qualified is False


@pytest.mark.asyncio
async def test_renderer_reconciliation_rejects_mismatched_host_and_invalid_polling() -> None:
    bad = ArtifexSettings()
    bad.render_nodes.primary = "remote"
    bad.render_nodes.nodes = {
        "remote": RenderNodeConfig(
            base_url="http://pc-b.example:8188",
        )
    }
    with pytest.raises(ValueError, match="one renderer-local"):
        await reconcile_renderer(bad, client=FakeComfy([]))
    with pytest.raises(ValueError, match="wait_seconds"):
        await reconcile_renderer(ArtifexSettings(), wait_seconds=-1)
    with pytest.raises(ValueError, match="poll_seconds"):
        await reconcile_renderer(ArtifexSettings(), poll_seconds=0)


def test_cli_reconcile_renderer_is_read_only_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.write_text("agent: {}\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []
    sample = RendererReconcileSample(
        checked_at="2026-10-08T00:00:00+00:00", queue_state="idle",
        queued=0, running=0, workflow_ready=True,
    )

    async def fake_reconcile(*args: Any, **kwargs: Any) -> RendererReconcileReport:
        calls.append(kwargs)
        return RendererReconcileReport(
            status="ready", ready=True, samples=1, last=sample,
            next_actions=("run_two_pc_deployment_verify",),
        )

    monkeypatch.setattr("artifex.cli.reconcile_renderer", fake_reconcile)
    runner = CliRunner()
    success = runner.invoke(
        app,
        [
            "onboard", "reconcile-renderer", "--config", str(config),
            "--wait-seconds", "30", "--poll-seconds", "2",
        ],
    )
    assert success.exit_code == 0, success.output
    assert calls[-1]["wait_seconds"] == 30
    assert calls[-1]["require_idle"] is True
    assert '"production_qualified": false' in success.output

    async def not_ready(*args: Any, **kwargs: Any) -> RendererReconcileReport:
        return RendererReconcileReport(
            status="busy", ready=False, samples=1,
            last=sample.model_copy(update={"queue_state": "busy", "running": 1}),
            next_actions=("wait_for_comfyui_running_and_pending_jobs_to_finish",),
        )

    monkeypatch.setattr("artifex.cli.reconcile_renderer", not_ready)
    failure = runner.invoke(
        app, ["onboard", "reconcile-renderer", "--config", str(config)],
    )
    assert failure.exit_code == 1
    assert '"status": "busy"' in failure.output

    missing = runner.invoke(
        app, ["onboard", "reconcile-renderer", "--config", str(tmp_path / "missing.yaml")],
    )
    assert missing.exit_code == 1
    assert "existing non-symlink config required" in missing.output
