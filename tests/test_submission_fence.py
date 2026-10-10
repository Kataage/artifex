from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from artifex.comfy import ComfyUIClient, ComfyUIError
from artifex.comfy.admission import ComfySubmissionFence, SubmissionFencedError
from artifex.config.models import ComfyUiConfig
from artifex.db import Database
from artifex.domain import AgentState
from artifex.operations.fence import manage_submission_fence
from artifex.runtime import RuntimeStore


def _queue() -> dict[str, Any]:
    return {"queue_running": [], "queue_pending": []}


class FakeComfy:
    def __init__(self, queues: list[dict[str, Any]] | None = None) -> None:
        self.queues = queues or [_queue()]
        self.calls = 0

    async def queue_snapshot(self) -> dict[str, Any]:
        value = self.queues[min(self.calls, len(self.queues) - 1)]
        self.calls += 1
        return value


def _runtime(tmp_path: Path) -> tuple[Database, RuntimeStore]:
    db = Database(f"sqlite:///{tmp_path / 'controller.db'}")
    db.migrate()
    runtime = RuntimeStore(db)
    runtime.reconcile_process_start()
    runtime.set_agent_state(AgentState.RUNNING)
    return db, runtime


@pytest.mark.asyncio
async def test_fence_preview_is_read_only_and_seal_persists_between_instances(
    tmp_path: Path,
) -> None:
    file = tmp_path / "admission.sqlite"
    fence = ComfySubmissionFence(file)
    assert fence.status() is False
    assert not file.exists()
    assert await fence.seal(lambda: asyncio.sleep(0, result=True)) is True
    assert fence.status() is True
    other = ComfySubmissionFence(file)
    with pytest.raises(SubmissionFencedError):
        async with other.admit():
            pytest.fail("sealed admission must refuse a POST")
    await other.release()
    assert fence.status() is False
    async with other.admit():
        pass


@pytest.mark.asyncio
async def test_seal_rejects_false_validation_and_leaves_traffic_open(
    tmp_path: Path,
) -> None:
    fence = ComfySubmissionFence(tmp_path / "admission.sqlite")
    assert not await fence.seal(lambda: asyncio.sleep(0, result=False))
    assert fence.status() is False
    async with fence.admit():
        pass


@pytest.mark.asyncio
async def test_seal_waits_for_actual_inflight_prompt_before_blocking_new_requests(
    tmp_path: Path,
) -> None:
    """A request that obtained admission first is never interrupted by sealing."""
    started = asyncio.Event()
    release_request = asyncio.Event()
    posts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/prompt"
        posts.append(str(request.url))
        started.set()
        await release_request.wait()
        return httpx.Response(200, json={"prompt_id": "before-fence", "number": 0})

    path = tmp_path / "admission.sqlite"
    config = ComfyUiConfig(
        base_url="http://localhost:8188", submission_fence_path=path,
        request_attempts=1,
    )
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://localhost:8188",
    )
    client = ComfyUIClient(config, client=http_client)
    request = asyncio.create_task(client.submit({"1": {"class_type": "Test", "inputs": {}}}))
    await asyncio.wait_for(started.wait(), timeout=5)

    validator_called = asyncio.Event()

    async def validate() -> bool:
        validator_called.set()
        return True

    sealed = asyncio.create_task(ComfySubmissionFence(path).seal(validate))
    await asyncio.sleep(0.05)
    assert not validator_called.is_set(), "sealing raced an in-flight submission"
    assert not sealed.done()
    release_request.set()
    receipt = await asyncio.wait_for(request, timeout=5)
    assert receipt.prompt_id == "before-fence"
    assert await asyncio.wait_for(sealed, timeout=5) is True
    assert validator_called.is_set()

    with pytest.raises(SubmissionFencedError):
        await client.submit({"1": {"class_type": "Test", "inputs": {}}})
    assert len(posts) == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_submission_error_still_releases_admission_for_maintenance(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("disconnected", request=request)

    path = tmp_path / "admission.sqlite"
    cfg = ComfyUiConfig(
        base_url="http://localhost:8188",
        submission_fence_path=path, request_attempts=1,
    )
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://localhost:8188",
    )
    client = ComfyUIClient(cfg, client=http_client)
    with pytest.raises(ComfyUIError, match="outcome is unknown") as raised:
        await client.submit({"1": {"class_type": "Test", "inputs": {}}})
    assert raised.value.retryable is False
    assert await ComfySubmissionFence(path).seal(
        lambda: asyncio.sleep(0, result=True)
    )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_manage_fence_requires_true_paused_and_double_observed_idle(
    tmp_path: Path,
) -> None:
    db, runtime = _runtime(tmp_path)
    fence = ComfySubmissionFence(tmp_path / "admission.sqlite")
    queue = FakeComfy()
    preview = await manage_submission_fence(
        db, runtime, queue, fence,  # type: ignore[arg-type]
    )
    assert preview.mode == "preview" and preview.state == "open"
    assert not preview.submission_blocked and not preview.restart_authorized
    assert runtime.get_agent_state() is AgentState.RUNNING

    sealed = await manage_submission_fence(
        db, runtime, queue, fence,  # type: ignore[arg-type]
        apply=True, wait_seconds=5, poll_seconds=0.01,
    )
    assert sealed.state == "sealed" and sealed.submission_blocked
    assert sealed.drain is not None
    assert sealed.drain.ready is True
    assert sealed.drain.samples >= 2
    assert sealed.external_comfyui_clients_fenced is False
    assert sealed.restart_authorized is False
    assert runtime.get_agent_state() is AgentState.PAUSED
    assert queue.calls >= 4

    same = await manage_submission_fence(
        db, runtime, queue, fence,  # type: ignore[arg-type]
        apply=True,
    )
    assert same.state == "sealed"
    assert same.submission_blocked is True

    released = await manage_submission_fence(
        db, runtime, queue, fence,  # type: ignore[arg-type]
        apply=True, release=True,
    )
    assert released.state == "released"
    assert not fence.status()
    assert runtime.get_agent_state() is AgentState.PAUSED
    db.dispose()


@pytest.mark.asyncio
async def test_manage_fence_refuses_busy_queue_and_does_not_seal(
    tmp_path: Path,
) -> None:
    db, runtime = _runtime(tmp_path)
    fence = ComfySubmissionFence(tmp_path / "admission.sqlite")
    queue = FakeComfy([
        {"queue_running": [["other-app"]], "queue_pending": []}
    ])
    report = await manage_submission_fence(
        db, runtime, queue, fence,  # type: ignore[arg-type]
        apply=True, wait_seconds=0,
    )
    assert report.state == "blocked_on_drain"
    assert report.drain is not None and report.drain.status == "draining"
    assert not report.submission_blocked
    assert not fence.status()
    assert runtime.get_agent_state() is AgentState.PAUSED
    db.dispose()


@pytest.mark.asyncio
async def test_manage_fence_rechecks_under_lock_and_refuses_late_job(
    tmp_path: Path,
) -> None:
    db, runtime = _runtime(tmp_path)
    fence = ComfySubmissionFence(tmp_path / "admission.sqlite")
    queue = FakeComfy([
        _queue(), _queue(),
        {"queue_running": [[1]], "queue_pending": []},
    ])
    report = await manage_submission_fence(
        db, runtime, queue, fence,  # type: ignore[arg-type]
        apply=True, wait_seconds=1, poll_seconds=0.01,
    )
    assert report.state == "blocked_on_drain"
    assert not report.submission_blocked and not fence.status()
    assert report.drain is not None and report.drain.status == "draining"
    db.dispose()


@pytest.mark.asyncio
async def test_manage_fence_release_requires_paused_and_explicit_apply(
    tmp_path: Path,
) -> None:
    db, runtime = _runtime(tmp_path)
    fence = ComfySubmissionFence(tmp_path / "admission.sqlite")
    queue = FakeComfy()
    with pytest.raises(ValueError, match="requires --apply"):
        await manage_submission_fence(
            db, runtime, queue, fence,  # type: ignore[arg-type]
            release=True,
        )
    with pytest.raises(ValueError, match="PAUSED"):
        await manage_submission_fence(
            db, runtime, queue, fence,  # type: ignore[arg-type]
            apply=True, release=True,
        )
    assert not fence.status()
    db.dispose()


def test_fence_refuses_symlinked_database_path(tmp_path: Path) -> None:
    source = tmp_path / "real.sqlite"
    source.write_text("not a real database", encoding="utf-8")
    link = tmp_path / "alias.sqlite"
    try:
        link.symlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("host does not permit symlinks")
    fence = ComfySubmissionFence(link)
    with pytest.raises(ValueError, match="symlinked"):
        fence.status()
