from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from artifex.cli import _run_daemon_with_llm_manager
from artifex.config.models import ArtifexSettings, LlmConfig, LlmServerConfig
from artifex.llm.server import ManagedLlmServer


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, *, timeout: float) -> int:
        assert timeout > 0
        return self.returncode if self.returncode is not None else 0


def _config(tmp_path: Path, *, enabled: bool = True) -> LlmConfig:
    cfg = LlmConfig(
        base_url="http://127.0.0.1:8899",
        model="my-artifex-llm",
        server=LlmServerConfig(
            enabled=enabled,
            executable="llama-server",
            gpu_layers=18,
            device="CUDA0",
            startup_timeout_seconds=1.0,
            poll_seconds=0.01,
            restart_limit=1,
            restart_backoff_seconds=0,
            log_path=tmp_path / "llama.log",
        ),
    )
    cfg.bootstrap.models_dir = tmp_path / "models"
    model = cfg.bootstrap.model_path()
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"dummy GGUF")
    return cfg


def _client(
    processes: list[FakeProcess],
    *,
    external: bool = False,
    alias: str = "my-artifex-llm",
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path not in {"/health", "/v1/models"}:
            raise AssertionError(str(request.url))
        healthy = external or (
            bool(processes) and processes[-1].poll() is None
        )
        if not healthy:
            return httpx.Response(503)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200, json={"data": [{"id": alias, "object": "model"}]}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_managed_llama_starts_explicit_binary_and_owns_only_its_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr("artifex.llm.server._executable", lambda _: "/bin/llama-server")
    processes: list[FakeProcess] = []
    captured: list[dict[str, Any]] = []

    def factory(args: list[str], **kwargs: Any) -> FakeProcess:
        captured.append({"args": args, "kwargs": kwargs})
        process = FakeProcess()
        processes.append(process)
        return process

    async with _client(processes) as client:
        manager = ManagedLlmServer(cfg, client=client, process_factory=factory)
        await manager.start()
        assert len(processes) == 1
        args = captured[0]["args"]
        assert args[0] == "/bin/llama-server"
        assert args[args.index("--model") + 1] == str(cfg.bootstrap.model_path().resolve())
        assert args[args.index("--alias") + 1] == cfg.model
        assert args[args.index("--gpu-layers") + 1] == "18"
        assert args[args.index("--device") + 1] == "CUDA0"
        assert args[args.index("--host") + 1] == "127.0.0.1"
        assert captured[0]["kwargs"]["shell"] is False
        await manager.close()

    assert processes[0].terminated
    assert not processes[0].killed


@pytest.mark.asyncio
async def test_existing_matching_server_is_never_terminated_or_restarted(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    processes: list[FakeProcess] = []

    def fail_spawn(*_args: Any, **_kwargs: Any) -> FakeProcess:
        raise AssertionError("external healthy server must never spawn")

    async with _client(processes, external=True) as client:
        manager = ManagedLlmServer(cfg, client=client, process_factory=fail_spawn)
        await manager.start()
        assert manager.process is None
        await manager.watch()
        await manager.close()


@pytest.mark.asyncio
async def test_wrong_model_at_existing_port_fails_without_killing_it(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    async with _client([], external=True, alias="not-our-model") as client:
        manager = ManagedLlmServer(cfg, client=client)
        with pytest.raises(RuntimeError, match="expected model alias"):
            await manager.start()
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://192.168.1.40:8899",
    "http://0.0.0.0:8899",
    "https://127.0.0.1:8899",
    "http://127.0.0.1:8899/v1",
    "http://127.0.0.1:8899?q=1",
])
async def test_managed_llama_rejects_non_loopback_or_ambiguous_urls(
    tmp_path: Path, url: str
) -> None:
    cfg = _config(tmp_path)
    cfg.base_url = url
    async with _client([]) as client:
        with pytest.raises(ValueError, match="loopback"):
            await ManagedLlmServer(cfg, client=client).start()


@pytest.mark.asyncio
async def test_managed_llama_recovers_crashed_owned_child_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("artifex.llm.server._executable", lambda _: "/bin/llama-server")
    cfg = _config(tmp_path)
    processes: list[FakeProcess] = []
    calls = 0

    def factory(*_args: Any, **_kwargs: Any) -> FakeProcess:
        item = FakeProcess()
        processes.append(item)
        return item

    async def fake_sleep(_delay: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            processes[0].returncode = 1
        elif calls == 4:
            raise RuntimeError("test stopped monitoring")
        await asyncio.sleep(0)

    async with _client(processes) as client:
        manager = ManagedLlmServer(
            cfg, client=client, process_factory=factory, sleep=fake_sleep
        )
        await manager.start()
        with pytest.raises(RuntimeError, match="test stopped monitoring"):
            await manager.watch()
        assert len(processes) == 2
        await manager.close()
    assert processes[1].terminated


@pytest.mark.asyncio
async def test_managed_llama_respects_restart_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("artifex.llm.server._executable", lambda _: "/bin/llama-server")
    cfg = _config(tmp_path)
    cfg.server.restart_limit = 0
    processes: list[FakeProcess] = []

    def factory(*_args: Any, **_kwargs: Any) -> FakeProcess:
        item = FakeProcess()
        processes.append(item)
        return item

    async def fail_once(_delay: float) -> None:
        processes[0].returncode = 1

    async with _client(processes) as client:
        manager = ManagedLlmServer(
            cfg, client=client, process_factory=factory, sleep=fail_once
        )
        await manager.start()
        with pytest.raises(RuntimeError, match="restart budget"):
            await manager.watch()
        assert len(processes) == 1
        await manager.close()


@pytest.mark.asyncio
async def test_managed_llama_requires_gguf_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("artifex.llm.server._executable", lambda _: "/bin/llama-server")
    cfg = _config(tmp_path)
    cfg.bootstrap.model_path().unlink()
    async with _client([]) as client:
        manager = ManagedLlmServer(cfg, client=client)
        with pytest.raises(FileNotFoundError, match="Selected GGUF"):
            await manager.start()
        await manager.close()


@pytest.mark.asyncio
async def test_daemon_orchestration_always_closes_owned_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class StubManager:
        def __init__(self, _cfg: LlmConfig) -> None:
            pass

        async def start(self) -> None:
            events.append("start")

        async def watch(self) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                events.append("watch_cancel")

        async def close(self) -> None:
            events.append("close")

    class StubApplication:
        async def run(self) -> None:
            events.append("daemon")
            await asyncio.sleep(0)

    settings = ArtifexSettings()
    settings.llm.server.enabled = True
    monkeypatch.setattr("artifex.cli.ManagedLlmServer", StubManager)
    monkeypatch.setattr("artifex.cli.build_application", lambda _: StubApplication())
    await _run_daemon_with_llm_manager(settings)
    assert events[0] == "start"
    assert "daemon" in events
    assert events[-1] == "close"
