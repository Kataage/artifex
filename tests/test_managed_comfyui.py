from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config import load_settings
from artifex.config.models import (
    ArtifexSettings,
    ManagedComfyConfig,
)
from artifex.render_node.comfy_process import (
    ManagedComfyUI,
    serve_managed_renderer,
)
from artifex.setup_renderer import configure_renderer


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


def _settings(tmp_path: Path) -> ArtifexSettings:
    exe = tmp_path / "Comfy UI" / "python_embeded" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    comfy = tmp_path / "Comfy UI" / "ComfyUI"
    comfy.mkdir()
    return ArtifexSettings(
        render_agent={
            "node_id": "gpu-b",
            "comfyui_process": ManagedComfyConfig(
                enabled=True,
                executable=exe,
                working_directory=comfy,
                arguments=("main.py", "--listen", "0.0.0.0", "--port", "8188"),
                startup_timeout_seconds=2,
                poll_seconds=0.01,
                restart_limit=1,
                restart_backoff_seconds=0,
                log_path=tmp_path / "logs" / "comfyui.log",
            ),
        },
        comfyui={"base_url": "http://127.0.0.1:8188"},
    )


def _client(processes: list[FakeProcess], *, external: bool = False) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/system_stats"
        healthy = external or (bool(processes) and processes[-1].poll() is None)
        if not healthy:
            return httpx.Response(503)
        return httpx.Response(
            200, json={"system": {"comfyui_version": "0.9.0"}, "devices": [{"name": "GPU"}]}
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_managed_comfyui_spawns_expected_process_and_closes_only_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARTIFEX_RENDER_NODE_TOKEN", "supersecret")
    processes: list[FakeProcess] = []
    calls: list[dict[str, Any]] = []

    def factory(args: list[str], **kwargs: Any) -> FakeProcess:
        calls.append({"args": args, "kwargs": kwargs})
        item = FakeProcess()
        processes.append(item)
        return item

    settings = _settings(tmp_path)
    stop = threading.Event()
    with _client(processes) as client:
        manager = ManagedComfyUI(settings, client=client, process_factory=factory)
        manager.start(stop)
        assert len(processes) == 1
        assert calls[0]["args"] == [
            str(settings.render_agent.comfyui_process.executable.resolve()),
            "main.py", "--listen", "0.0.0.0", "--port", "8188",
        ]
        assert calls[0]["kwargs"]["cwd"] == str(
            settings.render_agent.comfyui_process.working_directory.resolve()
        )
        assert calls[0]["kwargs"]["shell"] is False
        assert "ARTIFEX_RENDER_NODE_TOKEN" not in calls[0]["kwargs"]["env"]
        manager.close()
    # Closing a supervisor is not proof the ComfyUI GPU queue is drained.\n    assert not processes[0].terminated and not processes[0].killed
    assert settings.render_agent.comfyui_process.log_path.is_file()


def test_healthy_external_comfyui_is_never_terminated_or_spawned(tmp_path: Path) -> None:
    def no_spawn(*_args: Any, **_kwargs: Any) -> FakeProcess:
        pytest.fail("preexisting ComfyUI must not be spawned or killed")
    stop = threading.Event()
    with _client([], external=True) as client:
        manager = ManagedComfyUI(_settings(tmp_path), client=client, process_factory=no_spawn)
        manager.start(stop)
        assert manager.process is None
        stop.set()
        manager.watch(stop)
        manager.close()


def test_managed_comfyui_rejects_remote_endpoints(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.comfyui.base_url = "http://192.168.1.50:8188"
    stop = threading.Event()
    with _client([]) as client:
        manager = ManagedComfyUI(settings, client=client)
        with pytest.raises(ValueError, match="renderer-local"):
            manager.start(stop)


def test_comfyui_restart_after_owned_crash(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []

    def factory(*_args: Any, **_kwargs: Any) -> FakeProcess:
        item = FakeProcess()
        processes.append(item)
        return item

    class StopAfterRestart(threading.Event):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def wait(self, timeout: float | None = None) -> bool:
            self.calls += 1
            if self.calls >= 2 and len(processes) >= 2:
                self.set()
            return super().wait(0)

    stop = StopAfterRestart()
    with _client(processes) as client:
        manager = ManagedComfyUI(_settings(tmp_path), client=client, process_factory=factory)
        manager.start(stop)
        processes[0].returncode = 1
        manager.watch(stop)
        assert len(processes) == 2
        manager.close()
    # Exited first child was replaced; the new live child is left intact.\n    assert not processes[1].terminated and not processes[1].killed


def test_comfyui_restart_budget_is_bounded(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []

    def factory(*_args: Any, **_kwargs: Any) -> FakeProcess:
        item = FakeProcess()
        processes.append(item)
        return item

    settings = _settings(tmp_path)
    settings.render_agent.comfyui_process.restart_limit = 0

    class InstantEvent(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            return False

    with _client(processes) as client:
        manager = ManagedComfyUI(settings, client=client, process_factory=factory)
        manager.start(InstantEvent())
        processes[0].returncode = 1
        with pytest.raises(RuntimeError, match="restart budget"):
            manager.watch(InstantEvent())
        manager.close()
    assert len(processes) == 1


def test_external_comfyui_outage_does_not_take_process_ownership(tmp_path: Path) -> None:
    status = {"ok": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if not status["ok"]:
            return httpx.Response(503)
        return httpx.Response(200, json={"system": {}, "devices": []})

    class InstantEvent(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            return False

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        manager = ManagedComfyUI(_settings(tmp_path), client=client)
        manager.start(threading.Event())
        status["ok"] = False
        with pytest.raises(RuntimeError, match="External ComfyUI"):
            manager.watch(InstantEvent())
        manager.close()
    assert manager.process is None


def test_managed_renderer_propagates_monitor_failure_to_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class StubManaged:
        def __init__(self, _settings: ArtifexSettings) -> None:
            pass

        def start(self, _stop: threading.Event) -> None:
            events.append("start")

        def watch(self, _stop: threading.Event) -> None:
            events.append("watch")
            raise RuntimeError("GPU driver stopped")

        def close(self) -> None:
            events.append("close")

    def fake_serve(_settings: ArtifexSettings, *, stop_event: threading.Event) -> None:
        assert stop_event.wait(2.0)

    monkeypatch.setattr("artifex.render_node.comfy_process.ManagedComfyUI", StubManaged)
    monkeypatch.setattr(
        "artifex.render_node.comfy_process.serve_attestation", fake_serve
    )
    with pytest.raises(RuntimeError, match="GPU driver stopped"):
        serve_managed_renderer(_settings(tmp_path))
    assert events == ["start", "watch", "close"]


def test_renderer_configure_comfy_launch_is_updatable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "render-node.yaml"
    exe = tmp_path / "python.exe"
    cwd = tmp_path / "ComfyUI"
    result = configure_renderer(
        ArtifexSettings(),
        output_path=output,
        node_id="gpu-b",
        comfy_executable=exe,
        comfy_working_directory=cwd,
        comfy_arguments=("main.py", "--port", "8188"),
    )
    assert result.comfyui_managed
    settings = load_settings(user_config=output, env={})
    result = configure_renderer(
        settings, output_path=output, comfy_arguments=("main.py", "--port", "8200"),
        update=True,
    )
    assert result.comfyui_managed
    saved = load_settings(user_config=output, env={}).render_agent.comfyui_process
    assert saved.executable == exe.resolve()
    assert saved.working_directory == cwd.resolve()
    assert saved.arguments == ("main.py", "--port", "8200")
    configure_renderer(
        load_settings(user_config=output, env={}),
        output_path=output, disable_comfy_management=True, update=True,
    )
    assert not load_settings(user_config=output, env={}).render_agent.comfyui_process.enabled


def test_renderer_configure_rejects_missing_workdir_and_conflicting_flags(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="working_directory"):
        configure_renderer(
            ArtifexSettings(), output_path=tmp_path / "bad.yaml",
            comfy_executable=tmp_path / "python.exe",
        )
    with pytest.raises(ValueError, match="conflicts"):
        configure_renderer(
            ArtifexSettings(), output_path=tmp_path / "bad.yaml",
            comfy_executable=tmp_path / "python.exe", disable_comfy_management=True,
        )


def test_render_node_cli_configures_comfy_launch_without_hardcoded_paths(
    tmp_path: Path,
) -> None:
    output = tmp_path / "node.yaml"
    result = CliRunner().invoke(
        app, [
            "render-node", "configure", "--output", str(output),
            "--comfy-exe", str(tmp_path / "python.exe"),
            "--comfy-workdir", str(tmp_path / "ComfyUI"),
            "--comfy-arg=main.py", "--comfy-arg=--listen",
            "--comfy-arg=0.0.0.0",
        ]
    )
    assert result.exit_code == 0, result.output
    config = load_settings(user_config=output, env={})
    assert config.render_agent.comfyui_process.enabled
    assert config.render_agent.comfyui_process.arguments == (
        "main.py", "--listen", "0.0.0.0"
    )


def test_managed_comfyui_disabled_preserves_original_renderer_behavior(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.render_agent.comfyui_process.enabled = False
    with _client([]) as client:
        manager = ManagedComfyUI(settings, client=client)
        manager.start(threading.Event())
        manager.watch(threading.Event())
        manager.close()
