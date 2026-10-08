from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

import httpx

from artifex.config.models import LlmConfig


def _local_endpoint(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "Managed llama-server requires a loopback http URL with no path, "
            "credentials, query or fragment. Remote providers remain unmanaged."
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid managed llama-server port") from exc
    if port is None:
        raise ValueError("Managed llama-server URL requires an explicit port")
    return parsed.hostname, port


def _executable(value: str) -> str:
    location = Path(value).expanduser()
    if location.is_file():
        return str(location.resolve())
    found = shutil.which(value)
    if found is not None:
        return found
    raise FileNotFoundError(
        f"llama-server executable not found: {value}; "
        "configure llm.server.executable or add llama-server to PATH"
    )


class ManagedLlmServer:
    """Supervise only a process spawned by this manager.

    Attaches to an already healthy, model-matching localhost endpoint but never
    kills or adopts an external process. A bounded restart budget avoids loops
    during driver/VRAM failures. The manager does not install llama.cpp.
    """

    def __init__(
        self,
        config: LlmConfig,
        *,
        client: httpx.AsyncClient | None = None,
        process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self._client = client
        self._process_factory = process_factory
        self._sleep = sleep
        self._log: Any = None
        self._restarts = 0
        self._attached_external = False

    async def _probe(self) -> bool:
        owned = self._client is None
        http = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(3.0), follow_redirects=False
        )
        try:
            try:
                base = self.config.base_url.rstrip("/")
                health = await http.get(base + "/health")
                if health.status_code != 200:
                    return False
                response = await http.get(base + "/v1/models")
                if response.status_code != 200:
                    return False
                payload: Any = response.json()
                items = payload.get("data", ()) if isinstance(payload, dict) else ()
                ids = {
                    item.get("id")
                    for item in items
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                } if isinstance(items, list) else set()
                if self.config.model not in ids:
                    raise RuntimeError(
                        "An LLM server already occupies the configured endpoint "
                        f"but does not serve the expected model alias '{self.config.model}'. "
                        "Use a different port or align the server model setting."
                    )
                return True
            except (httpx.HTTPError, ValueError, TypeError):
                return False
        finally:
            if owned:
                await http.aclose()

    def _spawn(self, executable: str, host: str, port: int) -> None:
        model = self.config.bootstrap.model_path().expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(
                f"Selected GGUF does not exist: {model}; "
                "run 'artifex llm bootstrap' or change the bootstrap profile"
            )
        log_path = self.config.server.log_path.expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log is None:
            self._log = log_path.open("ab")
        args = [
            executable,
            "--model", str(model),
            "--alias", self.config.model,
            "--host", host,
            "--port", str(port),
            "--ctx-size", str(self.config.context_window_tokens),
            "--gpu-layers", str(self.config.server.gpu_layers),
        ]
        if self.config.server.device is not None:
            args.extend(["--device", self.config.server.device])
        # Keep controller secrets out of the inference subprocess.
        env = {
            key: value for key, value in os.environ.items()
            if not (
                key.upper().startswith("ARTIFEX_")
                and (
                    "TOKEN" in key.upper()
                    or "SECRET" in key.upper()
                    or "API_KEY" in key.upper()
                )
            )
        }
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = self._process_factory(
            args,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            shell=False,
            creationflags=creationflags,
        )

    async def _wait_ready(self) -> None:
        deadline = monotonic() + self.config.server.startup_timeout_seconds
        while monotonic() < deadline:
            process = self.process
            if process is None or process.poll() is not None:
                raise RuntimeError(
                    "Managed llama-server exited before readiness. "
                    f"Inspect {self.config.server.log_path} for details."
                )
            if await self._probe():
                return
            await self._sleep(min(1.0, max(0.01, deadline - monotonic())))
        raise TimeoutError(
            "Managed llama-server did not become ready within "
            f"{self.config.server.startup_timeout_seconds}s. "
            f"Inspect {self.config.server.log_path}."
        )

    async def start(self) -> None:
        if not self.config.server.enabled:
            return
        if self.config.backend != "llama_cpp":
            raise ValueError("Managed llama-server requires llm.backend=llama_cpp")
        host, port = _local_endpoint(self.config.base_url)
        if await self._probe():
            self._attached_external = True
            return
        executable = _executable(self.config.server.executable)
        try:
            self._spawn(executable, host, port)
            await self._wait_ready()
        except BaseException:
            await self.close()
            raise

    async def watch(self) -> None:
        if not self.config.server.enabled or self._attached_external:
            return
        failures = 0
        while True:
            await self._sleep(self.config.server.poll_seconds)
            if self.process is None:
                return
            if self.process.poll() is None and await self._probe():
                failures = 0
                continue
            failures += 1
            if self.process.poll() is None and failures < 3:
                continue
            failures = 0
            if self._restarts >= self.config.server.restart_limit:
                raise RuntimeError(
                    "Managed llama-server exceeded bounded restart budget "
                    f"({self._restarts}); inspect {self.config.server.log_path}"
                )
            self._restarts += 1
            await self._stop_owned()
            await self._sleep(self.config.server.restart_backoff_seconds)
            if await self._probe():
                # Another already healthy server took over this port.
                self._attached_external = True
                return
            host, port = _local_endpoint(self.config.base_url)
            self._spawn(_executable(self.config.server.executable), host, port)
            await self._wait_ready()

    async def _stop_owned(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, timeout=5.0)

    async def close(self) -> None:
        try:
            await self._stop_owned()
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
