from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable
from typing import IO
from urllib.parse import urlsplit

import httpx

from artifex.config.models import ArtifexSettings
from artifex.render_node.attestation import serve_attestation
from artifex.render_node.gateway import make_gateway_server, serve_gateway
from artifex.render_node.process_identity import (
    ComfyOwnershipReceipt,
    ComfyReceiptStore,
    expected_receipt,
    matches_owned_process,
    windows_process_identity,
)
from artifex.render_node.socket_audit import audit_renderer_sockets


def _local_comfy_url(value: str) -> None:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Managed ComfyUI has an invalid endpoint port") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or port is None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "Managed ComfyUI requires a renderer-local http://localhost:PORT "
            "or http://127.0.0.1:PORT URL; PC-A uses its separate LAN URL"
        )


class ManagedComfyUI:
    """Manage the renderer-local ComfyUI process only if we started it.

    Already healthy external instances are observed but never terminated.
    """

    def __init__(
        self,
        settings: ArtifexSettings,
        *,
        client: httpx.Client | None = None,
        process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self.settings = settings
        self.process: subprocess.Popen[bytes] | None = None
        self._client = client
        self._factory = process_factory
        self._log: IO[bytes] | None = None
        self._external = False
        self._restarts = 0
        self._adopted: ComfyOwnershipReceipt | None = None
        self._current_receipt: ComfyOwnershipReceipt | None = None
        self._receipts = ComfyReceiptStore(
            settings.render_agent.comfyui_process.ownership_receipt_path
        )

    def _healthy(self) -> bool:
        owns_client = self._client is None
        client = self._client or httpx.Client(
            timeout=httpx.Timeout(3.0), trust_env=False, follow_redirects=False
        )
        try:
            try:
                response = client.get(
                    self.settings.comfyui.base_url.rstrip("/") + "/system_stats"
                )
                if response.status_code != 200:
                    return False
                payload = response.json()
                return (
                    isinstance(payload, dict)
                    and isinstance(payload.get("system"), dict)
                    and isinstance(payload.get("devices"), list)
                )
            except (httpx.HTTPError, TypeError, ValueError):
                return False
        finally:
            if owns_client:
                client.close()

    def _start_owned(self) -> None:
        config = self.settings.render_agent.comfyui_process
        if config.executable is None or config.working_directory is None:
            raise ValueError("ComfyUI executable and working_directory must be configured")
        executable = config.executable.expanduser().resolve(strict=True)
        working_directory = config.working_directory.expanduser().resolve(strict=True)
        if not executable.is_file() or not working_directory.is_dir():
            raise ValueError("ComfyUI executable must be a file and working directory a folder")
        log_path = config.log_path.expanduser().resolve(strict=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log is None:
            self._log = log_path.open("ab")
        # Never leak Artifex's separate network tokens into the renderer process.
        env = {
            key: value
            for key, value in os.environ.items()
            if not (
                key.upper().startswith("ARTIFEX_")
                and any(word in key.upper() for word in ("TOKEN", "SECRET", "API_KEY"))
            )
        }
        self.process = self._factory(
            [str(executable), *config.arguments],
            cwd=str(working_directory),
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0,
        )

    def _wait_ready(self, stop: threading.Event) -> None:
        from time import monotonic

        config = self.settings.render_agent.comfyui_process
        deadline = monotonic() + config.startup_timeout_seconds
        while not stop.is_set() and monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited during startup; see {config.log_path}"
                )
            if self._healthy():
                self._verify_owned_listener()
                self._record_owned_process()
                return
            stop.wait(min(1.0, max(0.01, deadline - monotonic())))
        if stop.is_set():
            return
        raise TimeoutError(
            f"ComfyUI not ready after {config.startup_timeout_seconds}s; "
            f"inspect {config.log_path}"
        )

    def _verify_listener_for(self, pid: int) -> None:
        report = audit_renderer_sockets(
            self.settings, owned_pid=pid, gateway_pid=os.getpid(),
        )
        if report.status != "owned_loopback_observed":
            raise RuntimeError(
                "Protected ComfyUI socket ownership is not proven: "
                f"{report.status}; listener={report.listener_pids}; "
                f"direct_clients={report.unexpected_client_pids}; "
                "refusing automatic process shutdown or restart"
            )

    def _verify_owned_listener(self) -> None:
        if not self.settings.render_agent.gateway.enabled:
            return
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("Protected ComfyUI child is no longer alive")
        self._verify_listener_for(process.pid)

    def _require_free_upstream(self) -> None:
        # An unresponsive foreign process can still hold the port. Checking
        # only HTTP health does not establish that it is free for a new child.
        audit = audit_renderer_sockets(self.settings)
        if audit.status != "missing_listener":
            raise RuntimeError(
                "Protected ComfyUI upstream is not demonstrably unoccupied: "
                f"{audit.status}; refusing a competing child"
            )

    def _record_owned_process(self) -> None:
        if not self.settings.render_agent.gateway.enabled:
            return
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("Cannot record an exited ComfyUI process")
        identity = windows_process_identity(process.pid)
        if identity is None:
            raise RuntimeError("Cannot prove ComfyUI process start time")
        receipt = expected_receipt(self.settings, identity)
        if not matches_owned_process(self.settings, receipt, identity):
            raise RuntimeError("Launched ComfyUI executable/command line mismatch")
        self._receipts.save(receipt)
        self._current_receipt = receipt

    def _try_reattach(self) -> bool:
        """Adopt only the same Windows child that a past Artifex launched."""
        receipt = self._receipts.load()
        if receipt is None:
            return False
        observed = windows_process_identity(receipt.identity.pid)
        if observed is None:
            # Natural child death, with the TCP port demonstrably free, lets
            # the new supervisor safely launch a replacement.
            if self._healthy():
                raise RuntimeError("Foreign ComfyUI owns the old upstream port")
            self._require_free_upstream()
            self._receipts.retire(receipt)
            return False
        if not matches_owned_process(self.settings, receipt, observed):
            raise RuntimeError(
                "ComfyUI ownership receipt no longer matches the process; "
                "PID reuse, launch changes or foreign ownership suspected"
            )
        self._verify_listener_for(receipt.identity.pid)
        if not self._healthy():
            raise RuntimeError(
                "Verified original ComfyUI child remains alive but unhealthy; "
                "refusing to interrupt GPU work"
            )
        self._adopted = receipt
        self._current_receipt = receipt
        return True

    def _retire_exited_receipt(self) -> None:
        receipt = self._current_receipt
        if not self.settings.render_agent.gateway.enabled or receipt is None:
            return
        if windows_process_identity(receipt.identity.pid) is not None:
            raise RuntimeError(
                "Prior ComfyUI PID is still in use; refusing automatic replacement"
            )
        self._require_free_upstream()
        self._receipts.retire(receipt)
        self._current_receipt = None

    def start(self, stop: threading.Event) -> None:
        if not self.settings.render_agent.comfyui_process.enabled:
            return
        _local_comfy_url(self.settings.comfyui.base_url)
        if self.settings.render_agent.gateway.enabled:
            # A LAN-facing authenticated gateway can claim a single writer
            # only when Artifex owns the *locally loopback-bound* ComfyUI.
            args = self.settings.render_agent.comfyui_process.arguments
            listen_values: list[str] = []
            port_values: list[str] = []
            for index, arg in enumerate(args):
                if arg == "--listen":
                    listen_values.append(args[index + 1] if index + 1 < len(args) else "")
                elif arg.startswith("--listen="):
                    listen_values.append(arg.partition("=")[2])
                elif arg == "--port":
                    port_values.append(args[index + 1] if index + 1 < len(args) else "")
                elif arg.startswith("--port="):
                    port_values.append(arg.partition("=")[2])
            if listen_values != ["127.0.0.1"]:
                raise ValueError(
                    "Gateway-managed ComfyUI requires exactly one explicit "
                    "'--listen 127.0.0.1' option; do not expose its upstream LAN port"
                )
            expected_port = urlsplit(self.settings.comfyui.base_url).port
            if port_values != [str(expected_port)]:
                raise ValueError(
                    "Gateway-managed ComfyUI requires exactly one '--port PORT' "
                    "matching the local upstream URL"
                )
        if self.settings.render_agent.gateway.enabled:
            if self._try_reattach():
                return
            if self._healthy():
                raise RuntimeError(
                    "Gateway refuses already-running ComfyUI without a "
                    "matching verified ownership receipt"
                )
            self._require_free_upstream()
        elif self._healthy():
            self._external = True
            return
        try:
            self._start_owned()
            self._wait_ready(stop)
        except BaseException:
            self.close()
            raise

    def watch(self, stop: threading.Event) -> None:
        config = self.settings.render_agent.comfyui_process
        if not config.enabled:
            return
        failures = 0
        while not stop.wait(config.poll_seconds):
            healthy = self._healthy()
            if self._external:
                if not healthy:
                    raise RuntimeError(
                        "External ComfyUI is unavailable; Artifex cannot restart a "
                        "process it does not own"
                    )
                continue
            process = self.process
            if process is None:
                return
            if healthy and process.poll() is None:
                self._verify_owned_listener()
                failures = 0
                continue
            failures += 1
            if process.poll() is None:
                if failures < 3:
                    continue
                # HTTP health checks can fail while a live renderer is still
                # generating. No one has fenced PC-B loopback submissions,
                # so never terminate/kill a living renderer to restart it.
                raise RuntimeError(
                    "Managed ComfyUI stopped answering health checks but its "
                    "child process is still alive; refusing an unsafe restart"
                )
            failures = 0
            if self._restarts >= config.restart_limit:
                raise RuntimeError(
                    "Managed ComfyUI exhausted the configured restart budget; "
                    f"inspect {config.log_path}"
                )
            self._restarts += 1
            self._stop_owned()
            if stop.wait(config.restart_backoff_seconds):
                return
            if self._healthy():
                # A different process may have claimed the local port after
                # our child exited; never silently attach a protected gateway
                # to an unowned renderer.
                if self.settings.render_agent.gateway.enabled:
                    raise RuntimeError(
                        "Protected gateway lost exclusive ComfyUI ownership; "
                        "another process claimed the upstream port"
                    )
                self._external = True
                return
            self._start_owned()
            self._wait_ready(stop)

    def _stop_owned(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        # A stopped *supervisor* is not evidence that ComfyUI's GPU queue
        # and external loopback clients are idle. Detach a live child instead
        # of sending terminate()/kill() on task shutdown or crash. An
        # intentional owner-verified shutdown requires a separate protocol.

    def close(self) -> None:
        try:
            self._stop_owned()
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None


def serve_managed_renderer(settings: ArtifexSettings) -> None:
    """Run authenticated attestation and optional ComfyUI child as one task.

    A fatal ComfyUI supervision error shuts down the agent too so Windows
    Task Scheduler can restart the Artifex task within its own retry budget.
    """
    stop = threading.Event()
    manager = ManagedComfyUI(settings)
    errors: list[Exception] = []

    def supervise() -> None:
        try:
            manager.watch(stop)
        except Exception as exc:  # noqa: BLE001 - relay monitor failure to main thread
            errors.append(exc)
            stop.set()

    monitor: threading.Thread | None = None
    gateway_thread: threading.Thread | None = None
    try:
        manager.start(stop)
        if settings.render_agent.gateway.enabled:
            # Binding on the main thread makes startup fail immediately when
            # the protected gateway port is already claimed.
            server = make_gateway_server(settings)
            def serve_protected_gateway() -> None:
                try:
                    serve_gateway(server, stop)
                except Exception as exc:  # noqa: BLE001 - propagate fatal listener errors
                    errors.append(exc)
                    stop.set()

            gateway_thread = threading.Thread(
                target=serve_protected_gateway,
                name="artifex-render-gateway",
                daemon=True,
            )
            gateway_thread.start()
        if settings.render_agent.comfyui_process.enabled:
            monitor = threading.Thread(
                target=supervise, name="artifex-comfy-watch", daemon=True
            )
            monitor.start()
        serve_attestation(settings, stop_event=stop)
        if errors:
            raise RuntimeError(f"Renderer supervision failed: {errors[0]}") from errors[0]
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=5.0)
        if gateway_thread is not None:
            gateway_thread.join(timeout=5.0)
        manager.close()
