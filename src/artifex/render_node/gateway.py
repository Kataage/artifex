from __future__ import annotations

import hmac
import json
import re
import sqlite3
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event
from typing import Any, Iterator
from urllib.parse import urlsplit

import httpx

from artifex.comfy.admission import ComfySubmissionFence
from artifex.config.models import ArtifexSettings

_MAX_POST = 8 * 1024 * 1024
_MAX_RESPONSE = 256 * 1024 * 1024
_HISTORY = re.compile(r"^/history/[A-Za-z0-9_-]{1,128}$")
_GET = frozenset(("/system_stats", "/object_info", "/queue", "/history", "/view"))
_POST = frozenset(("/prompt", "/free", "/interrupt", "/queue"))


def _strict_local_upstream(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.port is None
        or parsed.path not in {"", "/"}
        or parsed.query or parsed.fragment
        or parsed.username is not None or parsed.password is not None
    ):
        raise ValueError(
            "Gateway upstream must be a local http://127.0.0.1:PORT ComfyUI"
        )


class RendererGatewayAdmission:
    """Durable PC-B barrier around admitted, authenticated HTTP mutations.

    A single SQLite writer transaction spans the whole upstream POST, so
    sealing cannot overtake a previously admitted request. This does not
    control local clients that bypass the gateway to talk to loopback ComfyUI.
    """

    def __init__(self, path: Path) -> None:
        self._fence = ComfySubmissionFence(path)

    @property
    def blocked(self) -> bool:
        return self._fence.status()

    @contextmanager
    def admit(self) -> Iterator[bool]:
        conn = self._fence._begin()
        try:
            yield not self._fence._blocked(conn)
        finally:
            self._fence._finish(conn, commit=False)

    def seal_if_idle(self, queue_snapshot: object) -> bool:
        # Called after acquiring the SAME writer lock as /prompt.
        with self.admit() as allowed:
            if not allowed:
                return True
            if not isinstance(queue_snapshot, dict):
                return False
            running = queue_snapshot.get("queue_running")
            pending = queue_snapshot.get("queue_pending")
            if not isinstance(running, list) or not isinstance(pending, list):
                return False
            if running or pending:
                return False
        # The gap between the two transactions would reopen the race!
        # The real implementation below performs recheck and update under
        # a single admission transaction; this method is not used.
        raise RuntimeError("use seal_with_queue for atomic gateway sealing")

    def seal_with_queue(self, get_queue: Any) -> tuple[bool, str]:
        conn = self._fence._begin()
        commit = False
        try:
            if self._fence._blocked(conn):
                return True, "already_sealed"
            snapshot = get_queue()
            if not isinstance(snapshot, dict):
                return False, "queue_unverifiable"
            running, pending = snapshot.get("queue_running"), snapshot.get("queue_pending")
            if not isinstance(running, list) or not isinstance(pending, list):
                return False, "queue_unverifiable"
            if running or pending:
                return False, "queue_busy"
            conn.execute(
                "INSERT INTO submission_fence(id, blocked) VALUES (1, 1) "
                "ON CONFLICT(id) DO UPDATE SET blocked = 1"
            )
            commit = True
            return True, "sealed"
        finally:
            self._fence._finish(conn, commit=commit)

    def release(self) -> None:
        conn = self._fence._begin()
        try:
            conn.execute(
                "INSERT INTO submission_fence(id, blocked) VALUES (1, 0) "
                "ON CONFLICT(id) DO UPDATE SET blocked = 0"
            )
            self._fence._finish(conn, commit=True)
        except BaseException:
            conn.close()
            raise


def make_gateway_server(
    settings: ArtifexSettings,
    *,
    upstream: httpx.Client | None = None,
    bind_host: str | None = None,
    port: int | None = None,
) -> ThreadingHTTPServer:
    """Construct (and bind) an explicitly opt-in authenticated API gateway.

    Never forward arbitrary paths, redirects, request headers or credentials
    to the local ComfyUI upstream. This is a small HTTP API proxy, not a
    generic proxy for the ComfyUI UI, WebSocket or custom-node endpoints.
    """
    cfg = settings.render_agent.gateway
    if not cfg.enabled:
        raise ValueError("Render gateway must be explicitly enabled")
    if not settings.render_agent.comfyui_process.enabled:
        raise ValueError("Render gateway requires owned managed ComfyUI")
    _strict_local_upstream(settings.comfyui.base_url)
    if (
        cfg.port == settings.render_agent.port
        or cfg.port == urlsplit(settings.comfyui.base_url).port
    ):
        raise ValueError("Gateway, attestation and ComfyUI ports must differ")
    token_name = settings.render_agent.token_env
    import os

    token = os.environ.get(token_name, "") if token_name else ""
    if not token or len(token) < 24:
        raise ValueError("Render gateway requires a configured token of at least 24 chars")
    if not settings.render_agent.require_token:
        raise ValueError("Render gateway requires authenticated attestation")
    gateway = RendererGatewayAdmission(cfg.admission_path)
    own_client = upstream is None
    proxy_client = upstream or httpx.Client(
        base_url=settings.comfyui.base_url.rstrip("/"),
        timeout=httpx.Timeout(settings.comfyui.timeout_seconds),
        follow_redirects=False,
        trust_env=False,
    )

    def fetch_queue() -> object:
        response = proxy_client.get("/queue")
        response.raise_for_status()
        return response.json()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ArtifexRenderGateway/1"

        def _json(self, code: HTTPStatus, body: dict[str, Any]) -> None:
            payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self._body(code, payload, "application/json")

        def _body(self, code: HTTPStatus, body: bytes, mime: str) -> None:
            self.send_response(code.value)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _authorized(self) -> bool:
            authorization = self.headers.get("Authorization", "")
            expected = "Bearer " + token
            return hmac.compare_digest(authorization.encode(), expected.encode())

        def _reject(self, code: HTTPStatus, detail: str) -> None:
            self._json(code, {"error": detail})

        def _proxy(self, method: str, path: str, *, body: bytes | None = None) -> None:
            url = urlsplit(self.path)
            try:
                # No forwarding Authorization, Cookie, X-Forwarded-* or
                # arbitrary client headers to unprotected localhost upstream.
                with proxy_client.stream(
                    method, path + ("?" + url.query if url.query else ""),
                    content=body, headers={"Content-Type": "application/json"} if body else {},
                ) as result:
                    if result.status_code >= 300:
                        self._reject(HTTPStatus.BAD_GATEWAY, "upstream_error")
                        return
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in result.iter_bytes():
                        size += len(chunk)
                        if size > _MAX_RESPONSE:
                            self._reject(HTTPStatus.BAD_GATEWAY, "upstream_response_too_large")
                            return
                        chunks.append(chunk)
                    mime = result.headers.get("content-type", "application/octet-stream")
                    self._body(HTTPStatus.OK, b"".join(chunks), mime)
            except (httpx.HTTPError, ValueError, OSError):
                self._reject(HTTPStatus.BAD_GATEWAY, "upstream_unavailable")

        def do_GET(self) -> None:
            if not self._authorized():
                self._reject(HTTPStatus.UNAUTHORIZED, "unauthorized")
                return
            path = urlsplit(self.path).path
            if path == "/v1/gateway/status":
                self._json(HTTPStatus.OK, {
                    "admission_sealed": gateway.blocked,
                    "upstream_loopback_configured": True,
                    "external_loopback_clients_fenced": False,
                    "restart_authorized": False,
                })
            elif path in _GET or _HISTORY.fullmatch(path):
                self._proxy("GET", path)
            else:
                self._reject(HTTPStatus.NOT_FOUND, "path_not_allowed")

        def do_POST(self) -> None:
            if not self._authorized():
                self._reject(HTTPStatus.UNAUTHORIZED, "unauthorized")
                return
            path = urlsplit(self.path).path
            if urlsplit(self.path).query or path not in _POST | {
                "/v1/gateway/seal", "/v1/gateway/release"
            }:
                self._reject(HTTPStatus.NOT_FOUND, "path_not_allowed")
                return
            length = self.headers.get("Content-Length")
            if length is None or not length.isdecimal() or int(length) > _MAX_POST:
                self._reject(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "invalid_content_length")
                return
            payload = self.rfile.read(int(length))
            if path == "/v1/gateway/seal":
                try:
                    sealed, reason = gateway.seal_with_queue(fetch_queue)
                except (httpx.HTTPError, ValueError, OSError, sqlite3.Error):
                    self._reject(HTTPStatus.SERVICE_UNAVAILABLE, "queue_or_fence_unavailable")
                    return
                self._json(HTTPStatus.OK if sealed else HTTPStatus.CONFLICT, {
                    "admission_sealed": sealed, "reason": reason,
                    "restart_authorized": False,
                })
                return
            if path == "/v1/gateway/release":
                try:
                    gateway.release()
                except (OSError, sqlite3.Error):
                    self._reject(HTTPStatus.SERVICE_UNAVAILABLE, "fence_unavailable")
                    return
                self._json(HTTPStatus.OK, {
                    "admission_sealed": False, "restart_authorized": False,
                })
                return
            try:
                with gateway.admit() as allowed:
                    if not allowed:
                        self._reject(HTTPStatus.LOCKED, "maintenance_sealed")
                        return
                    self._proxy("POST", path, body=payload)
            except (OSError, sqlite3.Error, sqlite3.OperationalError):
                self._reject(HTTPStatus.SERVICE_UNAVAILABLE, "fence_unavailable")

        def log_message(self, format: str, *args: object) -> None:
            return

    host = bind_host or cfg.bind_host
    bind_port = port if port is not None else cfg.port
    try:
        return ThreadingHTTPServer((host, bind_port), Handler)
    except BaseException:
        if own_client:
            proxy_client.close()
        raise


def serve_gateway(
    server: ThreadingHTTPServer, stop_event: Event,
) -> None:
    """Keep the server in the managed-renderer lifecycle, never detached."""
    server.timeout = 0.5
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        server.server_close()
