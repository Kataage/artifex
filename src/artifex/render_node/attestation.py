from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any
from urllib.parse import parse_qs, urlsplit

from artifex.config.models import ArtifexSettings
from artifex.loras.safetensors import (
    SafeTensorMetadataError,
    read_safetensors_metadata,
    sha256_file,
)
from artifex.render_node.models import (
    RenderAssetDigest,
    RenderLoRAInventoryItem,
    RenderNodeAttestation,
)
from artifex.render_node.remote_owner_readiness import (
    OwnerReadinessCache,
    can_collect_owner_readiness,
)
from artifex.render_node.survival_spool import latest_survival_trace


def _hash_file(path: Path, digest: Any) -> tuple[int, int]:
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, 1


def digest_path(label: str, path: Path) -> RenderAssetDigest:
    absolute = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    if absolute.is_file():
        total_bytes, file_count = _hash_file(absolute, digest)
    elif absolute.is_dir():
        files = sorted(
            (item for item in absolute.rglob("*") if item.is_file()),
            key=lambda item: item.as_posix().casefold(),
        )
        if not files:
            raise ValueError(f"asset directory is empty: {absolute}")
        for item in files:
            relative = item.relative_to(absolute).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            size, count = _hash_file(item, digest)
            total_bytes += size
            file_count += count
    else:
        raise ValueError(f"asset path is neither file nor directory: {absolute}")
    return RenderAssetDigest(
        label=label,
        path=str(absolute),
        sha256=digest.hexdigest(),
        bytes=total_bytes,
        file_count=file_count,
    )


def _nvidia_gpus() -> tuple[dict[str, object], ...]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    result: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            memory_mib: object = int(parts[3])
        except ValueError:
            memory_mib = parts[3]
        result.append(
            {
                "name": parts[0],
                "uuid": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": memory_mib,
            }
        )
    return tuple(result)


def _lora_inventory(
    settings: ArtifexSettings,
) -> tuple[tuple[RenderLoRAInventoryItem, ...], tuple[dict[str, str], ...]]:
    config = settings.render_agent
    extensions = {extension.casefold() for extension in config.extensions}
    items: list[RenderLoRAInventoryItem] = []
    errors: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for root in config.lora_roots:
        base = root.expanduser().resolve(strict=False)
        if not base.exists():
            errors.append({"path": str(base), "error": "LoRA root does not exist"})
            continue
        paths = sorted(
            (
                item.resolve(strict=False)
                for item in base.rglob("*")
                if item.is_file() and item.suffix.casefold() in extensions
            ),
            key=lambda item: item.as_posix().casefold(),
        )
        for path in paths:
            try:
                # Resolved symlinks may leave the configured directory.
                relative = path.relative_to(base).as_posix()
                if relative.casefold() in seen_names:
                    errors.append(
                        {
                            "path": str(path),
                            "error": "duplicate renderer-relative LoRA asset name",
                        }
                    )
                    continue
                metadata = read_safetensors_metadata(
                    path,
                    max_header_bytes=config.metadata_header_max_mib * 1024 * 1024,
                )
                checksum = sha256_file(path)
            except (OSError, SafeTensorMetadataError, ValueError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            seen_names.add(relative.casefold())
            items.append(
                RenderLoRAInventoryItem(
                    name=path.name,
                    path=str(path),
                    relative_path=relative,
                    sha256=checksum,
                    bytes=path.stat().st_size,
                    metadata=metadata,
                )
            )
    items.sort(key=lambda item: (item.relative_path.casefold(), item.sha256))
    return tuple(items), tuple(errors)


def build_attestation(settings: ArtifexSettings) -> RenderNodeAttestation:
    assets: list[RenderAssetDigest] = []
    errors: list[dict[str, str]] = []
    for label, path in sorted(settings.render_agent.asset_paths.items()):
        try:
            assets.append(digest_path(label, path))
        except (OSError, ValueError) as exc:
            errors.append({"path": str(path), "error": f"{label}: {exc}"})
    loras, lora_errors = _lora_inventory(settings)
    errors.extend(lora_errors)
    return RenderNodeAttestation(
        node_id=settings.render_agent.node_id,
        created_at=datetime.now(UTC),
        hostname=socket.gethostname(),
        os={
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        nvidia_gpus=_nvidia_gpus(),
        comfyui_base_url=settings.comfyui.base_url,
        assets=tuple(assets),
        loras=loras,
        inventory_errors=tuple(errors),
    )


def _token(settings: ArtifexSettings) -> str | None:
    name = settings.render_agent.token_env
    if name is None:
        return None
    return os.environ.get(name)


def serve_attestation(
    settings: ArtifexSettings, *,
    stop_event: Event | None = None,
    owner_config: Path | None = None,
) -> None:
    expected_token = _token(settings)
    if settings.render_agent.require_token and not expected_token:
        raise ValueError(
            "render-node token is required but the configured environment variable is empty"
        )

    # Background-only isolated launcher fixture. Protected GET requests only
    # read a memory snapshot, never launch Python or submit GPU jobs.
    owner_cache = OwnerReadinessCache(settings, owner_config=owner_config)
    worker_stop = Event()
    worker: Thread | None = None

    cache_lock = Lock()
    cached: tuple[float, RenderNodeAttestation] | None = None

    def snapshot(*, fresh: bool) -> RenderNodeAttestation:
        nonlocal cached
        with cache_lock:
            now = time.monotonic()
            if (
                not fresh
                and cached is not None
                and now - cached[0] < settings.render_agent.attestation_cache_seconds
            ):
                return cached[1]
            evidence = build_attestation(settings)
            cached = (time.monotonic(), evidence)
            return evidence

    class Handler(BaseHTTPRequestHandler):
        server_version = "ArtifexRenderNode/1"

        def _authorized(self) -> bool:
            if expected_token is None:
                return not settings.render_agent.require_token
            return self.headers.get("Authorization") == f"Bearer {expected_token}"

        def _json(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path == "/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "node_id": settings.render_agent.node_id,
                    },
                )
                return
            protected_path = url.path in {
                "/v1/owner-audit", "/v1/renderer-safety",
                "/v1/owner-readiness-evidence", "/v1/survival-evidence",
            }
            if url.path not in {
                "/v1/attestation", "/v1/owner-audit", "/v1/renderer-safety",
                "/v1/owner-readiness-evidence", "/v1/survival-evidence",
            }:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if (
                not self._authorized()
                or (protected_path and expected_token is None)
            ):
                # /health and legacy attestation can be configured public,
                # but owner and safety evidence always require a Bearer token.
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if url.path == "/v1/survival-evidence":
                if url.query:
                    self._json(
                        HTTPStatus.BAD_REQUEST, {"error": "query_not_supported"},
                    )
                    return
                # Only a configured directory's most recent immutable report.
                # GET never starts a Python child, Task Scheduler job, or GPU task.
                try:
                    sha, content = latest_survival_trace(settings)
                except (OSError, ValueError, UnicodeError):
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "survival_evidence_unavailable"},
                    )
                    return
                self._json(HTTPStatus.OK, {
                    "node_id": settings.render_agent.node_id,
                    "sha256": sha, "content": content,
                })
                return
            if url.path == "/v1/owner-readiness-evidence":
                if url.query or not can_collect_owner_readiness(settings, owner_config):
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "owner_readiness_unconfigured"},
                    )
                    return
                # Read-only cache access. No Python subprocess is created
                # by GET, even under repeated authenticated requests.
                report = owner_cache.snapshot()
                if report is None:
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "owner_readiness_not_sampled_or_stale"},
                    )
                    return
                self._json(HTTPStatus.OK, {
                    "node_id": settings.render_agent.node_id,
                    "report": report,
                })
                return
            if url.path == "/v1/renderer-safety":
                if (
                    url.query or owner_config is None
                    or owner_config.is_symlink() or not owner_config.is_file()
                ):
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "renderer_safety_unconfigured"},
                    )
                    return
                # An uncached Windows PID/TCP/Scheduler observation only.
                # This endpoint cannot authorize startup or modify services.
                try:
                    from artifex.render_node.startup_inspection import (
                        inspect_renderer_startup,
                    )

                    inspection = inspect_renderer_startup(
                        settings, config_path=owner_config,
                    )
                    self._json(HTTPStatus.OK, {
                        "node_id": settings.render_agent.node_id,
                        "inspection": inspection.model_dump(mode="json"),
                    })
                except Exception:  # noqa: BLE001 - no host paths or tokens in errors
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "renderer_safety_failed"},
                    )
                return
            if url.path == "/v1/owner-audit":
                if url.query or owner_config is None or owner_config.is_symlink() or not owner_config.is_file():
                    self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "owner_audit_unconfigured"})
                    return
                # No caching: this is a current, read-only PID/TCP/Scheduler snapshot.
                # Never expose command lines, tokens or filesystem content.
                try:
                    from artifex.render_node.owner_audit import observe_renderer_owner

                    audit = observe_renderer_owner(settings, config=owner_config)
                    self._json(HTTPStatus.OK, {
                        "node_id": settings.render_agent.node_id,
                        "audit": audit,
                    })
                except Exception:  # noqa: BLE001 - fail closed and avoid leaking host paths
                    self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "owner_audit_failed"})
                return
            try:
                fresh = parse_qs(url.query).get("fresh") == ["1"]
                payload: Any = snapshot(fresh=fresh).model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": type(exc).__name__, "detail": str(exc)[:2000]},
                )
                return
            self._json(HTTPStatus.OK, payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(
        (settings.render_agent.bind_host, settings.render_agent.port),
        Handler,
    )
    try:
        if can_collect_owner_readiness(settings, owner_config):
            worker = Thread(
                target=owner_cache.run, args=(worker_stop,),
                name="artifex-local-owner-evidence", daemon=True,
            )
            worker.start()
        if stop_event is None:
            server.serve_forever()
        else:
            server.timeout = 0.5
            while not stop_event.is_set():
                server.handle_request()
    finally:
        worker_stop.set()
        if worker is not None:
            worker.join(timeout=2)
        server.server_close()
