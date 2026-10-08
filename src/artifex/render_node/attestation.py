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
from threading import Event, Lock
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


def serve_attestation(settings: ArtifexSettings, *, stop_event: Event | None = None) -> None:
    expected_token = _token(settings)
    if settings.render_agent.require_token and not expected_token:
        raise ValueError(
            "render-node token is required but the configured environment variable is empty"
        )

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
            if url.path != "/v1/attestation":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
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
        if stop_event is None:
            server.serve_forever()
        else:
            server.timeout = 0.5
            while not stop_event.is_set():
                server.handle_request()
    finally:
        server.server_close()
