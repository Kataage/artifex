from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

_HEX = re.compile(r"[0-9a-fA-F]{64}")
_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_COMPONENT = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}")
_MAX_DOWNLOAD = 32 * 1024**3
_MAX_EXTRACTED = 2 * 1024**3
_MAX_FILES = 10000
_MODEL_FOLDERS = frozenset(
    {"checkpoints", "vae", "upscale_models", "loras", "ultralytics", "controlnet"}
)


class PinnedDependency(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: Literal["model", "custom_node"]
    url: str
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(gt=0, le=_MAX_DOWNLOAD)
    license_id: str = Field(min_length=2, max_length=100)
    license_url: str
    name: str
    model_folder: str | None = None
    repository: str | None = None
    commit: str | None = None

    @model_validator(mode="after")
    def validate_pinned(self) -> PinnedDependency:
        if not _COMPONENT.fullmatch(self.id):
            raise ValueError("Dependency id must be a safe filename component")
        if not _HEX.fullmatch(self.sha256):
            raise ValueError("Dependency SHA256 must be 64 hexadecimal digits")
        if not _COMPONENT.fullmatch(self.name):
            raise ValueError("Dependency name must be a basename without a path")
        license_url = urlsplit(self.license_url)
        if license_url.scheme != "https" or not license_url.netloc:
            raise ValueError("Dependency license_url must be HTTPS")
        parsed = urlsplit(self.url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("Dependencies must come from an HTTPS source")
        if self.kind == "custom_node":
            if (
                self.model_folder is not None or self.repository is None
                or self.commit is None or not _COMMIT.fullmatch(self.commit)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository)
                or self.url != (
                    f"https://codeload.github.com/{self.repository}/zip/{self.commit}"
                )
            ):
                raise ValueError(
                    "Custom node must be a full pinned GitHub repository commit ZIP"
                )
        else:
            if self.repository is not None or self.commit is not None:
                raise ValueError("Model dependencies cannot set repository/commit")
            if self.model_folder not in _MODEL_FOLDERS:
                raise ValueError("Model folder must be an approved ComfyUI model category")
            if not (
                parsed.hostname == "huggingface.co"
                and re.fullmatch(
                    r"/[^/]+/[^/]+/resolve/[0-9a-fA-F]{40}/[^/]+", parsed.path
                )
                or parsed.hostname == "github.com"
                and re.fullmatch(
                    r"/[^/]+/[^/]+/releases/download/[^/]+/[^/]+", parsed.path
                )
            ):
                raise ValueError(
                    "Model source must be a pinned Hugging Face revision or GitHub release"
                )
            if parsed.path.rsplit("/", 1)[-1] != self.name:
                raise ValueError("Source model filename must exactly match target filename")
        return self


class DependencyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifacts: tuple[PinnedDependency, ...] = ()

    @model_validator(mode="after")
    def unique_destinations(self) -> DependencyManifest:
        ids = [item.id.casefold() for item in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("Manifest dependency ids must be unique")
        paths = [
            f"{item.kind}/{item.model_folder or 'custom_nodes'}/{item.name}".casefold()
            for item in self.artifacts
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("Manifest destinations must be unique")
        return self


class InstalledArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    destination: Path
    sha256: str
    kind: Literal["model", "custom_node"]


class DependencyInstallResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    comfyui_directory: Path
    installed: tuple[InstalledArtifact, ...]
    restart_required: bool


def read_manifest(path: Path) -> DependencyManifest:
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("Dependency manifest exceeds 1 MiB")
    try:
        return DependencyManifest.model_validate_json(raw)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid pinned dependency manifest: {exc}") from exc


def _safe_install_root(root: Path) -> Path:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir() or not (root / "main.py").is_file():
        raise ValueError("ComfyUI directory must be a genuine source tree")
    receipt = root.parent / ".artifex-comfy-install.json"
    if not receipt.is_file():
        raise ValueError(
            "Third-party artifacts may only be installed in an isolated "
            "ComfyUI directory created by Artifex"
        )
    data: Any = json.loads(receipt.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("source") != "Comfy-Org/ComfyUI":
        raise ValueError("Unrecognized Artifex isolated ComfyUI receipt")
    return root


def _guard_destination(root: Path, item: PinnedDependency) -> Path:
    base = root / ("custom_nodes" if item.kind == "custom_node" else "models")
    if item.kind == "model":
        assert item.model_folder is not None
        base = base / item.model_folder
    for path in (base.parent, base):
        if path.is_symlink():
            raise ValueError(f"Refusing symlinked ComfyUI dependency directory: {path}")
    target = base / item.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing dependency: {target}")
    return target


def _unpack_node(source: Path, target: Path) -> None:
    count = 0
    size = 0
    seen: set[str] = set()
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len(infos) > _MAX_FILES:
            raise ValueError("Custom-node archive contains too many files")
        archive_root: str | None = None
        for info in infos:
            original = info.filename.replace("\\", "/")
            rel = PurePosixPath(original)
            if (
                not original or original.startswith("/") or not rel.parts
                or ".." in rel.parts or any(":" in part for part in rel.parts)
            ):
                raise ValueError("Unsafe custom-node archive path")
            if archive_root is None:
                archive_root = rel.parts[0]
            if rel.parts[0] != archive_root:
                raise ValueError("Custom-node archive has multiple root directories")
            if len(rel.parts) == 1:
                if not info.is_dir():
                    raise ValueError("Custom-node archive root is not a directory")
                continue
            relative = PurePosixPath(*rel.parts[1:])
            folded = str(relative).casefold()
            if folded in seen:
                raise ValueError("Duplicate or case-colliding custom-node archive entry")
            seen.add(folded)
            mode = (info.external_attr >> 16) & 0o170000
            if mode not in (0, 0o100000, 0o040000):
                raise ValueError("Custom-node archive contains a symlink/special file")
            if mode == 0o040000 and not info.is_dir():
                raise ValueError("Custom-node archive directory metadata is invalid")
            count += 1
            size += info.file_size
            if count > _MAX_FILES or size > _MAX_EXTRACTED:
                raise ValueError("Custom-node ZIP exceeds extraction limits")
            dest = target.joinpath(*relative.parts)
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, dest.open("xb") as output:
                actual = 0
                while chunk := src.read(1024 * 1024):
                    actual += len(chunk)
                    if actual > info.file_size or actual > _MAX_EXTRACTED:
                        raise ValueError("Custom-node ZIP decompression exceeds declared size")
                    output.write(chunk)
                if actual != info.file_size:
                    raise ValueError("Custom-node ZIP member is truncated")
    if not any((target / name).is_file() for name in ("__init__.py", "pyproject.toml")):
        raise ValueError("Custom-node package lacks __init__.py or pyproject.toml")


def _download(
    item: PinnedDependency, staging: Path, client: httpx.Client
) -> Path:
    target = staging / (item.id + ".download")
    digest = hashlib.sha256()
    length = 0
    with client.stream("GET", item.url) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Expected complete HTTP 200 for dependency download")
        with target.open("xb") as out:
            for data in response.iter_bytes(chunk_size=1024 * 1024):
                length += len(data)
                if length > item.size_bytes:
                    raise ValueError(f"Dependency download exceeded expected size: {item.id}")
                digest.update(data)
                out.write(data)
    if length != item.size_bytes or digest.hexdigest() != item.sha256.lower():
        raise ValueError(f"Dependency size/SHA256 check failed: {item.id}")
    return target


def install_manifest(
    manifest: DependencyManifest, comfy_root: Path, *,
    accept_licenses: bool = False,
    allow_custom_code: bool = False,
    client: httpx.Client | None = None,
) -> DependencyInstallResult:
    """Apply only operator-accepted, hashed artifacts to Artifex-owned ComfyUI.

    Never pip-install dependencies or execute install.py for a custom node.
    Custom nodes become runnable only on the next separate ComfyUI restart.
    """
    root = _safe_install_root(comfy_root)
    if manifest.artifacts and not accept_licenses:
        raise ValueError("Explicit --accept-licenses is required for third-party artifacts")
    if any(i.kind == "custom_node" for i in manifest.artifacts) and not allow_custom_code:
        raise ValueError("Explicit --allow-custom-code is required for custom nodes")
    destinations = [_guard_destination(root, item) for item in manifest.artifacts]
    own = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
        follow_redirects=True, trust_env=False,
    )
    try:
        with tempfile.TemporaryDirectory(prefix=".artifex-deps-", dir=root.parent) as temp:
            staging = Path(temp)
            prepared: list[Path] = []
            for index, item in enumerate(manifest.artifacts):
                downloaded = _download(item, staging, http)
                if item.kind == "custom_node":
                    folder = staging / f"extracted-{index}"
                    folder.mkdir()
                    _unpack_node(downloaded, folder)
                    prepared.append(folder)
                else:
                    prepared.append(downloaded)
            installed: list[InstalledArtifact] = []
            for item, src, target in zip(
                manifest.artifacts, prepared, destinations, strict=True
            ):
                # An operator or another process could have changed this path
                # during download. Re-check before every final atomic move.
                current = _guard_destination(root, item)
                if current != target:
                    raise RuntimeError("Dependency target unexpectedly changed")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    raise FileExistsError("Dependency destination appeared during install")
                src.rename(target)
                installed.append(
                    InstalledArtifact(
                        id=item.id, destination=target,
                        sha256=item.sha256.lower(), kind=item.kind,
                    )
                )
            return DependencyInstallResult(
                comfyui_directory=root,
                installed=tuple(installed),
                restart_required=any(i.kind == "custom_node" for i in manifest.artifacts),
            )
    finally:
        if own:
            http.close()
