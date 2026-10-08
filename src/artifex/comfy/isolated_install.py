from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

_COMFY_ARCHIVE = "https://codeload.github.com/Comfy-Org/ComfyUI/zip/"
_MAX_ARCHIVE = 512 * 1024 * 1024
_MAX_UNPACKED = 2 * 1024 * 1024 * 1024
_MAX_FILES = 15000
_TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu130": "https://download.pytorch.org/whl/cu130",
}


class ComfyInstallResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    commit: str
    directory: Path
    comfy_directory: Path
    archive_sha256: str = Field(min_length=64, max_length=64)
    dependencies_installed: bool
    python_executable: Path | None
    torch_backend: str | None
    installed: bool


def _commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError("ComfyUI install requires an exact 40-character commit SHA")
    return value.lower()


def _hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as data:
        while part := data.read(1024 * 1024):
            sha.update(part)
    return sha.hexdigest()


def _unpack_source(zip_path: Path, project: Path) -> None:
    """Only extract regular files from exactly one GitHub source-archive root."""
    count = 0
    uncompressed = 0
    actual = 0
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.infolist()
        if len(entries) > _MAX_FILES:
            raise ValueError("ComfyUI archive contains too many entries")
        prefix: str | None = None
        for entry in entries:
            path = PurePosixPath(entry.filename.replace("\\", "/"))
            parts = path.parts
            if (
                path.is_absolute() or not parts or len(parts) > 40
                or ".." in parts or ":" in parts[0]
            ):
                raise ValueError(f"Unsafe ComfyUI archive path: {entry.filename}")
            if prefix is None:
                prefix = parts[0]
            if parts[0] != prefix:
                raise ValueError("ComfyUI archive has multiple top-level roots")
            if len(parts) == 1:
                if not entry.is_dir():
                    raise ValueError("ComfyUI archive root must be a folder")
                continue
            relative = PurePosixPath(*parts[1:])
            folded = str(relative).casefold()
            if folded in seen:
                raise ValueError("ComfyUI archive contains duplicate/case-colliding paths")
            seen.add(folded)
            kind = (entry.external_attr >> 16) & 0o170000
            if kind not in (0, 0o100000, 0o040000):
                raise ValueError(f"ComfyUI archive has unsupported file type: {entry.filename}")
            if kind == 0o040000 and not entry.is_dir():
                raise ValueError("ComfyUI archive has inconsistent directory metadata")
            count += 1
            uncompressed += entry.file_size
            if count > _MAX_FILES or uncompressed > _MAX_UNPACKED:
                raise ValueError("ComfyUI archive exceeds extraction limits")
            target = project.joinpath(*relative.parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    actual += len(chunk)
                    if actual > _MAX_UNPACKED:
                        raise ValueError("ComfyUI archive decompression exceeds size limit")
                    output.write(chunk)
    if not (project / "main.py").is_file() or not (project / "requirements.txt").is_file():
        raise ValueError("Archive is not a valid ComfyUI source tree")


def _run_uv(args: list[str], *, cwd: Path) -> None:
    """Use argv with no shell; install only into isolated staging directory."""
    result = subprocess.run(
        args, cwd=str(cwd), check=False, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=1800, shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if platform.system() == "Windows" else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Isolated ComfyUI dependency install failed (exit={result.returncode}): "
            f"{result.stderr[-800:]}"
        )


def install_isolated_comfy(
    commit: str,
    output_root: Path,
    *,
    install_dependencies: bool = False,
    torch_backend: str | None = None,
    python: Path | None = None,
    expected_archive_sha256: str | None = None,
    client: httpx.Client | None = None,
) -> ComfyInstallResult:
    """Install pinned upstream ComfyUI source into a new folder only.

    Source is pinned to a full immutable commit URL. The optional checksum
    protects against a mismatch with a *separately supplied* expected value.
    HTTPS download alone is not an independent upstream software signature.
    """
    commit = _commit(commit)
    if install_dependencies and torch_backend not in _TORCH_INDEX:
        raise ValueError("Installing dependencies requires --torch-backend cpu|cu126|cu128|cu130")
    if not install_dependencies and (torch_backend is not None or python is not None):
        raise ValueError("--torch-backend and --python require --install-deps")
    if expected_archive_sha256 is not None and re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_archive_sha256
    ) is None:
        raise ValueError("--archive-sha256 must be 64 hexadecimal characters")

    root = output_root.expanduser().resolve(strict=False)
    destination = root / f"comfyui-{commit[:12]}"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite an existing ComfyUI directory: {destination}"
        )
    owns = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(connect=30, read=120, write=30, pool=30),
        follow_redirects=True, trust_env=False,
    )
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".artifex-comfy-", dir=root) as temp:
            staging = Path(temp)
            downloaded = staging / "source.zip"
            digest = hashlib.sha256()
            size = 0
            with http.stream("GET", _COMFY_ARCHIVE + commit) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError("Pinned ComfyUI archive endpoint did not return HTTP 200")
                with downloaded.open("xb") as output:
                    for data in response.iter_bytes(chunk_size=1024 * 1024):
                        size += len(data)
                        if size > _MAX_ARCHIVE:
                            raise ValueError("ComfyUI source archive is larger than supported")
                        digest.update(data)
                        output.write(data)
            archive_sha = digest.hexdigest()
            if expected_archive_sha256 is not None and (
                archive_sha != expected_archive_sha256.lower()
            ):
                raise ValueError("Pinned ComfyUI source archive SHA-256 mismatch")
            extracted = staging / "prepared"
            project = extracted / "ComfyUI"
            project.mkdir(parents=True)
            _unpack_source(downloaded, project)
            python_executable: Path | None = None
            if install_dependencies:
                if platform.system() != "Windows":
                    raise RuntimeError("Managed dependency installation requires native Windows")
                uv_exe = shutil.which("uv")
                if uv_exe is None:
                    raise FileNotFoundError("uv is required on PATH to install dependencies")
                interpreter = (python or Path(sys.executable)).expanduser().resolve(
                    strict=True
                )
                if not interpreter.is_file():
                    raise FileNotFoundError(f"Python interpreter not found: {interpreter}")
                virtual = project / ".venv"
                assert torch_backend is not None
                _run_uv(
                    [uv_exe, "venv", "--python", str(interpreter), str(virtual)],
                    cwd=project,
                )
                python_executable = virtual / "Scripts" / "python.exe"
                if not python_executable.is_file():
                    raise RuntimeError("uv did not create the isolated ComfyUI Python")
                _run_uv(
                    [
                        uv_exe, "pip", "install", "--python", str(python_executable),
                        "--index-url", _TORCH_INDEX[torch_backend],
                        "torch", "torchvision", "torchaudio",
                    ],
                    cwd=project,
                )
                _run_uv(
                    [
                        uv_exe, "pip", "install", "--python", str(python_executable),
                        "-r", str(project / "requirements.txt"),
                    ],
                    cwd=project,
                )

            receipt: dict[str, Any] = {
                "source": "Comfy-Org/ComfyUI",
                "commit": commit,
                "archive_sha256": archive_sha,
                "main_py_sha256": _hash(project / "main.py"),
                "dependencies_installed": install_dependencies,
                "torch_backend": torch_backend,
            }
            (extracted / ".artifex-comfy-install.json").write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            if destination.exists() or destination.is_symlink():
                raise FileExistsError("Isolated ComfyUI destination appeared during setup")
            extracted.rename(destination)
            return ComfyInstallResult(
                commit=commit,
                directory=destination,
                comfy_directory=destination / "ComfyUI",
                archive_sha256=archive_sha,
                dependencies_installed=install_dependencies,
                python_executable=(
                    destination / "ComfyUI" / ".venv" / "Scripts" / "python.exe"
                    if install_dependencies else None
                ),
                torch_backend=torch_backend,
                installed=True,
            )
    finally:
        if owns:
            http.close()
