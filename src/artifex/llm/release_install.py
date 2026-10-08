from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/"
_OWNER_PREFIX = "https://github.com/ggml-org/llama.cpp/releases/download/"
_MAX_ARCHIVE_BYTES = 2 * 1024**3
_MAX_CONTENT_BYTES = 5 * 1024**3
_MAX_MEMBERS = 2048


class OfficialLlamaAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tag: str
    name: str
    bytes: int
    sha256: str = Field(min_length=64, max_length=64)
    url: str


class LlamaInstallResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tag: str
    asset: str
    archive_sha256: str
    directory: Path
    executable: Path
    executable_sha256: str
    installed: bool


def _digest_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def _assert_tag(tag: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", tag) is None:
        raise ValueError("Release tag must be a simple pinned tag (e.g. b12345)")


def _assert_asset(name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,240}\.zip", name) is None:
        raise ValueError("Release asset must be an exact .zip filename without paths")


def official_release_assets(
    tag: str, *, client: httpx.Client | None = None
) -> tuple[OfficialLlamaAsset, ...]:
    """List Windows x64 official release ZIPs with GitHub-provided SHA-256.

    The checksum is published by GitHub for that release asset; it is not
    an independent signature or guarantee of upstream supply-chain security.
    """
    _assert_tag(tag)
    owns = client is None
    http = client or httpx.Client(timeout=20, follow_redirects=True, trust_env=False)
    try:
        response = http.get(
            API + quote(tag, safe=""),
            headers={"Accept": "application/vnd.github+json"},
        )
        response.raise_for_status()
        data: Any = response.json()
        if not isinstance(data, dict) or data.get("tag_name") != tag:
            raise ValueError("GitHub release tag does not match requested tag")
        listed = data.get("assets")
        if not isinstance(listed, list):
            raise TypeError("GitHub release has no assets list")
        found: list[OfficialLlamaAsset] = []
        for entry in listed:
            if not isinstance(entry, dict):
                continue
            name, digest = entry.get("name"), entry.get("digest")
            if not isinstance(name, str) or not isinstance(digest, str):
                continue
            if not (name.endswith(".zip") and "win" in name.casefold()
                    and "x64" in name.casefold()):
                continue
            if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest) is None:
                continue
            try:
                _assert_asset(name)
            except ValueError:
                continue
            url = entry.get("browser_download_url")
            if not isinstance(url, str) or not url.startswith(
                _OWNER_PREFIX + quote(tag, safe="") + "/"
            ):
                continue
            if urlsplit(url).path.rsplit("/", 1)[-1] != name:
                continue
            size = entry.get("size")
            if not isinstance(size, int) or not 0 < size <= _MAX_ARCHIVE_BYTES:
                continue
            found.append(
                OfficialLlamaAsset(
                    tag=tag,
                    name=name,
                    bytes=size,
                    sha256=digest[7:].lower(),
                    url=url,
                )
            )
        return tuple(sorted(found, key=lambda item: item.name))
    finally:
        if owns:
            http.close()


def _safe_member(name: str, info: zipfile.ZipInfo) -> PurePosixPath:
    raw = name.replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        not raw or raw.startswith("/") or not path.parts or ".." in path.parts
        or ":" in path.parts[0]
    ):
        raise ValueError(f"Unsafe archive member path: {name}")
    mode = (info.external_attr >> 16) & 0o170000
    if mode == 0o120000:
        raise ValueError(f"Archive contains a symlink: {name}")
    return path


def extract_verified_zip(archive: Path, destination: Path) -> Path:
    total = 0
    binaries: list[Path] = []
    members_seen: set[str] = set()
    with zipfile.ZipFile(archive, "r") as packaged:
        infos = packaged.infolist()
        if len(infos) > _MAX_MEMBERS:
            raise ValueError("Archive contains too many members")
        for info in infos:
            relative = _safe_member(info.filename, info)
            folded = str(relative).casefold()
            if folded in members_seen:
                raise ValueError("Archive has duplicate or case-colliding members")
            members_seen.add(folded)
            if info.is_dir():
                continue
            total += info.file_size
            if total > _MAX_CONTENT_BYTES:
                raise ValueError("Archive exceeds safe uncompressed size")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with packaged.open(info) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            if target.name.casefold() == "llama-server.exe":
                binaries.append(target)
    if len(binaries) != 1:
        raise ValueError("Official llama.cpp ZIP must contain one llama-server.exe")
    return binaries[0]


def install_official_llama(
    tag: str,
    asset_name: str,
    output_root: Path,
    *,
    client: httpx.Client | None = None,
) -> LlamaInstallResult:
    """Explicit pinned, checksum-validated native install; no shell, no overwrite."""
    _assert_tag(tag)
    _assert_asset(asset_name)
    own = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(connect=30, read=120, write=30, pool=30),
        follow_redirects=True,
        trust_env=False,
    )
    try:
        candidates = official_release_assets(tag, client=http)
        matches = [asset for asset in candidates if asset.name == asset_name]
        if len(matches) != 1:
            raise ValueError(
                f"Asset {asset_name} lacks official SHA-256 metadata for tag {tag}; "
                "inspect 'onboard llama-assets' for verified choices"
            )
        asset = matches[0]
        root = output_root.expanduser().resolve(strict=False)
        target = root / f"llama-cpp-{tag}-{asset.name[:-4]}"
        receipt = target / ".artifex-install.json"
        if target.exists():
            if not target.is_dir() or not receipt.is_file():
                raise FileExistsError(f"Refusing to overwrite unrecognized install: {target}")
            try:
                data = json.loads(receipt.read_text(encoding="utf-8"))
                relative = PurePosixPath(data["relative_executable"])
                binary = target.joinpath(*relative.parts)
                expected = data["executable_sha256"]
                if (
                    data["tag"] != tag or data["asset"] != asset.name
                    or data["archive_sha256"] != asset.sha256
                    or _safe_receipt_binary_path(relative) is False
                    or not binary.is_file()
                    or _digest_file(binary) != expected
                ):
                    raise ValueError("Existing llama.cpp install does not match receipt")
                return LlamaInstallResult(
                    tag=tag, asset=asset.name, archive_sha256=asset.sha256,
                    directory=target, executable=binary, executable_sha256=expected,
                    installed=False,
                )
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise FileExistsError(
                    f"Existing installation cannot be trusted: {target}"
                ) from exc
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".artifex-llama-", dir=root) as tmpdir:
            staging = Path(tmpdir)
            archive = staging / "download.zip"
            extracted = staging / "extracted"
            extracted.mkdir()
            digest = hashlib.sha256()
            count = 0
            with http.stream("GET", asset.url) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError("Official release download did not return HTTP 200")
                with archive.open("xb") as output:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        count += len(chunk)
                        if count > _MAX_ARCHIVE_BYTES:
                            raise ValueError("Official release archive exceeds size limit")
                        digest.update(chunk)
                        output.write(chunk)
            if count != asset.bytes or digest.hexdigest() != asset.sha256:
                raise ValueError("Downloaded llama.cpp ZIP failed official SHA-256/size check")
            executable = extract_verified_zip(archive, extracted)
            relative_binary = executable.relative_to(extracted)
            binary_sha = _digest_file(executable)
            (extracted / ".artifex-install.json").write_text(
                json.dumps(
                    {
                        "tag": tag,
                        "asset": asset.name,
                        "archive_sha256": asset.sha256,
                        "relative_executable": relative_binary.as_posix(),
                        "executable_sha256": binary_sha,
                    },
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            if target.exists():
                raise FileExistsError(f"Install target was created concurrently: {target}")
            extracted.rename(target)
            return LlamaInstallResult(
                tag=tag, asset=asset.name, archive_sha256=asset.sha256,
                directory=target, executable=target / relative_binary,
                executable_sha256=binary_sha, installed=True,
            )
    finally:
        if own:
            http.close()


def _safe_receipt_binary_path(path: PurePosixPath) -> bool:
    return bool(
        path.parts and not path.is_absolute() and ".." not in path.parts
        and ":" not in path.parts[0]
        and path.name.casefold() == "llama-server.exe"
    )
