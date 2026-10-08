from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config.models import LlmConfig, LlmModelProfileConfig


class LlmBootstrapResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str
    path: Path
    downloaded: bool
    resumed_from_bytes: int
    bytes: int
    sha256: str
    source_url: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_url(profile: LlmModelProfileConfig) -> str | None:
    if profile.source == "local":
        return None
    if profile.source == "url":
        assert profile.url is not None
        return profile.url
    assert profile.repository is not None
    assert profile.filename is not None
    repository = "/".join(quote(part, safe="") for part in profile.repository.split("/"))
    revision = quote(profile.revision, safe="")
    filename = quote(profile.filename, safe="")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{filename}?download=true"


def _expected_download_size(
    response: httpx.Response,
    *,
    resume_from: int,
    append: bool,
) -> int | None:
    if append:
        content_range = response.headers.get("Content-Range")
        if content_range is None:
            raise ValueError("resumed LLM download is missing Content-Range")
        try:
            unit, value = content_range.split(" ", 1)
            byte_range, total_text = value.split("/", 1)
            start_text, end_text = byte_range.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            total = int(total_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid LLM download Content-Range: {content_range}"
            ) from exc
        if unit.casefold() != "bytes" or start != resume_from or end < start:
            raise ValueError(
                f"unexpected LLM download Content-Range: {content_range}"
            )
        if end >= total:
            raise ValueError(
                f"invalid LLM download Content-Range total: {content_range}"
            )
        return total

    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    try:
        size = int(content_length)
    except ValueError as exc:
        raise ValueError(
            f"invalid LLM download Content-Length: {content_length}"
        ) from exc
    if size < 0:
        raise ValueError(f"invalid LLM download Content-Length: {content_length}")
    return size


def _integrity_receipt(target: Path) -> Path:
    return target.with_name(target.name + ".integrity.json")


def _expected_sha256(
    profile: LlmModelProfileConfig,
    source_url: str,
    target: Path,
    http: httpx.Client,
) -> str:
    if profile.sha256 is not None:
        return profile.sha256.casefold()

    receipt = _integrity_receipt(target)
    if receipt.is_file():
        try:
            record = json.loads(receipt.read_text(encoding="utf-8"))
            sha = record["sha256"]
            if (
                record.get("source_url") == source_url
                and isinstance(sha, str)
                and re.fullmatch(r"[0-9a-fA-F]{64}", sha)
            ):
                return str(sha).casefold()
        except (OSError, ValueError, KeyError, TypeError):
            pass

    if profile.source != "huggingface":
        raise ValueError(
            "download profiles using source=url must configure sha256; "
            "unverified remote models cannot be adopted"
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40}", profile.revision):
        raise ValueError(
            "Hugging Face profiles without sha256 must use an immutable "
            "40-character revision commit"
        )
    # Xet/LFS metadata lives on Hugging Face's resolve redirect. Following
    # the external CDN redirect with HEAD can fail or lose X-Linked-Etag.
    metadata_url = source_url
    for _ in range(5):
        response = http.head(metadata_url, follow_redirects=False)
        response.raise_for_status()
        for header_name in ("x-linked-etag", "etag"):
            raw = response.headers.get(header_name, "").strip('"')
            if re.fullmatch(r"[0-9a-fA-F]{64}", raw):
                return str(raw).casefold()
        if not 300 <= response.status_code < 400:
            break
        location = response.headers.get("Location")
        if not location:
            break
        next_url = urljoin(metadata_url, location)
        if urlsplit(next_url).hostname != urlsplit(source_url).hostname:
            break
        metadata_url = next_url
    raise ValueError("Hugging Face did not supply a verifiable model SHA-256")


def _write_integrity_receipt(target: Path, *, source_url: str, sha256: str) -> None:
    receipt = _integrity_receipt(target)
    temporary = receipt.with_name(receipt.name + ".tmp")
    temporary.write_text(
        json.dumps({"source_url": source_url, "sha256": sha256}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt)


def bootstrap_llm(
    config: LlmConfig,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> LlmBootstrapResult:
    bootstrap = config.bootstrap
    if not bootstrap.enabled:
        raise ValueError("LLM bootstrap is disabled")

    profile = bootstrap.selected()
    target = bootstrap.model_path().expanduser()
    if profile.source == "local":
        target = target.resolve(strict=True)
        digest = _sha256(target)
        if profile.sha256 is not None and digest.casefold() != profile.sha256.casefold():
            raise ValueError("local LLM model SHA-256 does not match configured profile")
        return LlmBootstrapResult(
            profile=bootstrap.profile,
            path=target,
            downloaded=False,
            resumed_from_bytes=0,
            bytes=target.stat().st_size,
            sha256=digest,
            source_url=None,
        )

    target = target.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_url = _source_url(profile)
    assert source_url is not None
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=timeout)
    try:
        expected_sha = _expected_sha256(profile, source_url, target, http)
        if target.is_file() and not force:
            digest = _sha256(target)
            if digest.casefold() != expected_sha:
                raise ValueError(
                    "existing LLM model SHA-256 does not match the pinned source; "
                    "rerun with --force"
                )
            _write_integrity_receipt(
                target, source_url=source_url, sha256=expected_sha
            )
            return LlmBootstrapResult(
                profile=bootstrap.profile,
                path=target,
                downloaded=False,
                resumed_from_bytes=0,
                bytes=target.stat().st_size,
                sha256=digest,
                source_url=source_url,
            )

        partial = target.with_name(target.name + ".part")
        if force:
            # Do not delete a valid installed model until replacement is verified.
            partial.unlink(missing_ok=True)

        resume_from = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        with http.stream("GET", source_url, headers=headers) as response:
            response.raise_for_status()
            append = resume_from > 0 and response.status_code == 206
            if resume_from > 0 and not append:
                resume_from = 0
            expected_size = _expected_download_size(
                response,
                resume_from=resume_from,
                append=append,
            )
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if expected_size is not None and partial.stat().st_size != expected_size:
            raise ValueError(
                "LLM download ended before the expected file size was reached: "
                f"{partial.stat().st_size} != {expected_size}"
            )

        digest = _sha256(partial)
        if digest.casefold() != expected_sha:
            partial.unlink(missing_ok=True)
            raise ValueError("downloaded LLM model SHA-256 does not match pinned source")
        partial.replace(target)
        _write_integrity_receipt(
            target, source_url=source_url, sha256=expected_sha
        )
        return LlmBootstrapResult(
            profile=bootstrap.profile,
            path=target,
            downloaded=True,
            resumed_from_bytes=resume_from,
            bytes=target.stat().st_size,
            sha256=digest,
            source_url=source_url,
        )
    finally:
        if owns_client:
            http.close()
