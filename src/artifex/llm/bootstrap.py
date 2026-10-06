from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote

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
    if target.is_file() and not force:
        digest = _sha256(target)
        if profile.sha256 is None or digest.casefold() == profile.sha256.casefold():
            return LlmBootstrapResult(
                profile=bootstrap.profile,
                path=target,
                downloaded=False,
                resumed_from_bytes=0,
                bytes=target.stat().st_size,
                sha256=digest,
                source_url=_source_url(profile),
            )
        raise ValueError(
            "existing LLM model SHA-256 does not match profile; rerun with --force"
        )

    source_url = _source_url(profile)
    assert source_url is not None
    partial = target.with_name(target.name + ".part")
    if force:
        partial.unlink(missing_ok=True)
        target.unlink(missing_ok=True)

    resume_from = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=timeout)
    try:
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
    finally:
        if owns_client:
            http.close()

    digest = _sha256(partial)
    if profile.sha256 is not None and digest.casefold() != profile.sha256.casefold():
        partial.unlink(missing_ok=True)
        raise ValueError("downloaded LLM model SHA-256 does not match configured profile")
    partial.replace(target)
    return LlmBootstrapResult(
        profile=bootstrap.profile,
        path=target,
        downloaded=True,
        resumed_from_bytes=resume_from,
        bytes=target.stat().st_size,
        sha256=digest,
        source_url=source_url,
    )
