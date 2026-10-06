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


def bootstrap_llm(config: LlmConfig, *, force: bool = False) -> LlmBootstrapResult:
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
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", source_url, headers=headers) as response:
            response.raise_for_status()
            append = resume_from > 0 and response.status_code == 206
            if resume_from > 0 and not append:
                resume_from = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

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
