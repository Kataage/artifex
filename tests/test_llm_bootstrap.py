from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from artifex.config.models import (
    LlmBootstrapConfig,
    LlmConfig,
    LlmModelProfileConfig,
)
from artifex.llm import bootstrap_llm


def test_llm_bootstrap_resumes_partial_download_and_reuses_verified_file(
    tmp_path: Path,
) -> None:
    payload = b"abcdefgh"
    digest = hashlib.sha256(payload).hexdigest()
    config = LlmConfig(
        bootstrap=LlmBootstrapConfig(
            models_dir=tmp_path,
            profile="test",
            profiles={
                "test": LlmModelProfileConfig(
                    source="url",
                    url="https://models.test/model.gguf",
                    filename="model.gguf",
                    sha256=digest,
                )
            },
        )
    )
    partial = tmp_path / "model.gguf.part"
    partial.write_bytes(payload[:4])
    calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("Range"))
        assert request.url == "https://models.test/model.gguf"
        assert request.headers.get("Range") == "bytes=4-"
        return httpx.Response(
            206,
            content=payload[4:],
            headers={"Content-Range": "bytes 4-7/8"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = bootstrap_llm(config, client=client)
    client.close()

    assert result.downloaded is True
    assert result.resumed_from_bytes == 4
    assert result.sha256 == digest
    assert result.path.read_bytes() == payload
    assert calls == ["bytes=4-"]

    def unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"network should not be used: {request.url}")

    reuse_client = httpx.Client(transport=httpx.MockTransport(unexpected))
    reused = bootstrap_llm(config, client=reuse_client)
    reuse_client.close()

    assert reused.downloaded is False
    assert reused.sha256 == digest
    assert reused.path == result.path


def test_llm_bootstrap_rejects_truncated_download(tmp_path: Path) -> None:
    config = LlmConfig(
        bootstrap=LlmBootstrapConfig(
            models_dir=tmp_path,
            profile="test",
            profiles={
                "test": LlmModelProfileConfig(
                    source="url",
                    url="https://models.test/model.gguf",
                    filename="model.gguf",
                    sha256=hashlib.sha256(b"complete").hexdigest(),
                )
            },
        )
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"half",
            headers={"Content-Length": "8"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="expected file size"):
        bootstrap_llm(config, client=client)
    client.close()

    assert not (tmp_path / "model.gguf").exists()
    assert (tmp_path / "model.gguf.part").read_bytes() == b"half"


def test_huggingface_checksum_receipt_prevents_corrupt_reuse(tmp_path: Path) -> None:
    payload = b"GGUF-pinned-contents"
    digest = hashlib.sha256(payload).hexdigest()
    config = LlmConfig(
        bootstrap=LlmBootstrapConfig(
            models_dir=tmp_path,
            profile="test",
            profiles={
                "test": LlmModelProfileConfig(
                    source="huggingface",
                    repository="org/repo",
                    revision="1234567890abcdef",
                    filename="model.gguf",
                )
            },
        )
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"X-Linked-Etag": f'"{digest}"'})
        return httpx.Response(
            200, content=payload, headers={"Content-Length": str(len(payload))}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = bootstrap_llm(config, client=client)
    assert first.sha256 == digest
    assert calls == ["HEAD", "GET"]
    assert (tmp_path / "model.gguf.integrity.json").is_file()

    def unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"reused verified model must be offline: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        reused = bootstrap_llm(config, client=client)
    assert reused.downloaded is False
    (tmp_path / "model.gguf").write_bytes(b"tampered")
    with (
        httpx.Client(transport=httpx.MockTransport(unexpected)) as client,
        pytest.raises(ValueError, match="SHA-256"),
    ):
        bootstrap_llm(config, client=client)


def test_url_download_must_have_pinned_checksum(tmp_path: Path) -> None:
    config = LlmConfig(
        bootstrap=LlmBootstrapConfig(
            models_dir=tmp_path,
            profile="unsigned",
            profiles={
                "unsigned": LlmModelProfileConfig(
                    source="url",
                    url="https://example.test/model.gguf",
                    filename="model.gguf",
                )
            },
        )
    )
    with pytest.raises(ValueError, match="must configure sha256"):
        bootstrap_llm(config)
