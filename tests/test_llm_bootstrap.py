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
