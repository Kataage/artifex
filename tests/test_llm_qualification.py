from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from artifex.config.models import LlmConfig
from artifex.db import Database
from artifex.llm import (
    LlmCallRepository,
    LlmQualificationService,
    OpenAICompatibleClient,
    StructuredGenerator,
)


@pytest.mark.asyncio
async def test_llm_qualification_records_versioned_provenance_and_metrics(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'qualification.sqlite3').as_posix()}")
    database.migrate()
    provenance = LlmCallRepository(database)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions/input_tokens":
            body = json.loads(request.content)
            text = "".join(
                str(message.get("content", ""))
                for message in body.get("messages", [])
            )
            return httpx.Response(200, json={"input_tokens": max(10, len(text) // 2)})
        if request.url.path == "/props":
            return httpx.Response(200, json={"build_info": "llama.cpp-test-build"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "concept_hook": "window silhouette",
                                        "setting": "sunset room",
                                        "mood": "calm",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 12},
                },
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://llm.test",
    )
    config = LlmConfig(
        base_url="http://llm.test",
        request_attempts=1,
        structured_repair_attempts=1,
        context_window_tokens=8192,
        reserved_output_tokens=1536,
        repair_headroom_tokens=512,
        token_count_mode="llama_cpp",
    )
    client = OpenAICompatibleClient(
        config,
        client=http_client,
        provenance=provenance,
    )
    service = LlmQualificationService(
        config,
        StructuredGenerator(client, repair_attempts=1),
        provenance,
    )

    report = await service.run(samples=4)

    assert report.samples_requested == 4
    assert report.samples_succeeded == 4
    assert report.structured_success_pct == 100.0
    assert report.call_count == 4
    assert report.average_context_utilization > 0
    assert report.response_repetition_pct > 0

    rows = provenance.recent(limit=10)
    assert rows
    latest = rows[0]
    assert latest["runtime_version"] == "llama.cpp-test-build"
    request_json = latest["request_json"]
    assert request_json["schema_version"] == "v1"
    assert request_json["prompt_version"] == "qualification_canary.v1"
    assert request_json["config_digest"]
    assert request_json["token_count_source"] == "llama_cpp"

    await http_client.aclose()
    database.dispose()
