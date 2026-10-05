from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from artifex.config.models import LlmConfig
from artifex.llm import ChatMessage, OpenAICompatibleClient, StructuredGenerator


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class ScriptedClient:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        del messages, response_schema, schema_name
        output = self.outputs[self.calls]
        self.calls += 1
        return output


@pytest.mark.asyncio
async def test_structured_generator_repairs_invalid_output() -> None:
    client = ScriptedClient(["not-json", '{"value": 7}'])
    generator = StructuredGenerator(client, repair_attempts=1)

    answer = await generator.generate(
        Answer,
        [ChatMessage(role="user", content="return a value")],
        schema_name="answer",
    )

    assert answer.value == 7
    assert client.calls == 2


@pytest.mark.asyncio
async def test_openai_client_uses_json_schema_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"value": 1}'}}]},
        )

    monkeypatch.setenv("ARTIFEX_TEST_KEY", "secret")
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://llm.test",
    )
    client = OpenAICompatibleClient(
        LlmConfig(
            base_url="http://llm.test",
            api_key_env="ARTIFEX_TEST_KEY",
            request_attempts=1,
        ),
        client=http_client,
    )

    result = await client.complete(
        [ChatMessage(role="user", content="hello")],
        response_schema=Answer.model_json_schema(),
        schema_name="answer",
    )

    assert result == '{"value": 1}'
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert captured["authorization"] == "Bearer secret"
    await http_client.aclose()
