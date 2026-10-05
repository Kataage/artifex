from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from artifex.config.models import LlmConfig


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LlmClient(Protocol):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> str: ...


class OpenAICompatibleClient:
    """OpenAI-compatible chat client suitable for llama.cpp's llama-server."""

    def __init__(
        self,
        config: LlmConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self._config.temperature,
        }

        if self._config.structured_output == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        headers: dict[str, str] = {}
        if self._config.api_key_env:
            token = os.environ.get(self._config.api_key_env)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        last_error: Exception | None = None
        for attempt in range(self._config.request_attempts):
            try:
                response = await self._client.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("LLM response content must be a string")
                return content
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 >= self._config.request_attempts:
                    break
                await asyncio.sleep(self._config.retry_backoff_seconds * (attempt + 1))

        assert last_error is not None
        raise RuntimeError("LLM request failed after configured attempts") from last_error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
