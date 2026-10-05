from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from artifex.config.models import LlmConfig
from artifex.llm.budget import ContextBudgetExceeded
from artifex.llm.provenance import LlmCallRepository


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
        provenance: LlmCallRepository | None = None,
    ) -> None:
        self._config = config
        self._provenance = provenance
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_seconds),
        )
        self._runtime_version: str | None = None
        self._runtime_checked = False

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        payload = self._payload(messages, response_schema, schema_name)
        headers = self._headers()
        input_tokens, count_source = await self._count_input_tokens(payload, headers)

        if (
            self._config.enforce_token_budget
            and input_tokens > self._config.max_input_tokens
        ):
            raise ContextBudgetExceeded(
                input_tokens=input_tokens,
                max_input_tokens=self._config.max_input_tokens,
                context_window_tokens=self._config.context_window_tokens,
                reserved_output_tokens=self._config.reserved_output_tokens,
                repair_headroom_tokens=self._config.repair_headroom_tokens,
            )

        runtime_version = await self._get_runtime_version()
        context_digest = _digest_json(
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ]
        )
        schema_digest = _digest_json(response_schema)
        repair_index = sum(
            1 for message in messages if message.role == "assistant"
        )

        call_id: str | None = None
        if self._provenance is not None:
            call_id = self._provenance.begin(
                backend=self._config.backend,
                model=self._config.model,
                runtime_version=runtime_version,
                schema_name=schema_name,
                context_digest=context_digest,
                schema_digest=schema_digest,
                input_tokens=input_tokens,
                repair_index=repair_index,
                temperature=self._config.temperature,
                request_json={
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                    "response_schema": response_schema,
                    "max_tokens": self._config.reserved_output_tokens,
                    "token_count_source": count_source,
                },
            )

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
                output_tokens = _usage_tokens(body, "completion_tokens")
                if output_tokens is None:
                    output_tokens = await self._count_output_tokens(
                        content,
                        headers,
                    )
                if call_id is not None:
                    self._provenance.succeed(
                        call_id,
                        response_text=content,
                        response_digest=_digest_text(content),
                        output_tokens=output_tokens,
                    )
                return content
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt + 1 >= self._config.request_attempts:
                    break
                await asyncio.sleep(
                    self._config.retry_backoff_seconds * (attempt + 1)
                )

        assert last_error is not None
        if call_id is not None:
            self._provenance.fail(call_id, error_text=str(last_error))
        raise RuntimeError("LLM request failed after configured attempts") from last_error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.reserved_output_tokens,
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
        return payload

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._config.api_key_env:
            token = os.environ.get(self._config.api_key_env)
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _count_input_tokens(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[int, str]:
        mode = self._config.token_count_mode
        if mode == "disabled":
            return _estimate_tokens(payload), "estimate-disabled"
        if mode == "estimate":
            return _estimate_tokens(payload), "estimate"

        try:
            response = await self._client.post(
                "/v1/chat/completions/input_tokens",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()
            value = body.get("input_tokens")
            if isinstance(value, int) and value >= 0:
                return value, "llama_cpp"
        except (httpx.HTTPError, ValueError, TypeError):
            pass
        return _estimate_tokens(payload), "estimate-fallback"

    async def _count_output_tokens(
        self,
        content: str,
        headers: dict[str, str],
    ) -> int:
        if self._config.token_count_mode == "llama_cpp":
            try:
                response = await self._client.post(
                    "/tokenize",
                    json={"content": content, "add_special": False},
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
                tokens = body.get("tokens")
                if isinstance(tokens, list):
                    return len(tokens)
            except (httpx.HTTPError, ValueError, TypeError):
                pass
        return _estimate_text_tokens(content)

    async def _get_runtime_version(self) -> str | None:
        if self._runtime_checked:
            return self._runtime_version
        self._runtime_checked = True
        if self._provenance is None or self._config.backend != "llama_cpp":
            return None
        try:
            response = await self._client.get("/props")
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                return None
            build = body.get("build_info")
            if isinstance(build, str) and build.strip():
                self._runtime_version = build.strip()[:240]
        except (httpx.HTTPError, ValueError, TypeError):
            self._runtime_version = None
        return self._runtime_version


def _usage_tokens(body: object, key: str) -> int | None:
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return value if isinstance(value, int) and value >= 0 else None


def _estimate_tokens(payload: object) -> int:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _estimate_text_tokens(text)


def _estimate_text_tokens(text: str) -> int:
    # Conservative fallback for compatibility endpoints lacking llama.cpp's
    # exact token-count API. Native llama.cpp production uses exact counting.
    return max(1, len(text))


def _digest_json(payload: object) -> str:
    return _digest_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
