from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from artifex.llm.client import ChatMessage, LlmClient

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredGenerationError(RuntimeError):
    pass


class StructuredGenerator:
    def __init__(self, client: LlmClient, *, repair_attempts: int = 2) -> None:
        if repair_attempts < 0:
            raise ValueError("repair_attempts must be non-negative")
        self._client = client
        self._repair_attempts = repair_attempts

    async def generate(
        self,
        model_type: type[ModelT],
        messages: Sequence[ChatMessage],
        *,
        schema_name: str,
    ) -> ModelT:
        schema = model_type.model_json_schema()
        current_messages = list(messages)
        last_error: Exception | None = None
        last_output = ""

        for repair_index in range(self._repair_attempts + 1):
            last_output = await self._client.complete(
                current_messages,
                response_schema=schema,
                schema_name=schema_name,
            )
            try:
                data = json.loads(last_output)
                return model_type.model_validate(data)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                last_error = exc
                if repair_index >= self._repair_attempts:
                    break
                current_messages = [
                    *messages,
                    ChatMessage(role="assistant", content=last_output),
                    ChatMessage(
                        role="user",
                        content=(
                            "The previous response failed strict JSON/schema validation. "
                            "Return a corrected JSON object only, with no markdown or commentary. "
                            f"Validation error: {exc}"
                        ),
                    ),
                ]

        raise StructuredGenerationError(
            f"structured generation failed after {self._repair_attempts + 1} attempts"
        ) from last_error
