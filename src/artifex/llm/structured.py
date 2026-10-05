from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ValidationError

from artifex.llm.client import ChatMessage, LlmClient


class StructuredGenerationError(RuntimeError):
    pass


class StructuredGenerator:
    def __init__(self, client: LlmClient, *, repair_attempts: int = 2) -> None:
        if repair_attempts < 0:
            raise ValueError("repair_attempts must be non-negative")
        self._client = client
        self._repair_attempts = repair_attempts

    async def generate[ModelT: BaseModel](
        self,
        model_type: type[ModelT],
        messages: Sequence[ChatMessage],
        *,
        schema_name: str,
        post_validator: Callable[[ModelT], None] | None = None,
    ) -> ModelT:
        schema = model_type.model_json_schema()
        current_messages = list(messages)
        last_error: Exception | None = None

        for repair_index in range(self._repair_attempts + 1):
            output = await self._client.complete(
                current_messages,
                response_schema=schema,
                schema_name=schema_name,
            )
            try:
                data = json.loads(output)
                result = model_type.model_validate(data)
                if post_validator is not None:
                    post_validator(result)
                return result
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                last_error = exc
                if repair_index >= self._repair_attempts:
                    break
                current_messages = [
                    *messages,
                    ChatMessage(role="assistant", content=output),
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
