from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

from artifex.config.models import EvaluationConfig
from artifex.evaluation.models import EvaluationContext, RawEvaluationSignals


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OpenAICompatibleVisionEvaluationProvider:
    def __init__(
        self,
        config: EvaluationConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not config.vision_base_url:
            raise ValueError("evaluation.vision_base_url must be configured")
        if not config.vision_model:
            raise ValueError("evaluation.vision_model must be configured")
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.vision_base_url.rstrip("/"),
            timeout=httpx.Timeout(config.vision_timeout_seconds),
        )

    async def evaluate(self, context: EvaluationContext) -> RawEvaluationSignals:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Evaluate the generated illustration. Score every requested field from "
                    "0 to 1. identity means the requested character(s) are visually "
                    "recognizable and consistent; alignment means the image follows the "
                    "scene/prompt; face_quality and technical_quality cover visible defects; "
                    "aesthetic covers overall presentation; continuity compares with any "
                    "adjacent images; integrity means the output is a complete decodable "
                    "image without obvious corruption. Use concise machine-readable reason "
                    "codes such as identity_low, face_quality_low, technical_quality_low, "
                    "continuity_low. Independently classify the visible output "
                    "content_rating as general, suggestive, adult, or explicit. "
                    "Use content_labels only when visibly supported, choosing concise "
                    "policy labels such as nudity, sexualized_nudity, sexual_activity, "
                    "fetish, age_ambiguous, sexualized_minor, "
                    "nonconsensual_sexual_content, incest, "
                    "synthetic_nonconsensual_intimate, real_person_sexual_activity, "
                    "hyperrealistic_sexual_activity, or real_animal_sexual_activity. "
                    "Do not decide publication eligibility; only describe/classify the "
                    "output. Requested characters: "
                    f"{', '.join(context.character_ids)}. Positive prompt: "
                    f"{context.positive_prompt}. Negative prompt: {context.negative_prompt}."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": _data_url(context.image_path)},
            },
        ]
        for character_id in context.character_ids:
            references = context.identity_reference_image_paths.get(character_id, ())
            for reference in references[:2]:
                if not reference.exists():
                    continue
                content.append(
                    {
                        "type": "text",
                        "text": f"Identity reference for requested character {character_id}:",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_url(reference)},
                    }
                )

        for adjacent in context.adjacent_image_paths[:2]:
            if adjacent.exists():
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_url(adjacent)},
                    }
                )

        payload: dict[str, Any] = {
            "model": self._config.vision_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict illustration quality evaluator. Do not rewrite "
                        "prompts and do not provide prose outside the JSON schema."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "artifex_evaluation",
                    "strict": True,
                    "schema": RawEvaluationSignals.model_json_schema(),
                },
            },
        }
        headers: dict[str, str] = {}
        if self._config.vision_api_key_env:
            token = os.environ.get(self._config.vision_api_key_env)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        last_error: Exception | None = None
        for attempt in range(self._config.vision_request_attempts):
            try:
                response = await self._client.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
                raw = body["choices"][0]["message"]["content"]
                if not isinstance(raw, str):
                    raise TypeError("vision evaluator response content must be a string")
                return RawEvaluationSignals.model_validate(json.loads(raw))
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt + 1 >= self._config.vision_request_attempts:
                    break
                await asyncio.sleep(
                    self._config.vision_retry_backoff_seconds * (attempt + 1)
                )

        assert last_error is not None
        raise RuntimeError("vision evaluation failed after configured attempts") from last_error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
