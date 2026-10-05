from __future__ import annotations

import os
from urllib.parse import urlencode

import httpx

from artifex.config.models import ResearchConfig
from artifex.research.models import (
    ProviderHealth,
    ProviderResult,
    ProviderState,
    ResearchSearchRequest,
    SearchSource,
)
from artifex.research.security import sanitize_text


class GelbooruMetadataProvider:
    name = "gelbooru"
    capabilities = frozenset({SearchSource.TAGS})

    def __init__(
        self,
        config: ResearchConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.gelbooru_base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_seconds),
            headers={"User-Agent": "Artifex/0.1 research-metadata"},
        )

    async def search(
        self,
        request: ResearchSearchRequest,
    ) -> tuple[ProviderResult, ...]:
        if request.source is not SearchSource.TAGS:
            raise ValueError("Gelbooru metadata provider supports tags only")

        params: dict[str, str | int] = {
            "page": "dapi",
            "s": "tag",
            "q": "index",
            "json": 1,
            "name_pattern": request.query,
            "limit": request.max_results,
            "orderby": "count",
            "order": "DESC",
        }
        self._apply_auth(params)
        try:
            response = await self._client.get("/index.php", params=params)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"Gelbooru metadata search failed: {exc}") from exc
        raw_tags = _tag_list(body)

        results: list[ProviderResult] = []
        for raw in raw_tags[: request.max_results]:
            name = sanitize_text(raw.get("name"), max_chars=300)
            if not name:
                continue
            count = raw.get("count")
            tag_type = raw.get("type")
            query = urlencode({"page": "post", "s": "list", "tags": name})
            results.append(
                ProviderResult(
                    provider=self.name,
                    source=SearchSource.TAGS,
                    url=f"{self._config.gelbooru_base_url.rstrip('/')}/index.php?{query}",
                    title=name,
                    snippet=f"tag count={count}; type={tag_type}",
                    provider_source="gelbooru-tag-api",
                    score=float(count) if isinstance(count, int | float) else None,
                    metadata={
                        "tag_id": raw.get("id"),
                        "count": count,
                        "type": tag_type,
                    },
                )
            )
        return tuple(results)

    async def extract(self, url: str, *, max_chars: int) -> str:
        del url, max_chars
        raise RuntimeError("Gelbooru metadata provider does not fetch post content")

    async def health(self) -> ProviderHealth:
        if not self._config.gelbooru_enabled:
            return ProviderHealth(
                provider=self.name,
                state=ProviderState.DISABLED,
                capabilities=tuple(self.capabilities),
                detail="disabled by configuration",
            )
        return ProviderHealth(
            provider=self.name,
            state=ProviderState.HEALTHY,
            capabilities=tuple(self.capabilities),
            detail="metadata provider configured; credentials are optional until required",
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _apply_auth(self, params: dict[str, str | int]) -> None:
        key_env = self._config.gelbooru_api_key_env
        user_env = self._config.gelbooru_user_id_env
        api_key = os.environ.get(key_env) if key_env else None
        user_id = os.environ.get(user_env) if user_env else None
        if api_key and user_id:
            params["api_key"] = api_key
            params["user_id"] = user_id


def _tag_list(body: object) -> list[dict[str, object]]:
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if isinstance(body, dict):
        raw = body.get("tag", body.get("tags", ()))
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []
