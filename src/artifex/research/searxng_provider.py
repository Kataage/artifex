from __future__ import annotations

from datetime import datetime
from typing import Any

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

_SAFESEARCH = {"off": 0, "moderate": 1, "on": 2}
_CATEGORY = {
    SearchSource.WEB: "general",
    SearchSource.IMAGES: "images",
    SearchSource.NEWS: "news",
    SearchSource.VIDEOS: "videos",
}


class SearXNGResearchProvider:
    name = "searxng"
    capabilities = frozenset(_CATEGORY)

    def __init__(
        self,
        config: ResearchConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not config.searxng_base_url:
            raise ValueError("research.searxng_base_url is not configured")
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.searxng_base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    async def search(
        self,
        request: ResearchSearchRequest,
    ) -> tuple[ProviderResult, ...]:
        category = _CATEGORY.get(request.source)
        if category is None:
            raise ValueError(f"SearXNG does not support {request.source.value}")
        params: dict[str, Any] = {
            "q": request.query,
            "format": "json",
            "categories": category,
            "language": request.region,
            "safesearch": _SAFESEARCH[request.safesearch.value],
        }
        if request.timelimit:
            params["time_range"] = {
                "d": "day",
                "w": "week",
                "m": "month",
                "y": "year",
            }[request.timelimit]

        try:
            response = await self._client.get("/search", params=params)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"SearXNG search failed: {exc}") from exc
        raw_results = body.get("results", ())
        if not isinstance(raw_results, list):
            raise TypeError("SearXNG response is missing results list")

        results: list[ProviderResult] = []
        for raw in raw_results[: request.max_results]:
            if not isinstance(raw, dict):
                continue
            url = str(raw.get("url", ""))
            results.append(
                ProviderResult(
                    provider=self.name,
                    source=request.source,
                    url=url,
                    title=sanitize_text(raw.get("title"), max_chars=500),
                    snippet=sanitize_text(
                        raw.get("content"),
                        max_chars=self._config.max_snippet_chars,
                    ),
                    image_url=str(raw.get("img_src", raw.get("thumbnail_src", ""))) or None,
                    published_at=_parse_datetime(raw.get("publishedDate")),
                    provider_source=sanitize_text(
                        raw.get("engine"),
                        max_chars=200,
                    ) or None,
                    score=(
                        float(raw["score"])
                        if isinstance(raw.get("score"), int | float)
                        else None
                    ),
                    metadata={
                        "category": raw.get("category"),
                        "engines": raw.get("engines"),
                    },
                )
            )
        return tuple(results)

    async def extract(self, url: str, *, max_chars: int) -> str:
        del url, max_chars
        raise RuntimeError("SearXNG provider does not implement content extraction")

    async def health(self) -> ProviderHealth:
        try:
            response = await self._client.get("/search", params={"q": "test", "format": "json"})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return ProviderHealth(
                provider=self.name,
                state=ProviderState.DEGRADED,
                capabilities=tuple(sorted(self.capabilities, key=lambda item: item.value)),
                detail=f"remote endpoint unavailable: {exc}",
            )
        return ProviderHealth(
            provider=self.name,
            state=ProviderState.HEALTHY,
            capabilities=tuple(sorted(self.capabilities, key=lambda item: item.value)),
            detail="remote endpoint reachable",
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None
