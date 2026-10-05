from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException

from artifex.config.models import ResearchConfig
from artifex.research.models import (
    ProviderHealth,
    ProviderResult,
    ProviderState,
    ResearchSearchRequest,
    SearchSource,
)
from artifex.research.security import sanitize_text


class DDGSResearchProvider:
    name = "ddgs"
    capabilities = frozenset(
        {
            SearchSource.WEB,
            SearchSource.IMAGES,
            SearchSource.NEWS,
            SearchSource.VIDEOS,
        }
    )

    def __init__(self, config: ResearchConfig) -> None:
        self._config = config

    async def search(
        self,
        request: ResearchSearchRequest,
    ) -> tuple[ProviderResult, ...]:
        try:
            raw = await asyncio.to_thread(self._search_sync, request)
        except TimeoutException as exc:
            raise TimeoutError(f"DDGS search timed out: {exc}") from exc
        except DDGSException as exc:
            raise RuntimeError(f"DDGS search failed: {exc}") from exc

        return tuple(
            self._normalize(request.source, item)
            for item in raw
            if isinstance(item, dict)
        )

    def _search_sync(
        self,
        request: ResearchSearchRequest,
    ) -> list[dict[str, Any]]:
        client = DDGS(timeout=self._config.timeout_seconds)
        kwargs: dict[str, Any] = {
            "region": request.region,
            "safesearch": request.safesearch.value,
            "timelimit": request.timelimit,
            "max_results": request.max_results,
            "backend": request.backend or self._config.ddgs_backend,
        }
        if request.source is SearchSource.WEB:
            return client.text(request.query, **kwargs)
        if request.source is SearchSource.IMAGES:
            return client.images(request.query, **kwargs)
        if request.source is SearchSource.NEWS:
            return client.news(request.query, **kwargs)
        if request.source is SearchSource.VIDEOS:
            return client.videos(request.query, **kwargs)
        raise ValueError(f"DDGS does not support {request.source.value}")

    async def extract(self, url: str, *, max_chars: int) -> str:
        try:
            result = await asyncio.to_thread(
                lambda: DDGS(timeout=self._config.timeout_seconds).extract(
                    url,
                    fmt="text_plain",
                )
            )
        except TimeoutException as exc:
            raise TimeoutError(f"DDGS extraction timed out: {exc}") from exc
        except DDGSException as exc:
            raise RuntimeError(f"DDGS extraction failed: {exc}") from exc
        content = result.get("content", "")
        return sanitize_text(content, max_chars=max_chars)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.name,
            state=ProviderState.HEALTHY,
            capabilities=tuple(sorted(self.capabilities, key=lambda item: item.value)),
            detail="embedded Python provider available",
        )

    async def aclose(self) -> None:
        return None

    def _normalize(
        self,
        source: SearchSource,
        raw: dict[str, Any],
    ) -> ProviderResult:
        if source is SearchSource.WEB:
            url = str(raw.get("href", ""))
            title = sanitize_text(raw.get("title"), max_chars=500)
            snippet = sanitize_text(raw.get("body"), max_chars=self._config.max_snippet_chars)
            image_url = None
            published_at = None
            provider_source = None
        elif source is SearchSource.IMAGES:
            url = str(raw.get("url", raw.get("image", "")))
            title = sanitize_text(raw.get("title"), max_chars=500)
            snippet = sanitize_text(
                raw.get("source", ""),
                max_chars=self._config.max_snippet_chars,
            )
            image_url = str(raw.get("image", "")) or None
            published_at = None
            provider_source = sanitize_text(raw.get("source"), max_chars=200) or None
        elif source is SearchSource.NEWS:
            url = str(raw.get("url", ""))
            title = sanitize_text(raw.get("title"), max_chars=500)
            snippet = sanitize_text(raw.get("body"), max_chars=self._config.max_snippet_chars)
            image_url = str(raw.get("image", "")) or None
            published_at = _parse_datetime(raw.get("date"))
            provider_source = sanitize_text(raw.get("source"), max_chars=200) or None
        elif source is SearchSource.VIDEOS:
            url = str(raw.get("content", raw.get("embed_url", "")))
            title = sanitize_text(raw.get("title"), max_chars=500)
            snippet = sanitize_text(
                raw.get("description"),
                max_chars=self._config.max_snippet_chars,
            )
            image_url = None
            published_at = None
            provider_source = sanitize_text(raw.get("publisher"), max_chars=200) or None
        else:
            raise ValueError(f"unsupported DDGS result source: {source.value}")

        return ProviderResult(
            provider=self.name,
            source=source,
            url=url,
            title=title,
            snippet=snippet,
            image_url=image_url,
            published_at=published_at,
            provider_source=provider_source,
            metadata={
                key: value
                for key, value in raw.items()
                if key not in {
                    "title",
                    "href",
                    "body",
                    "url",
                    "image",
                    "date",
                    "source",
                    "content",
                    "description",
                    "publisher",
                }
                and isinstance(value, (str, int, float, bool, type(None)))
            },
        )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None
