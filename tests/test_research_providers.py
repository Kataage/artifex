from __future__ import annotations

from typing import Any

import httpx
import pytest

from artifex.config.models import ResearchConfig
from artifex.research.ddgs_provider import DDGSResearchProvider
from artifex.research.gelbooru_provider import GelbooruMetadataProvider
from artifex.research.models import (
    ResearchIntent,
    ResearchSearchRequest,
    SearchSource,
)
from artifex.research.searxng_provider import SearXNGResearchProvider


class FakeDDGS:
    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        del query, kwargs
        return [{"title": "Web", "href": "https://example.com", "body": "Body"}]

    def images(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        del query, kwargs
        return [{
            "title": "Image",
            "url": "https://example.com/page",
            "image": "https://example.com/image.jpg",
            "source": "Bing",
        }]

    def news(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        del query, kwargs
        return [{
            "title": "News",
            "url": "https://example.com/news",
            "body": "News body",
            "date": "2026-10-06T00:00:00+00:00",
            "source": "Bing",
        }]

    def videos(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        del query, kwargs
        return [{
            "title": "Video",
            "content": "https://example.com/video",
            "description": "Description",
            "publisher": "Publisher",
        }]

    def extract(self, url: str, fmt: str) -> dict[str, str]:
        del url, fmt
        return {"url": "https://example.com", "content": "Extracted content"}


@pytest.mark.asyncio
async def test_ddgs_provider_normalizes_all_native_search_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("artifex.research.ddgs_provider.DDGS", FakeDDGS)
    provider = DDGSResearchProvider(ResearchConfig())

    for source in (
        SearchSource.WEB,
        SearchSource.IMAGES,
        SearchSource.NEWS,
        SearchSource.VIDEOS,
    ):
        results = await provider.search(
            ResearchSearchRequest(
                query="test",
                source=source,
                intent=ResearchIntent.EVERGREEN,
            )
        )
        assert len(results) == 1
        assert results[0].provider == "ddgs"
        assert results[0].source is source

    assert await provider.extract("https://example.com", max_chars=100) == "Extracted content"


@pytest.mark.asyncio
async def test_optional_searxng_provider_uses_common_schema() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        return httpx.Response(
            200,
            json={
                "results": [{
                    "url": "https://example.com/searx",
                    "title": "SearX",
                    "content": "Result",
                    "engine": "bing",
                    "score": 0.8,
                }]
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://searx.test",
    )
    provider = SearXNGResearchProvider(
        ResearchConfig(searxng_base_url="http://searx.test"),
        client=client,
    )
    results = await provider.search(
        ResearchSearchRequest(query="test", source=SearchSource.WEB)
    )

    assert results[0].provider == "searxng"
    assert results[0].url == "https://example.com/searx"
    await client.aclose()


@pytest.mark.asyncio
async def test_gelbooru_provider_returns_metadata_only_tag_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/index.php"
        return httpx.Response(
            200,
            json={
                "tag": [{
                    "id": 1,
                    "name": "example_tag",
                    "count": 1234,
                    "type": 0,
                }]
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://gelbooru.test",
    )
    provider = GelbooruMetadataProvider(
        ResearchConfig(gelbooru_base_url="https://gelbooru.test"),
        client=client,
    )
    results = await provider.search(
        ResearchSearchRequest(
            query="example%",
            source=SearchSource.TAGS,
        )
    )

    assert len(results) == 1
    assert results[0].title == "example_tag"
    assert results[0].metadata["count"] == 1234
    assert results[0].image_url is None
    await client.aclose()
