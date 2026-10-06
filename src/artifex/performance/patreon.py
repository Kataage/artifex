from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from artifex.config.models import PatreonConfig
from artifex.performance.models import PublicationLink
from artifex.performance.repository import PerformanceRepository


class PatreonPost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    is_public: bool | None = None
    is_paid: bool | None = None
    tiers: tuple[str, ...] = ()
    raw: dict[str, Any] = Field(default_factory=dict)


class PatreonV2PublicationProvider:
    def __init__(
        self,
        config: PatreonConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._owned_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.api_base_url.rstrip("/"),
            timeout=config.api_timeout_seconds,
        )

    async def fetch_post(self, post_id: str) -> PatreonPost:
        if not self._config.api_enabled:
            raise RuntimeError("patreon.api_enabled is false")
        token = os.environ.get(self._config.api_access_token_env)
        if not token:
            raise RuntimeError(
                f"missing Patreon API token env: "
                f"{self._config.api_access_token_env}"
            )
        response = await self._client.get(
            f"/api/oauth2/v2/posts/{post_id}",
            params={
                "fields[post]": (
                    "title,url,published_at,is_public,is_paid,tiers"
                )
            },
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": self._config.api_user_agent,
                "Accept": "application/vnd.api+json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, dict):
            raise TypeError("Patreon post response is missing data object")
        attributes = data.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        raw_tiers = attributes.get("tiers")
        tiers: tuple[str, ...]
        if isinstance(raw_tiers, list | tuple):
            tiers = tuple(str(item) for item in raw_tiers)
        else:
            tiers = ()
        published_raw = attributes.get("published_at")
        published_at = None
        if isinstance(published_raw, str) and published_raw.strip():
            published_at = datetime.fromisoformat(published_raw)
        return PatreonPost(
            id=str(data.get("id") or post_id),
            title=(
                str(attributes["title"])
                if attributes.get("title") is not None
                else None
            ),
            url=(
                str(attributes["url"])
                if attributes.get("url") is not None
                else None
            ),
            published_at=published_at,
            is_public=(
                bool(attributes["is_public"])
                if attributes.get("is_public") is not None
                else None
            ),
            is_paid=(
                bool(attributes["is_paid"])
                if attributes.get("is_paid") is not None
                else None
            ),
            tiers=tiers,
            raw=payload,
        )

    async def sync_post(
        self,
        repository: PerformanceRepository,
        *,
        post_id: str,
        pack_id: str,
        scene_id: str | None = None,
        publication_tier: str | None = None,
    ) -> PublicationLink:
        post = await self.fetch_post(post_id)
        return repository.upsert_publication(
            platform="patreon",
            external_post_id=post.id,
            pack_id=pack_id,
            scene_id=scene_id,
            publication_tier=publication_tier,
            url=post.url,
            published_at=post.published_at,
            source="patreon_api_v2",
            metadata={
                "title": post.title,
                "is_public": post.is_public,
                "is_paid": post.is_paid,
                "tiers": list(post.tiers),
                "api_version": "v2",
            },
        )

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()
