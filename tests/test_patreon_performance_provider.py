from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from artifex.config.models import PatreonConfig
from artifex.db import Database
from artifex.db.models import PackRow
from artifex.domain import PackState
from artifex.performance import PerformanceRepository
from artifex.performance.patreon import PatreonV2PublicationProvider


@pytest.mark.asyncio
async def test_patreon_v2_sync_binds_canonical_post_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'patreon.sqlite3').as_posix()}")
    database.migrate()
    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            PackRow(
                id="pack-1",
                state=PackState.FINALIZED.value,
                format_type="evergreen",
                payload_json={},
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": {
                    "type": "post",
                    "id": "12345",
                    "attributes": {
                        "title": "Artifex post",
                        "url": "https://www.patreon.com/posts/12345",
                        "published_at": "2026-10-06T12:00:00Z",
                        "is_public": False,
                        "is_paid": False,
                        "tiers": ["tier-a"],
                    },
                }
            },
        )

    monkeypatch.setenv("PATREON_ACCESS_TOKEN", "secret-token")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://www.patreon.com",
    )
    provider = PatreonV2PublicationProvider(
        PatreonConfig(api_enabled=True),
        client=client,
    )
    repository = PerformanceRepository(database)

    link = await provider.sync_post(
        repository,
        post_id="12345",
        pack_id="pack-1",
        publication_tier="member",
    )

    assert link.external_post_id == "12345"
    assert link.pack_id == "pack-1"
    assert link.publication_tier == "member"
    assert link.source == "patreon_api_v2"
    assert link.metadata["title"] == "Artifex post"
    assert link.metadata["tiers"] == ["tier-a"]
    assert repository.snapshots_for_publication(link.id) == ()
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer secret-token"
    assert requests[0].headers["user-agent"] == "Artifex - Patreon Performance Sync"
    assert "/api/oauth2/v2/posts/12345" in str(requests[0].url)

    await client.aclose()
    database.dispose()


@pytest.mark.asyncio
async def test_patreon_v2_sync_requires_explicit_opt_in(
    tmp_path: Path,
) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(500)
        ),
        base_url="https://www.patreon.com",
    )
    provider = PatreonV2PublicationProvider(PatreonConfig(), client=client)

    with pytest.raises(RuntimeError, match="api_enabled"):
        await provider.fetch_post("12345")

    await client.aclose()


@pytest.mark.asyncio
async def test_patreon_v2_sync_requires_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PATREON_ACCESS_TOKEN", raising=False)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(500)
        ),
        base_url="https://www.patreon.com",
    )
    provider = PatreonV2PublicationProvider(
        PatreonConfig(api_enabled=True),
        client=client,
    )

    assert os.environ.get("PATREON_ACCESS_TOKEN") is None
    with pytest.raises(RuntimeError, match="missing Patreon API token"):
        await provider.fetch_post("12345")

    await client.aclose()



@pytest.mark.asyncio
async def test_patreon_v2_http_failure_is_wrapped_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATREON_ACCESS_TOKEN", "secret-token")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(503, text="unavailable")
        ),
        base_url="https://www.patreon.com",
    )
    provider = PatreonV2PublicationProvider(
        PatreonConfig(api_enabled=True),
        client=client,
    )

    with pytest.raises(RuntimeError, match="Patreon API request failed"):
        await provider.fetch_post("12345")

    await client.aclose()
