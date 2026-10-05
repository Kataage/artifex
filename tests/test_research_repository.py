from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from artifex.db import Database
from artifex.research.models import (
    ProviderResult,
    ResearchIntent,
    ResearchRunState,
    ResearchSearchRequest,
    SearchSource,
)
from artifex.research.repository import ResearchRepository


def test_research_repository_persists_and_reuses_cached_search(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'research.sqlite3').as_posix()}")
    database.migrate()
    ids = iter(("run-1", "evidence-1"))
    repository = ResearchRepository(database, id_factory=lambda: next(ids))
    now = datetime(2026, 10, 6, 12, tzinfo=UTC)
    request = ResearchSearchRequest(
        query="Hololive fanart ideas",
        source=SearchSource.WEB,
        intent=ResearchIntent.CURRENT,
    )

    run_id = repository.create_run(
        request,
        expires_at=now + timedelta(hours=2),
        now=now,
    )
    evidence = repository.persist_results(
        run_id,
        (
            ProviderResult(
                provider="fake",
                source=SearchSource.WEB,
                url="https://example.com/path?b=2&a=1#fragment",
                title="Example",
                snippet="Useful external evidence",
            ),
        ),
        adult=False,
        max_snippet_chars=1200,
        now=now,
    )
    repository.complete_run(
        run_id,
        provider="fake",
        state=ResearchRunState.COMPLETED,
        completed_at=now,
    )

    cached = repository.cached_search(request, now=now + timedelta(minutes=30))

    assert len(evidence) == 1
    assert evidence[0].canonical_url == "https://example.com/path?a=1&b=2"
    assert cached is not None
    assert cached.cached is True
    assert cached.run_id == "run-1"
    assert cached.evidence[0].id == "evidence-1"
    database.dispose()
