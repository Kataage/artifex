from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import func, select

from artifex.db import Database
from artifex.db.models import ResearchBriefRow, ResearchEvidenceRow, ResearchRunRow
from artifex.research.models import (
    ProviderResult,
    ResearchBrief,
    ResearchEvidence,
    ResearchRunState,
    ResearchSearchRequest,
    ResearchSearchResponse,
)
from artifex.research.security import (
    canonicalize_url,
    content_hash,
    normalize_query,
    sanitize_text,
)


def _new_id() -> str:
    return uuid4().hex


class ResearchRepository:
    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def new_id(self) -> str:
        return self._id_factory()

    def count_runs_since(self, since: datetime) -> int:
        with self._database.session() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(ResearchRunRow)
                    .where(ResearchRunRow.requested_at >= since)
                )
                or 0
            )

    def create_run(
        self,
        request: ResearchSearchRequest,
        *,
        expires_at: datetime,
        now: datetime | None = None,
    ) -> str:
        requested_at = now or datetime.now(UTC)
        run_id = self._id_factory()
        with self._database.session() as session:
            session.add(
                ResearchRunRow(
                    id=run_id,
                    query=request.query,
                    normalized_query=normalize_query(request.query),
                    intent=request.intent.value,
                    source=request.source.value,
                    provider=None,
                    safesearch=request.safesearch.value,
                    adult=request.adult,
                    state=ResearchRunState.RUNNING.value,
                    requested_at=requested_at,
                    completed_at=None,
                    expires_at=expires_at,
                    payload_json={
                        "region": request.region,
                        "timelimit": request.timelimit,
                        "max_results": request.max_results,
                        "backend": request.backend,
                    },
                )
            )
        return run_id

    def complete_run(
        self,
        run_id: str,
        *,
        provider: str,
        state: ResearchRunState,
        errors: Sequence[str] = (),
        completed_at: datetime | None = None,
    ) -> None:
        if state not in {ResearchRunState.COMPLETED, ResearchRunState.DEGRADED}:
            raise ValueError("complete_run requires completed/degraded state")
        when = completed_at or datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(ResearchRunRow, run_id)
            if row is None:
                raise KeyError(f"unknown research run: {run_id}")
            row.provider = provider
            row.state = state.value
            row.completed_at = when
            payload = dict(row.payload_json)
            payload["errors"] = list(errors)
            row.payload_json = payload

    def fail_run(
        self,
        run_id: str,
        *,
        error: str,
        completed_at: datetime | None = None,
    ) -> None:
        when = completed_at or datetime.now(UTC)
        with self._database.session() as session:
            row = session.get(ResearchRunRow, run_id)
            if row is None:
                raise KeyError(f"unknown research run: {run_id}")
            row.state = ResearchRunState.FAILED.value
            row.completed_at = when
            payload = dict(row.payload_json)
            payload["errors"] = [error]
            row.payload_json = payload

    def persist_results(
        self,
        run_id: str,
        results: Sequence[ProviderResult],
        *,
        adult: bool,
        max_snippet_chars: int,
        now: datetime | None = None,
    ) -> tuple[ResearchEvidence, ...]:
        observed_at = now or datetime.now(UTC)
        seen_urls: set[str] = set()
        evidence: list[ResearchEvidence] = []

        with self._database.session() as session:
            if session.get(ResearchRunRow, run_id) is None:
                raise KeyError(f"unknown research run: {run_id}")

            for result in results:
                try:
                    canonical_url = canonicalize_url(result.url)
                except ValueError:
                    continue
                if canonical_url in seen_urls:
                    continue
                seen_urls.add(canonical_url)

                title = sanitize_text(result.title, max_chars=500)
                snippet = sanitize_text(
                    result.snippet,
                    max_chars=max_snippet_chars,
                )
                digest = content_hash(
                    canonical_url,
                    title,
                    snippet,
                    result.image_url or "",
                )
                item = ResearchEvidence(
                    id=self._id_factory(),
                    run_id=run_id,
                    provider=result.provider,
                    source=result.source,
                    rank=len(evidence) + 1,
                    url=result.url,
                    canonical_url=canonical_url,
                    title=title,
                    snippet=snippet,
                    image_url=result.image_url,
                    published_at=result.published_at,
                    observed_at=observed_at,
                    content_hash=digest,
                    adult=adult,
                    metadata={
                        **result.metadata,
                        "provider_source": result.provider_source,
                        "score": result.score,
                    },
                )
                evidence.append(item)
                session.add(
                    ResearchEvidenceRow(
                        id=item.id,
                        run_id=item.run_id,
                        provider=item.provider,
                        source=item.source.value,
                        rank=item.rank,
                        url=item.url,
                        canonical_url=item.canonical_url,
                        title=item.title,
                        snippet=item.snippet,
                        image_url=item.image_url,
                        published_at=item.published_at,
                        observed_at=item.observed_at,
                        content_hash=item.content_hash,
                        adult=item.adult,
                        metadata_json=item.metadata,
                    )
                )
        return tuple(evidence)

    def cached_search(
        self,
        request: ResearchSearchRequest,
        *,
        now: datetime | None = None,
    ) -> ResearchSearchResponse | None:
        current = now or datetime.now(UTC)
        with self._database.session() as session:
            row = session.scalar(
                select(ResearchRunRow)
                .where(
                    ResearchRunRow.normalized_query == normalize_query(request.query),
                    ResearchRunRow.intent == request.intent.value,
                    ResearchRunRow.source == request.source.value,
                    ResearchRunRow.safesearch == request.safesearch.value,
                    ResearchRunRow.adult == request.adult,
                    ResearchRunRow.state.in_(
                        (
                            ResearchRunState.COMPLETED.value,
                            ResearchRunState.DEGRADED.value,
                        )
                    ),
                    ResearchRunRow.expires_at.is_not(None),
                    ResearchRunRow.expires_at > current,
                )
                .order_by(ResearchRunRow.completed_at.desc(), ResearchRunRow.id.desc())
                .limit(1)
            )
            if row is None:
                return None
            evidence_rows = session.scalars(
                select(ResearchEvidenceRow)
                .where(ResearchEvidenceRow.run_id == row.id)
                .order_by(ResearchEvidenceRow.rank.asc(), ResearchEvidenceRow.id.asc())
            ).all()
            errors = row.payload_json.get("errors", ())
            if not isinstance(errors, list):
                errors = []

        return ResearchSearchResponse(
            run_id=row.id,
            query=row.query,
            source=request.source,
            intent=request.intent,
            cached=True,
            degraded=row.state == ResearchRunState.DEGRADED.value,
            evidence=tuple(self._evidence_from_row(item) for item in evidence_rows),
            errors=tuple(str(item) for item in errors),
        )

    def evidence(self, evidence_id: str) -> ResearchEvidence | None:
        with self._database.session() as session:
            row = session.get(ResearchEvidenceRow, evidence_id)
            return None if row is None else self._evidence_from_row(row)

    def require_evidence(self, evidence_id: str) -> ResearchEvidence:
        item = self.evidence(evidence_id)
        if item is None:
            raise KeyError(f"unknown research evidence: {evidence_id}")
        return item

    def patch_evidence_metadata(
        self,
        evidence_id: str,
        patch: dict[str, object],
    ) -> ResearchEvidence:
        with self._database.session() as session:
            row = session.get(ResearchEvidenceRow, evidence_id)
            if row is None:
                raise KeyError(f"unknown research evidence: {evidence_id}")
            row.metadata_json = {**row.metadata_json, **patch}
        return self.require_evidence(evidence_id)

    def save_brief(self, brief: ResearchBrief) -> ResearchBrief:
        with self._database.session() as session:
            session.add(
                ResearchBriefRow(
                    id=brief.id,
                    topic=brief.topic,
                    cache_key=brief.cache_key,
                    adult=brief.adult,
                    state=(
                        ResearchRunState.DEGRADED.value
                        if brief.degraded
                        else ResearchRunState.COMPLETED.value
                    ),
                    payload_json=brief.model_dump(mode="json"),
                    created_at=brief.generated_at,
                    expires_at=brief.expires_at,
                )
            )
        return brief

    def cached_brief(
        self,
        cache_key: str,
        *,
        adult: bool,
        now: datetime | None = None,
    ) -> ResearchBrief | None:
        current = now or datetime.now(UTC)
        with self._database.session() as session:
            row = session.scalar(
                select(ResearchBriefRow)
                .where(
                    ResearchBriefRow.cache_key == cache_key,
                    ResearchBriefRow.adult == adult,
                    ResearchBriefRow.expires_at > current,
                )
                .order_by(
                    ResearchBriefRow.created_at.desc(),
                    ResearchBriefRow.id.desc(),
                )
                .limit(1)
            )
            if row is None:
                return None
            return ResearchBrief.model_validate(row.payload_json)

    @staticmethod
    def _evidence_from_row(row: ResearchEvidenceRow) -> ResearchEvidence:
        return ResearchEvidence(
            id=row.id,
            run_id=row.run_id,
            provider=row.provider,
            source=row.source,
            rank=row.rank,
            url=row.url,
            canonical_url=row.canonical_url,
            title=row.title,
            snippet=row.snippet,
            image_url=row.image_url,
            published_at=row.published_at,
            observed_at=row.observed_at,
            content_hash=row.content_hash,
            adult=row.adult,
            metadata=dict(row.metadata_json),
        )
