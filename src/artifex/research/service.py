from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from artifex.config.models import ResearchConfig
from artifex.research.models import (
    ProviderHealth,
    ProviderResult,
    ResearchBrief,
    ResearchBriefItem,
    ResearchEvidence,
    ResearchIntent,
    ResearchRunState,
    ResearchSearchRequest,
    ResearchSearchResponse,
    SafeSearch,
    SearchSource,
)
from artifex.research.repository import ResearchRepository
from artifex.research.router import ResearchProviderError, ResearchRouter
from artifex.research.security import (
    normalize_query,
    sanitize_text,
    validate_external_url,
)


class ResearchPolicyError(ValueError):
    pass


class ResearchService:
    def __init__(
        self,
        config: ResearchConfig,
        repository: ResearchRepository,
        router: ResearchRouter,
    ) -> None:
        self._config = config
        self._repository = repository
        self._router = router

    async def search(
        self,
        request: ResearchSearchRequest,
        *,
        now: datetime | None = None,
        use_cache: bool = True,
    ) -> ResearchSearchResponse:
        current = now or datetime.now(UTC)
        effective = self._effective_request(request)

        if use_cache:
            cached = self._repository.cached_search(effective, now=current)
            if cached is not None:
                return cached

        since = current - timedelta(hours=24)
        if self._repository.count_runs_since(since) >= self._config.daily_request_budget:
            raise ResearchPolicyError("daily research request budget is exhausted")

        expires_at = current + self._ttl(effective.intent)
        run_id = self._repository.create_run(
            effective,
            expires_at=expires_at,
            now=current,
        )
        try:
            raw_results, errors = await self._router.search(effective)
            safe_results = self._filter_results(raw_results)
            if not safe_results:
                raise ResearchProviderError("all provider results failed URL/security validation")
            evidence = self._repository.persist_results(
                run_id,
                safe_results[: effective.max_results],
                adult=effective.adult,
                max_snippet_chars=self._config.max_snippet_chars,
                now=current,
            )
            if not evidence:
                raise ResearchProviderError("provider returned no persistable evidence")
        except (ResearchProviderError, RuntimeError, ValueError, TimeoutError, OSError) as exc:
            self._repository.fail_run(run_id, error=str(exc), completed_at=current)
            raise

        state = ResearchRunState.DEGRADED if errors else ResearchRunState.COMPLETED
        self._repository.complete_run(
            run_id,
            provider=evidence[0].provider,
            state=state,
            errors=errors,
            completed_at=current,
        )
        return ResearchSearchResponse(
            run_id=run_id,
            query=effective.query,
            source=effective.source,
            intent=effective.intent,
            cached=False,
            degraded=bool(errors),
            evidence=evidence,
            errors=errors,
        )

    async def inspect(
        self,
        evidence_id: str,
        *,
        extract: bool = False,
    ) -> dict[str, object]:
        evidence = self._repository.require_evidence(evidence_id)
        payload = evidence.model_dump(mode="json")
        if not extract:
            return payload

        if evidence.source is SearchSource.TAGS:
            payload["extracted_text"] = ""
            payload["extraction_note"] = "tag metadata evidence does not fetch post content"
            return payload

        url = validate_external_url(evidence.canonical_url, self._config)
        try:
            extracted = await self._router.extract(
                evidence.provider,
                url,
                max_chars=self._config.max_extract_chars,
            )
        except (ResearchProviderError, RuntimeError, ValueError, TimeoutError, OSError):
            extracted = await self._router.extract(
                "ddgs",
                url,
                max_chars=self._config.max_extract_chars,
            )
        extracted = sanitize_text(extracted, max_chars=self._config.max_extract_chars)
        self._repository.patch_evidence_metadata(
            evidence_id,
            {"extracted_text": extracted, "untrusted_external_data": True},
        )
        payload["extracted_text"] = extracted
        payload["untrusted_external_data"] = True
        return payload

    async def brief(
        self,
        topic: str,
        requests: Sequence[ResearchSearchRequest],
        *,
        adult: bool = False,
        now: datetime | None = None,
        use_cache: bool = True,
    ) -> ResearchBrief:
        current = now or datetime.now(UTC)
        if adult and not self._config.adult_enabled:
            raise ResearchPolicyError("adult-rated research is disabled")
        bounded = tuple(requests[: self._config.max_queries_per_cycle])
        if not bounded:
            raise ValueError("ResearchBrief requires at least one search request")

        cache_key = self._brief_cache_key(topic, bounded, adult)
        if use_cache:
            cached = self._repository.cached_brief(
                cache_key,
                adult=adult,
                now=current,
            )
            if cached is not None:
                return cached

        responses: list[ResearchSearchResponse] = []
        errors: list[str] = []
        started = time.monotonic()
        for request in bounded:
            if time.monotonic() - started >= self._config.max_cycle_seconds:
                errors.append("research cycle time budget exhausted")
                break
            effective = request.model_copy(update={"adult": adult or request.adult})
            try:
                responses.append(await self.search(effective, now=current))
            except (ResearchProviderError, ResearchPolicyError, RuntimeError, ValueError, TimeoutError, OSError) as exc:
                errors.append(f"{request.source.value}:{request.intent.value}: {exc}")

        evidence = self._select_evidence(responses)
        items: list[ResearchBriefItem] = []
        findings: list[str] = []
        used_chars = 0
        for item in evidence:
            finding = sanitize_text(
                f"{item.title}: {item.snippet}" if item.snippet else item.title,
                max_chars=self._config.max_snippet_chars,
            )
            if not finding:
                continue
            projected = used_chars + len(finding)
            if projected > self._config.max_brief_chars:
                break
            used_chars = projected
            findings.append(finding)
            items.append(
                ResearchBriefItem(
                    evidence_id=item.id,
                    source=item.source,
                    provider=item.provider,
                    title=item.title,
                    finding=finding,
                    url=item.canonical_url,
                    published_at=item.published_at,
                )
            )

        degraded = bool(errors) or not items or any(response.degraded for response in responses)
        if not findings:
            findings.append(
                "No fresh external evidence was available. "
                "Do not make current/trend claims; use evergreen/exploration only."
            )

        run_ids = tuple(dict.fromkeys(response.run_id for response in responses))
        evidence_ids = tuple(item.evidence_id for item in items)
        expires_at = current + min(
            (self._ttl(request.intent) for request in bounded),
            default=timedelta(hours=self._config.cache_ttl_hours),
        )
        confidence = self._freshness_confidence(evidence, current)
        if degraded:
            confidence = min(confidence, 0.5)

        brief = ResearchBrief(
            id=self._repository.new_id()
            topic=sanitize_text(topic, max_chars=500),
            research_run_ids=run_ids,
            evidence_ids=evidence_ids,
            items=tuple(items),
            key_findings=tuple(findings),
            generated_at=current,
            expires_at=expires_at,
            freshness_confidence=confidence,
            degraded=degraded,
            adult=adult,
            cache_key=cache_key,
        )
        return self._repository.save_brief(brief)

    async def health(self) -> tuple[ProviderHealth, ...]:
        return await self._router.health()

    async def aclose(self) -> None:
        await self._router.aclose()

    def _effective_request(
        self,
        request: ResearchSearchRequest,
    ) -> ResearchSearchRequest:
        adult = request.adult or request.intent is ResearchIntent.ADULT
        if adult and not self._config.adult_enabled:
            raise ResearchPolicyError("adult-rated research is disabled")
        safesearch = (
            SafeSearch(self._config.adult_safesearch)
            if adult
            else request.safesearch
        )
        return request.model_copy(
            update={
                "adult": adult,
                "safesearch": safesearch,
                "max_results": min(
                    request.max_results,
                    self._config.max_results_per_query,
                ),
            }
        )

    def _filter_results(
        self,
        results: Sequence[ProviderResult],
    ) -> tuple[ProviderResult, ...]:
        safe: list[ProviderResult] = []
        for result in results:
            try:
                canonical = validate_external_url(result.url, self._config)
                image_url = result.image_url
                if image_url:
                    validate_external_url(image_url, self._config)
            except ValueError:
                continue
            safe.append(
                result.model_copy(
                    update={
                        "url": canonical,
                        "title": sanitize_text(result.title, max_chars=500),
                        "snippet": sanitize_text(
                            result.snippet,
                            max_chars=self._config.max_snippet_chars,
                        ),
                    }
                )
            )
        return tuple(safe)

    def _select_evidence(
        self,
        responses: Sequence[ResearchSearchResponse],
    ) -> tuple[ResearchEvidence, ...]:
        deduped: dict[str, ResearchEvidence] = {}
        for response in responses:
            for item in response.evidence:
                existing = deduped.get(item.content_hash)
                if existing is None or item.rank < existing.rank:
                    deduped[item.content_hash] = item
        ordered = sorted(
            deduped.values(),
            key=lambda item: (
                item.rank,
                item.source.value,
                item.provider,
                item.id,
            ),
        )
        return tuple(ordered[: self._config.max_brief_items])

    def _ttl(self, intent: ResearchIntent) -> timedelta:
        if intent is ResearchIntent.CURRENT:
            hours = self._config.current_cache_ttl_hours
        elif intent is ResearchIntent.EVERGREEN:
            hours = self._config.evergreen_cache_ttl_hours
        else:
            hours = self._config.cache_ttl_hours
        return timedelta(hours=hours)

    @staticmethod
    def _freshness_confidence(
        evidence: Sequence[ResearchEvidence],
        now: datetime,
    ) -> float:
        if not evidence:
            return 0.0
        scores: list[float] = []
        for item in evidence:
            timestamp = item.published_at or item.observed_at
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            age_hours = max(
                0.0,
                (now.astimezone(UTC) - timestamp.astimezone(UTC)).total_seconds()
                / 3600.0,
            )
            scores.append(math.exp(-age_hours / (24.0 * 14.0)))
        return max(0.0, min(1.0, sum(scores) / len(scores)))

    @staticmethod
    def _brief_cache_key(
        topic: str,
        requests: Sequence[ResearchSearchRequest],
        adult: bool,
    ) -> str:
        payload = {
            "topic": normalize_query(topic),
            "adult": adult,
            "requests": [
                {
                    "query": normalize_query(request.query),
                    "source": request.source.value,
                    "intent": request.intent.value,
                    "region": request.region,
                    "safesearch": request.safesearch.value,
                    "timelimit": request.timelimit,
                    "backend": request.backend,
                }
                for request in requests
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
