from __future__ import annotations

from collections.abc import Iterable

from artifex.research.models import (
    ProviderHealth,
    ProviderResult,
    ProviderState,
    ResearchSearchRequest,
)
from artifex.research.provider import ResearchProvider


class ResearchProviderError(RuntimeError):
    pass


class ResearchRouter:
    def __init__(
        self,
        providers: Iterable[ResearchProvider],
        *,
        provider_order: tuple[str, ...],
    ) -> None:
        self._providers = {provider.name: provider for provider in providers}
        self._order = provider_order

    async def search(
        self,
        request: ResearchSearchRequest,
    ) -> tuple[tuple[ProviderResult, ...], tuple[str, ...]]:
        errors: list[str] = []
        candidates = [
            self._providers[name]
            for name in self._order
            if name in self._providers
            and request.source in self._providers[name].capabilities
        ]
        if request.backend is not None:
            candidates = [
                provider for provider in candidates
                if provider.name == request.backend
            ]

        if not candidates:
            raise ResearchProviderError(
                f"no research provider supports source={request.source.value}"
            )

        for provider in candidates:
            try:
                results = await provider.search(request)
            except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            if results:
                return results, tuple(errors)
            errors.append(f"{provider.name}: no results")

        raise ResearchProviderError(
            "all research providers failed: " + "; ".join(errors)
        )

    async def extract(
        self,
        provider_name: str,
        url: str,
        *,
        max_chars: int,
    ) -> str:
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ResearchProviderError(f"unknown research provider: {provider_name}")
        return await provider.extract(url, max_chars=max_chars)

    async def health(self) -> tuple[ProviderHealth, ...]:
        reports: list[ProviderHealth] = []
        for name in self._order:
            provider = self._providers.get(name)
            if provider is None:
                reports.append(
                    ProviderHealth(
                        provider=name,
                        state=ProviderState.DISABLED,
                        capabilities=(),
                        detail="not configured",
                    )
                )
                continue
            reports.append(await provider.health())
        return tuple(reports)

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
