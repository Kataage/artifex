from __future__ import annotations

from typing import Protocol

from artifex.research.models import (
    ProviderHealth,
    ProviderResult,
    ResearchSearchRequest,
    SearchSource,
)


class ResearchProvider(Protocol):
    name: str
    capabilities: frozenset[SearchSource]

    async def search(
        self,
        request: ResearchSearchRequest,
    ) -> tuple[ProviderResult, ...]: ...

    async def extract(self, url: str, *, max_chars: int) -> str: ...

    async def health(self) -> ProviderHealth: ...

    async def aclose(self) -> None: ...
