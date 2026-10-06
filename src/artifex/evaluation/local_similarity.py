from __future__ import annotations

import hashlib
import re
from pathlib import Path

from PIL import Image

from artifex.evaluation.semantic import EmbeddingModelDescriptor

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


class LocalSimilarityEmbeddingProvider:
    """Deterministic, dependency-light fallback for concept/image similarity."""

    def __init__(self, *, text_dimensions: int = 256, image_size: int = 8) -> None:
        if text_dimensions < 32:
            raise ValueError("text_dimensions must be at least 32")
        if image_size < 4:
            raise ValueError("image_size must be at least 4")
        self._text_dimensions = text_dimensions
        self._image_size = image_size

    @property
    def descriptor(self) -> EmbeddingModelDescriptor:
        return EmbeddingModelDescriptor(
            provider="local_hash_thumbnail",
            model=f"hash-{self._text_dimensions}/rgb-{self._image_size}",
            revision="v1",
            quality_tier="degraded",
        )

    async def aclose(self) -> None:
        return None

    async def embed_text(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._text_dimensions
        tokens = _TOKEN.findall(text.casefold())
        if not tokens:
            vector[0] = 1.0
            return tuple(vector)

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self._text_dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        vector[0] += 1e-6
        return tuple(vector)

    async def embed_image(self, path: Path) -> tuple[float, ...]:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(
                (self._image_size, self._image_size),
                Image.Resampling.LANCZOS,
            )
            pixels = tuple(rgb.getdata())

        vector: list[float] = []
        for red, green, blue in pixels:
            vector.extend(
                (
                    (red / 255.0) - 0.5,
                    (green / 255.0) - 0.5,
                    (blue / 255.0) - 0.5,
                )
            )
        vector.append(1e-6)
        return tuple(vector)
