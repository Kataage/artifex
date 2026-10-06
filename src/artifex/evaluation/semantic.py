from __future__ import annotations

import asyncio
import hashlib
import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from PIL import Image
from sqlalchemy import func, select

from artifex.db import Database
from artifex.db.models import SemanticEmbeddingRow


@dataclass(frozen=True, slots=True)
class EmbeddingModelDescriptor:
    provider: str
    model: str
    revision: str
    quality_tier: Literal["production", "degraded"]


class DescribedEmbeddingProvider(Protocol):
    @property
    def descriptor(self) -> EmbeddingModelDescriptor: ...

    async def embed_text(self, text: str) -> Sequence[float]: ...

    async def embed_image(self, path: Path) -> Sequence[float]: ...


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm == 0:
        raise ValueError("semantic embedding has zero norm")
    return tuple(float(value) / norm for value in values)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class SigLIP2EmbeddingProvider:
    """Lazy local SigLIP2 embeddings suitable for Windows-native production."""

    def __init__(
        self,
        *,
        model: str = "google/siglip2-base-patch16-224",
        revision: str | None = None,
        device: Literal["cpu", "cuda", "auto"] = "cpu",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        self._model_id = model
        self._requested_revision = revision
        self._resolved_revision = revision or "main"
        self._device_request = device
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only
        self._torch: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device = "cpu"

    @property
    def descriptor(self) -> EmbeddingModelDescriptor:
        return EmbeddingModelDescriptor(
            provider="transformers_siglip2",
            model=self._model_id,
            revision=self._resolved_revision,
            quality_tier="production",
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP2 semantic embeddings require the 'semantic' extra: "
                "uv sync --extra semantic"
            ) from exc

        auto_model = transformers.AutoModel
        auto_processor = transformers.AutoProcessor
        load_kwargs: dict[str, Any] = {
            "local_files_only": self._local_files_only,
            "trust_remote_code": False,
        }
        if self._requested_revision is not None:
            load_kwargs["revision"] = self._requested_revision
        if self._cache_dir is not None:
            load_kwargs["cache_dir"] = str(self._cache_dir)

        processor = auto_processor.from_pretrained(self._model_id, **load_kwargs)
        model = auto_model.from_pretrained(self._model_id, **load_kwargs)

        if self._device_request == "auto":
            device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
        else:
            device = self._device_request
        if device == "cuda" and not bool(torch.cuda.is_available()):
            raise RuntimeError("semantic_device=cuda but CUDA is unavailable")

        model = model.to(device)
        model.eval()
        resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
        if isinstance(resolved, str) and resolved:
            self._resolved_revision = resolved

        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device

    def _move_inputs(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in values.items()
        }

    def _embed_text_sync(self, text: str) -> tuple[float, ...]:
        self._ensure_loaded()
        assert self._torch is not None
        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(
            text=[text],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        moved = self._move_inputs(dict(inputs))
        with self._torch.inference_mode():
            features = self._model.get_text_features(**moved)
        values = features[0].detach().float().cpu().tolist()
        return _normalize(tuple(float(value) for value in values))

    def _embed_image_sync(self, path: Path) -> tuple[float, ...]:
        self._ensure_loaded()
        assert self._torch is not None
        assert self._processor is not None
        assert self._model is not None
        with Image.open(path) as source:
            image = source.convert("RGB").copy()
        inputs = self._processor(images=image, return_tensors="pt")
        moved = self._move_inputs(dict(inputs))
        with self._torch.inference_mode():
            features = self._model.get_image_features(**moved)
        values = features[0].detach().float().cpu().tolist()
        return _normalize(tuple(float(value) for value in values))

    async def embed_text(self, text: str) -> tuple[float, ...]:
        return await asyncio.to_thread(self._embed_text_sync, text)

    async def embed_image(self, path: Path) -> tuple[float, ...]:
        return await asyncio.to_thread(self._embed_image_sync, path)

    async def aclose(self) -> None:
        if self._model is None:
            return
        torch = self._torch
        model = self._model
        self._model = None
        self._processor = None
        if torch is not None and self._device == "cuda":
            await asyncio.to_thread(model.to, "cpu")
            torch.cuda.empty_cache()


class SemanticEmbeddingRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def get(
        self,
        *,
        subject_type: str,
        subject_id: str,
        modality: str,
        descriptor: EmbeddingModelDescriptor,
        content_hash: str,
    ) -> tuple[float, ...] | None:
        with self._database.session() as session:
            row = session.scalar(
                select(SemanticEmbeddingRow).where(
                    SemanticEmbeddingRow.subject_type == subject_type,
                    SemanticEmbeddingRow.subject_id == subject_id,
                    SemanticEmbeddingRow.modality == modality,
                    SemanticEmbeddingRow.provider == descriptor.provider,
                    SemanticEmbeddingRow.model == descriptor.model,
                    SemanticEmbeddingRow.revision == descriptor.revision,
                )
            )
            if row is None or row.content_hash != content_hash:
                return None
            vector = tuple(float(value) for value in row.vector_json)
            if len(vector) != row.dimensions:
                return None
            return vector

    def upsert(
        self,
        *,
        subject_type: str,
        subject_id: str,
        modality: str,
        descriptor: EmbeddingModelDescriptor,
        content_hash: str,
        vector: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        now = datetime.now(UTC)
        values = [float(value) for value in vector]
        with self._database.session() as session:
            row = session.scalar(
                select(SemanticEmbeddingRow).where(
                    SemanticEmbeddingRow.subject_type == subject_type,
                    SemanticEmbeddingRow.subject_id == subject_id,
                    SemanticEmbeddingRow.modality == modality,
                    SemanticEmbeddingRow.provider == descriptor.provider,
                    SemanticEmbeddingRow.model == descriptor.model,
                    SemanticEmbeddingRow.revision == descriptor.revision,
                )
            )
            if row is None:
                session.add(
                    SemanticEmbeddingRow(
                        id=uuid4().hex,
                        subject_type=subject_type,
                        subject_id=subject_id,
                        modality=modality,
                        content_hash=content_hash,
                        provider=descriptor.provider,
                        model=descriptor.model,
                        revision=descriptor.revision,
                        dimensions=len(values),
                        vector_json=values,
                        metadata_json=dict(metadata or {}),
                        created_at=now,
                        updated_at=now,
                    )
                )
                return
            row.content_hash = content_hash
            row.dimensions = len(values)
            row.vector_json = values
            row.metadata_json = dict(metadata or {})
            row.updated_at = now

    def count(self) -> int:
        with self._database.session() as session:
            value = session.scalar(select(func.count()).select_from(SemanticEmbeddingRow))
            return int(value or 0)


class SemanticIndex:
    """Persistent cache keyed by subject, source hash and embedding model version."""

    def __init__(
        self,
        repository: SemanticEmbeddingRepository,
        provider: DescribedEmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._provider = provider

    @property
    def descriptor(self) -> EmbeddingModelDescriptor:
        return self._provider.descriptor

    @property
    def repository(self) -> SemanticEmbeddingRepository:
        return self._repository

    async def embed_text(
        self,
        subject_type: str,
        subject_id: str,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[float, ...]:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        descriptor = self._provider.descriptor
        cached = self._repository.get(
            subject_type=subject_type,
            subject_id=subject_id,
            modality="text",
            descriptor=descriptor,
            content_hash=content_hash,
        )
        if cached is not None:
            return cached
        vector = _normalize(await self._provider.embed_text(text))
        descriptor = self._provider.descriptor
        self._repository.upsert(
            subject_type=subject_type,
            subject_id=subject_id,
            modality="text",
            descriptor=descriptor,
            content_hash=content_hash,
            vector=vector,
            metadata=metadata,
        )
        return vector

    async def embed_image(
        self,
        subject_type: str,
        subject_id: str,
        path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[float, ...]:
        content_hash = await asyncio.to_thread(_file_sha256, path)
        descriptor = self._provider.descriptor
        cached = self._repository.get(
            subject_type=subject_type,
            subject_id=subject_id,
            modality="image",
            descriptor=descriptor,
            content_hash=content_hash,
        )
        if cached is not None:
            return cached
        vector = _normalize(await self._provider.embed_image(path))
        descriptor = self._provider.descriptor
        merged_metadata = {"path": str(path), **dict(metadata or {})}
        self._repository.upsert(
            subject_type=subject_type,
            subject_id=subject_id,
            modality="image",
            descriptor=descriptor,
            content_hash=content_hash,
            vector=vector,
            metadata=merged_metadata,
        )
        return vector
