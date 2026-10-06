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
from artifex.db.models import (
    ConceptRow,
    GenerationAttemptRow,
    PackRow,
    SceneRow,
    SemanticEmbeddingRow,
)
from artifex.domain import PackState


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

    @staticmethod
    def image_subject_id(path: Path) -> str:
        return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()

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


@dataclass(frozen=True, slots=True)
class SemanticBackfillReport:
    concepts_scanned: int
    concepts_indexed: int
    images_scanned: int
    images_indexed: int
    missing_image_paths: int


def _concept_payload_text(payload: Mapping[str, Any]) -> str | None:
    if payload.get("selected") is not True:
        return None
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        return None

    character_ids = candidate.get("character_ids", ())
    if isinstance(character_ids, list | tuple):
        characters = ",".join(str(value) for value in character_ids)
    else:
        characters = str(character_ids)
    fields = (
        characters,
        str(candidate.get("format", "")),
        str(candidate.get("theme", "")),
        str(candidate.get("setting", "")),
        str(candidate.get("mood", "")),
        str(candidate.get("visual_hook", "")),
        str(candidate.get("progression", "")),
    )
    text = " | ".join(value for value in fields if value)
    return text or None


def _attempt_output_paths(attempt: GenerationAttemptRow) -> tuple[Path, ...]:
    raw = attempt.provenance_json.get("output_paths", ())
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(Path(value) for value in raw if isinstance(value, str) and value)


class SemanticArchiveIndexer:
    """Backfill every selected Concept and finalized selected image into SQLite."""

    def __init__(self, database: Database, index: SemanticIndex) -> None:
        self._database = database
        self._index = index

    async def backfill(self) -> SemanticBackfillReport:
        with self._database.session() as session:
            concepts = tuple(
                session.scalars(
                    select(ConceptRow).order_by(
                        ConceptRow.created_at.asc(),
                        ConceptRow.id.asc(),
                    )
                ).all()
            )
            attempts = tuple(
                session.scalars(
                    select(GenerationAttemptRow)
                    .join(
                        SceneRow,
                        SceneRow.selected_attempt_id == GenerationAttemptRow.id,
                    )
                    .join(PackRow, SceneRow.pack_id == PackRow.id)
                    .where(PackRow.state == PackState.FINALIZED.value)
                    .order_by(
                        GenerationAttemptRow.created_at.asc(),
                        GenerationAttemptRow.id.asc(),
                    )
                ).all()
            )
            for row in (*concepts, *attempts):
                session.expunge(row)

        concepts_indexed = 0
        for row in concepts:
            text = _concept_payload_text(row.payload_json)
            if text is None:
                continue
            candidate = row.payload_json.get("candidate")
            character_ids = (
                candidate.get("character_ids", ())
                if isinstance(candidate, dict)
                else ()
            )
            await self._index.embed_text(
                "concept",
                row.id,
                text,
                metadata={"character_ids": list(character_ids)},
            )
            concepts_indexed += 1

        images_scanned = 0
        images_indexed = 0
        missing_image_paths = 0
        for attempt in attempts:
            for path in _attempt_output_paths(attempt):
                images_scanned += 1
                if not path.is_file():
                    missing_image_paths += 1
                    continue
                await self._index.embed_image(
                    "image",
                    SemanticIndex.image_subject_id(path),
                    path,
                    metadata={
                        "path": str(path),
                        "attempt_id": attempt.id,
                        "scene_id": attempt.scene_id,
                        "historical_finalized": True,
                    },
                )
                images_indexed += 1

        return SemanticBackfillReport(
            concepts_scanned=len(concepts),
            concepts_indexed=concepts_indexed,
            images_scanned=images_scanned,
            images_indexed=images_indexed,
            missing_image_paths=missing_image_paths,
        )
