from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
_SHA256 = re.compile(r"[a-fA-F0-9]{64}")
_MAX_BYTES = 1024 * 1024


class ApprovedModelSource(BaseModel):
    """Explicit operator-selected correspondence; not third-party trust attestation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["checkpoint", "refiner", "vae", "upscale_model", "lora", "static_lora"]
    model_name: str
    repository: str
    file_path: str | None = None
    expected_sha256: str | None = None
    rationale: str = Field(min_length=8, max_length=500)

    @model_validator(mode="after")
    def validate_source(self) -> ApprovedModelSource:
        if not _COMPONENT.fullmatch(self.model_name) or self.model_name in {".", ".."}:
            raise ValueError("Model source name must be a simple basename")
        if not _REPO.fullmatch(self.repository):
            raise ValueError("Model repository must be a literal Hugging Face owner/repository")
        if any(part in {".", ".."} for part in self.repository.split("/")):
            raise ValueError("Model repository cannot contain path traversal")
        if self.file_path is not None:
            components = self.file_path.split("/")
            if (
                not 1 <= len(components) <= 8
                or not all(_COMPONENT.fullmatch(c) and c not in {".", ".."} for c in components)
                or components[-1] != self.model_name
            ):
                raise ValueError("Exact source file path must safely end in model name")
        if self.expected_sha256 is not None and not _SHA256.fullmatch(
            self.expected_sha256
        ):
            raise ValueError("Expected model hash must be 64 hexadecimal characters")
        return self


class ModelSourceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    sources: tuple[ApprovedModelSource, ...] = ()

    @model_validator(mode="after")
    def validate_unique_roles(self) -> ModelSourceRegistry:
        keys = [(s.role.casefold(), s.model_name.casefold()) for s in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Only one explicitly approved publisher is allowed per model role/name"
            )
        if len(self.sources) > 512:
            raise ValueError("Model source registry exceeds 512 bindings")
        return self


def read_model_source_registry(
    path: Path, *, if_missing_empty: bool = False,
) -> ModelSourceRegistry:
    if path.is_symlink():
        raise ValueError("Refusing to read symlinked model source registry")
    if not path.exists():
        if if_missing_empty:
            return ModelSourceRegistry()
        raise FileNotFoundError(f"Model source registry does not exist: {path}")
    if not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError("Model source registry is not a regular JSON file <= 1 MiB")
    return ModelSourceRegistry.model_validate_json(path.read_bytes())


def save_model_source_registry(path: Path, registry: ModelSourceRegistry) -> Path:
    """Atomic writes in selected directory; never follows or overwrites symlinks."""
    raw = (registry.model_dump_json(indent=2) + "\n").encode("utf-8")
    if len(raw) > _MAX_BYTES:
        raise ValueError("Model source registry exceeds the maximum file size")
    if path.exists() and not path.is_file():
        raise ValueError("Refusing to overwrite non-file model source registry")
    if path.is_symlink():
        raise ValueError("Refusing to overwrite symlinked model source registry")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".artifex-source-", suffix=".tmp",
            dir=path.parent, delete=False
        ) as handle:
            temp = Path(handle.name)
            handle.write(raw)
        if path.is_symlink():
            raise ValueError("Refusing to replace symlinked model source registry")
        os.replace(temp, path)
        return path
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def register_model_source(
    registry: ModelSourceRegistry, source: ApprovedModelSource,
) -> ModelSourceRegistry:
    if any(
        s.role == source.role and s.model_name.casefold() == source.model_name.casefold()
        for s in registry.sources
    ):
        raise FileExistsError("An approved source already exists for this model role/name")
    return ModelSourceRegistry(
        sources=tuple(
            sorted(
                (*registry.sources, source),
                key=lambda s: (s.role, s.model_name.casefold()),
            )
        )
    )


def unregister_model_source(
    registry: ModelSourceRegistry, *, role: str, model_name: str, repository: str,
) -> ModelSourceRegistry:
    matching = [
        s for s in registry.sources
        if s.role == role and s.model_name.casefold() == model_name.casefold()
        and s.repository.casefold() == repository.casefold()
    ]
    if len(matching) != 1:
        raise ValueError("No matching role/name/repository registration to remove")
    return ModelSourceRegistry(
        sources=tuple(s for s in registry.sources if s != matching[0])
    )
