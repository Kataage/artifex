from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifex.characters import CharacterRegistry
from artifex.domain import LoRAProfile, LoRAState
from artifex.loras.registry import LoRARegistry
from artifex.loras.safetensors import (
    SafeTensorMetadataError,
    read_safetensors_metadata,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    discovered: tuple[LoRAProfile, ...]
    unchanged: tuple[LoRAProfile, ...]
    failed: tuple[tuple[Path, str], ...]


def _stable_auto_id(path: Path) -> str:
    normalized = str(path.expanduser().resolve(strict=False)).casefold()
    return "auto_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _metadata_text(path: Path, metadata: dict[str, Any]) -> str:
    return " ".join(
        (
            path.stem,
            *[str(key) for key in metadata],
            *[str(value) for value in metadata.values()],
        )
    )


def _infer_model_families(path: Path, metadata: dict[str, Any]) -> tuple[str, ...]:
    text = _metadata_text(path, metadata).casefold()
    result: list[str] = []
    if "illustrious" in text or "ilxl" in text:
        result.append("ilxl")
    if "pony" in text:
        result.append("pony")
    if "sdxl" in text or "stable-diffusion-xl" in text:
        result.append("sdxl")
    if "sd1.5" in text or "sd15" in text or "stable-diffusion-v1-5" in text:
        result.append("sd15")
    return tuple(dict.fromkeys(result))


def _infer_trigger_tags(metadata: dict[str, Any]) -> tuple[str, ...]:
    keys = (
        "modelspec.trigger_phrase",
        "modelspec.trigger_phrases",
        "ss_trigger_words",
        "trigger_words",
    )
    values: list[str] = []
    for key in keys:
        raw = metadata.get(key)
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            values.extend(str(item).strip() for item in parsed if str(item).strip())
        else:
            values.extend(
                item.strip()
                for item in candidate.replace("\n", ",").split(",")
                if item.strip()
            )
    return tuple(dict.fromkeys(values))


class LoRADiscovery:
    def __init__(
        self,
        registry: LoRARegistry,
        characters: CharacterRegistry,
        *,
        extensions: Iterable[str] = (".safetensors",),
        max_header_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self._registry = registry
        self._characters = characters
        self._extensions = {extension.casefold() for extension in extensions}
        self._max_header_bytes = max_header_bytes

    def scan(self, roots: Iterable[Path]) -> DiscoveryResult:
        discovered: list[LoRAProfile] = []
        unchanged: list[LoRAProfile] = []
        failed: list[tuple[Path, str]] = []

        paths: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in self._extensions
            )

        for path in sorted(paths, key=lambda item: item.as_posix().casefold()):
            try:
                profile, changed = self._inspect(path)
            except (OSError, SafeTensorMetadataError, ValueError) as exc:
                failed.append((path, str(exc)))
                continue
            if changed:
                discovered.append(profile)
            else:
                unchanged.append(profile)

        return DiscoveryResult(
            discovered=tuple(discovered),
            unchanged=tuple(unchanged),
            failed=tuple(failed),
        )

    def _inspect(self, path: Path) -> tuple[LoRAProfile, bool]:
        absolute = path.expanduser().resolve(strict=True)
        checksum = sha256_file(absolute)
        metadata = read_safetensors_metadata(
            absolute,
            max_header_bytes=self._max_header_bytes,
        )
        existing = self._registry.find_by_path(absolute)
        if existing is not None and existing.checksum == checksum:
            return existing, False

        text = _metadata_text(absolute, metadata)
        targets = self._characters.match_text(text)
        model_families = _infer_model_families(absolute, metadata)
        trigger_tags = _infer_trigger_tags(metadata)

        if existing is None:
            lora_id = _stable_auto_id(absolute)
            source = "filesystem-discovery"
        else:
            lora_id = existing.id
            source = existing.source or "filesystem-discovery"

        profile = LoRAProfile(
            id=lora_id,
            path=absolute,
            state=LoRAState.DISCOVERED,
            lora_type="character" if targets else "other",
            target_character_ids=targets,
            model_families=model_families,
            trigger_tags=trigger_tags,
            recommended_weight=existing.recommended_weight if existing else 1.0,
            readiness=0.0,
            checksum=checksum,
            incompatible_lora_ids=existing.incompatible_lora_ids if existing else (),
            source=source,
            metadata=metadata,
        )
        return self._registry.upsert(profile), True
