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
    removed: tuple[LoRAProfile, ...] = ()
    invalidated: tuple[LoRAProfile, ...] = ()


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


def _infer_lora_type(
    path: Path,
    metadata: dict[str, Any],
    targets: tuple[str, ...],
    existing: LoRAProfile | None,
) -> str:
    explicit_keys = (
        "artifex.lora_type",
        "artifex_lora_type",
        "modelspec.lora_type",
    )
    for key in explicit_keys:
        raw = metadata.get(key)
        if isinstance(raw, str) and raw.strip().casefold() in {
            "character",
            "outfit",
            "style",
            "utility",
        }:
            return raw.strip().casefold()

    if existing is not None and existing.lora_type in {
        "character",
        "outfit",
        "style",
        "utility",
    }:
        return existing.lora_type

    text = _metadata_text(path, metadata).casefold()
    if "outfit" in text or "costume" in text:
        return "outfit"
    if "style" in text:
        return "style"
    if any(
        marker in text
        for marker in (
            "slider",
            "detail",
            "enhancer",
            "lighting",
            "hands",
            "anatomy",
            "utility",
        )
    ):
        return "utility"
    return "character" if targets else "other"


def _under_root(path: Path, roots: tuple[Path, ...]) -> bool:
    absolute = path.expanduser().resolve(strict=False)
    for root in roots:
        try:
            absolute.relative_to(root)
            return True
        except ValueError:
            continue
    return False


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
        removed: list[LoRAProfile] = []
        invalidated: list[LoRAProfile] = []

        normalized_roots = tuple(
            root.expanduser().resolve(strict=False)
            for root in roots
        )
        paths: list[Path] = []
        for root in normalized_roots:
            if not root.exists():
                continue
            paths.extend(
                path.resolve(strict=False)
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in self._extensions
            )
        paths = sorted(
            dict.fromkeys(paths),
            key=lambda item: item.as_posix().casefold(),
        )
        seen_paths = {str(path).casefold() for path in paths}

        for path in paths:
            try:
                profile, changed = self._inspect(path)
            except (OSError, SafeTensorMetadataError, ValueError) as exc:
                failed.append((path, str(exc)))
                existing = self._registry.find_by_path(path)
                if existing is not None:
                    invalidated.append(
                        self._registry.invalidate(
                            existing.id,
                            target=LoRAState.FAILED,
                            reason=f"asset inspection failed: {exc}",
                        )
                    )
                continue
            if changed:
                discovered.append(profile)
            else:
                unchanged.append(profile)

        for profile in self._registry.list():
            if profile.source != "filesystem-discovery":
                continue
            if not _under_root(profile.path, normalized_roots):
                continue
            normalized = str(
                profile.path.expanduser().resolve(strict=False)
            ).casefold()
            if normalized in seen_paths:
                continue
            if (
                profile.state is LoRAState.DISABLED
                and profile.missing_since is not None
            ):
                continue
            removed.append(
                self._registry.invalidate(
                    profile.id,
                    target=LoRAState.DISABLED,
                    reason=f"LoRA file disappeared: {profile.path}",
                    missing=True,
                )
            )

        return DiscoveryResult(
            discovered=tuple(discovered),
            unchanged=tuple(unchanged),
            failed=tuple(failed),
            removed=tuple(removed),
            invalidated=tuple(invalidated),
        )

    def _inspect(self, path: Path) -> tuple[LoRAProfile, bool]:
        absolute = path.expanduser().resolve(strict=True)
        checksum = sha256_file(absolute)
        metadata = read_safetensors_metadata(
            absolute,
            max_header_bytes=self._max_header_bytes,
        )
        existing = self._registry.find_by_path(absolute)
        if (
            existing is not None
            and existing.checksum == checksum
            and existing.missing_since is None
            and existing.metadata.get("asset_status") != "invalidated"
        ):
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

        metadata = dict(metadata)
        metadata["asset_status"] = "present"
        if existing is not None:
            metadata["previous_checksum"] = existing.checksum

        profile = LoRAProfile(
            id=lora_id,
            path=absolute,
            state=LoRAState.DISCOVERED,
            lora_type=_infer_lora_type(
                absolute,
                metadata,
                targets,
                existing,
            ),
            target_character_ids=targets or (
                existing.target_character_ids if existing is not None else ()
            ),
            model_families=model_families or (
                existing.model_families if existing is not None else ()
            ),
            trigger_tags=trigger_tags or (
                existing.trigger_tags if existing is not None else ()
            ),
            recommended_weight=(
                existing.recommended_weight if existing else 1.0
            ),
            readiness=0.0,
            checksum=checksum,
            incompatible_lora_ids=(
                existing.incompatible_lora_ids if existing else ()
            ),
            source=source,
            missing_since=None,
            metadata=metadata,
        )
        return self._registry.upsert(profile), True
