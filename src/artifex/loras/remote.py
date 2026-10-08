from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from artifex.characters import CharacterRegistry
from artifex.config.models import RenderNodesConfig
from artifex.domain import LoRAProfile, LoRAState
from artifex.loras.discovery import (
    DiscoveryResult,
    LoRADiscovery,
    _infer_lora_type,
    _infer_model_families,
    _infer_trigger_tags,
    _metadata_text,
)
from artifex.loras.registry import LoRARegistry

if TYPE_CHECKING:
    from artifex.render_node.models import RenderNodeAttestation



def _remote_id(node_id: str, relative_path: str) -> str:
    key = f"{node_id}:{relative_path.casefold()}".encode()
    return "remote_" + hashlib.sha256(key).hexdigest()[:20]


def _merge_results(results: list[DiscoveryResult]) -> DiscoveryResult:
    return DiscoveryResult(
        discovered=tuple(item for result in results for item in result.discovered),
        unchanged=tuple(item for result in results for item in result.unchanged),
        failed=tuple(item for result in results for item in result.failed),
        removed=tuple(item for result in results for item in result.removed),
        invalidated=tuple(item for result in results for item in result.invalidated),
    )


class RemoteLoRADiscovery:
    def __init__(
        self,
        registry: LoRARegistry,
        characters: CharacterRegistry,
    ) -> None:
        self._registry = registry
        self._characters = characters

    def sync(self, attestation: RenderNodeAttestation) -> DiscoveryResult:
        source = f"render-node:{attestation.node_id}"
        discovered: list[LoRAProfile] = []
        unchanged: list[LoRAProfile] = []
        removed: list[LoRAProfile] = []
        failed: list[tuple[Path, str]] = []
        seen_ids: set[str] = set()

        for raw_error in attestation.inventory_errors:
            failed.append(
                (
                    Path(raw_error.get("path", f"render-node:{attestation.node_id}")),
                    raw_error.get("error", "remote inventory error"),
                )
            )

        for item in attestation.loras:
            lora_id = _remote_id(attestation.node_id, item.relative_path)
            seen_ids.add(lora_id)
            existing = self._registry.get(lora_id)
            if (
                existing is not None
                and existing.checksum == item.sha256
                and existing.missing_since is None
                and existing.metadata.get("asset_status") == "present"
            ):
                unchanged.append(existing)
                continue

            semantic_path = Path(item.relative_path)
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "asset_status": "present",
                    "asset_name": item.relative_path,
                    "remote_path": item.path,
                    "remote_node_id": attestation.node_id,
                    "remote_attested_at": attestation.created_at.isoformat(),
                }
            )
            if existing is not None:
                metadata["previous_checksum"] = existing.checksum

            text = _metadata_text(semantic_path, metadata)
            targets = self._characters.match_text(text)
            model_families = _infer_model_families(semantic_path, metadata)
            trigger_tags = _infer_trigger_tags(metadata)
            virtual_path = (
                Path(".artifex-remote")
                / attestation.node_id
                / Path(item.relative_path.replace("\\", "/"))
            )
            profile = LoRAProfile(
                id=lora_id,
                path=virtual_path,
                state=LoRAState.DISCOVERED,
                lora_type=_infer_lora_type(
                    semantic_path,
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
                    existing.recommended_weight if existing is not None else 1.0
                ),
                readiness=0.0,
                checksum=item.sha256,
                incompatible_lora_ids=(
                    existing.incompatible_lora_ids if existing is not None else ()
                ),
                source=source,
                missing_since=None,
                metadata=metadata,
            )
            discovered.append(self._registry.upsert(profile))

        # A partially failed inventory is not proof that absent LoRAs were deleted.
        # Preserve last-known-good entries until a complete inventory succeeds.
        if attestation.inventory_errors:
            return DiscoveryResult(
                discovered=tuple(discovered),
                unchanged=tuple(unchanged),
                failed=tuple(failed),
                removed=(),
            )

        for profile in self._registry.list():
            if profile.source != source or profile.id in seen_ids:
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
                    reason=(
                        f"LoRA disappeared from render node "
                        f"{attestation.node_id}: {profile.metadata.get('asset_name', profile.id)}"
                    ),
                    missing=True,
                )
            )

        return DiscoveryResult(
            discovered=tuple(discovered),
            unchanged=tuple(unchanged),
            failed=tuple(failed),
            removed=tuple(removed),
        )


class RenderAwareLoRADiscovery(LoRADiscovery):
    def __init__(
        self,
        registry: LoRARegistry,
        characters: CharacterRegistry,
        render_nodes: RenderNodesConfig,
        *,
        extensions: tuple[str, ...] = (".safetensors",),
        max_header_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        super().__init__(
            registry,
            characters,
            extensions=extensions,
            max_header_bytes=max_header_bytes,
        )
        self._render_nodes = render_nodes
        self._remote = RemoteLoRADiscovery(registry, characters)

    def scan(self, roots: Iterable[Path]) -> DiscoveryResult:
        # Deferred to avoid render_node -> attestation -> loras -> render_node
        # circular imports during isolated renderer startup on Windows.
        from artifex.render_node.client import fetch_render_attestation

        results = [super().scan(roots)]
        for node_id, config in sorted(self._render_nodes.nodes.items()):
            if not config.enabled or not config.attestation_url:
                continue
            try:
                attestation = fetch_render_attestation(node_id, config)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                results.append(
                    DiscoveryResult(
                        discovered=(),
                        unchanged=(),
                        failed=((Path(f"render-node-{node_id}"), str(exc)),),
                    )
                )
                continue
            results.append(self._remote.sync(attestation))
        return _merge_results(results)
