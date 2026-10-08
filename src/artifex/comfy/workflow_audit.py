from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from artifex.comfy import ComfyUIClient, WorkflowTemplateRegistry
from artifex.comfy.models import WorkflowAssetRequirement
from artifex.config.models import ArtifexSettings
from artifex.deployment import _request


class MissingWorkflowAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    node_class: str
    input_name: str
    requested: str
    available: tuple[str, ...]


class WorkflowAuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str
    source_sha256: str
    ready: bool
    required_node_count: int
    required_asset_count: int
    missing_node_types: tuple[str, ...]
    missing_assets: tuple[MissingWorkflowAsset, ...]
    unverifiable_assets: tuple[MissingWorkflowAsset, ...]


class WorkflowAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    comfyui_url: str
    custom_node_folders: tuple[str, ...]
    entries: tuple[WorkflowAuditEntry, ...]
    guidance: tuple[str, ...]


def _asset_check(
    asset: WorkflowAssetRequirement,
    object_info: Mapping[str, Any],
) -> tuple[bool | None, MissingWorkflowAsset | None]:
    """True=confirmed, False=missing, None=not verifiable from ComfyUI schema."""
    node = object_info.get(asset.node_class)
    if not isinstance(node, Mapping):
        return None, MissingWorkflowAsset(
            label=asset.label,
            node_class=asset.node_class,
            input_name=asset.input_name,
            requested=asset.value,
            available=(),
        )
    options = ComfyUIClient._input_choices(node, asset.input_name)
    if not options:
        return None, MissingWorkflowAsset(
            label=asset.label,
            node_class=asset.node_class,
            input_name=asset.input_name,
            requested=asset.value,
            available=(),
        )
    if any(ComfyUIClient._asset_name_matches(asset.value, option) for option in options):
        return True, None
    return False, MissingWorkflowAsset(
        label=asset.label,
        node_class=asset.node_class,
        input_name=asset.input_name,
        requested=asset.value,
        available=options[:30],
    )


def _node_folders(root: Path | None) -> tuple[str, ...]:
    if root is None:
        return ()
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Custom node directory must be a directory: {root}")
    paths = sorted(
        (
            path.name
            for path in root.iterdir()
            if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
        ),
        key=str.casefold,
    )
    return tuple(paths[:256])


async def audit_workflows(
    settings: ArtifexSettings,
    *,
    client: ComfyUIClient | None = None,
    custom_nodes_root: Path | None = None,
) -> WorkflowAudit:
    """Strict read-only audit of real production+repair graphs against /object_info.

    If no input-choice enumeration is exposed, asset presence is UNKNOWN, not
    silently interpreted as available. Never guess a repository from a class.
    """
    primary = settings.render_nodes.primary_node()
    url = primary[1].base_url if primary is not None else settings.comfyui.base_url
    owned_client = client is None
    comfy = client or ComfyUIClient(
        settings.comfyui.model_copy(update={"base_url": url})
    )
    try:
        raw = await comfy.object_info()
    finally:
        if owned_client:
            await comfy.aclose()
    if not isinstance(raw, dict):
        raise ValueError("ComfyUI /object_info must be a JSON object")

    registry = WorkflowTemplateRegistry.with_packaged_templates()
    entries: list[WorkflowAuditEntry] = []
    for template_id in dict.fromkeys(
        (settings.comfyui.default_template, settings.production.repair_workflow_template)
    ):
        template = registry.require(template_id)
        req = _request(
            settings,
            output_prefix="ARTIFEX/read-only-audit",
            width=settings.production.width,
            height=settings.production.height,
        )
        requirements = template.requirements(req)
        missing_nodes = tuple(
            sorted(name for name in requirements.node_types if name not in raw)
        )
        missing_assets: list[MissingWorkflowAsset] = []
        unknown: list[MissingWorkflowAsset] = []
        for asset in requirements.assets:
            state, finding = _asset_check(asset, raw)
            if state is False and finding is not None:
                missing_assets.append(finding)
            elif state is None and finding is not None:
                unknown.append(finding)
        entries.append(
            WorkflowAuditEntry(
                template_id=template_id,
                source_sha256=requirements.source_sha256,
                ready=not missing_nodes and not missing_assets and not unknown,
                required_node_count=len(requirements.node_types),
                required_asset_count=len(requirements.assets),
                missing_node_types=missing_nodes,
                missing_assets=tuple(missing_assets),
                unverifiable_assets=tuple(unknown),
            )
        )
    folders = _node_folders(custom_nodes_root)
    guidance: list[str] = []
    if any(entry.missing_node_types for entry in entries):
        guidance.append(
            "Required ComfyUI node class types are absent from the running server. "
            "Install the correct compatible custom-node repositories separately; "
            "folder names are not a reliable mapping from class types."
        )
    if any(entry.missing_assets for entry in entries):
        guidance.append(
            "Some workflow model choices are absent from ComfyUI loader choices. "
            "Verify configured checkpoint, refiner, VAE, upscaler, detector and LoRAs."
        )
    if any(entry.unverifiable_assets for entry in entries):
        guidance.append(
            "Some model inputs do not expose a choice list through /object_info; "
            "their availability is unverified, not a passing check."
        )
    return WorkflowAudit(
        ready=all(entry.ready for entry in entries),
        comfyui_url=url,
        custom_node_folders=folders,
        entries=tuple(entries),
        guidance=tuple(guidance),
    )
