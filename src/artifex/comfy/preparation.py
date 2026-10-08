from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.comfy.dependency_installer import (
    DependencyInstallResult,
    _safe_install_root,
    install_manifest,
)
from artifex.comfy.dependency_resolver import resolve_missing_dependencies
from artifex.comfy.model_drafter import draft_hf_models
from artifex.comfy.model_sources import read_model_source_registry
from artifex.comfy.workflow_audit import audit_workflows
from artifex.config.models import ArtifexSettings


class RendererPreparationReport(BaseModel):
    """Never equate an installation plan with a verified running renderer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: str
    audit_ready_before: bool
    models_requested: tuple[str, ...]
    approved_sources: int
    pinned_models: tuple[dict[str, Any], ...]
    unresolved_models: tuple[dict[str, Any], ...]
    missing_node_types: tuple[str, ...]
    node_candidates: tuple[dict[str, Any], ...]
    node_discovery_error: str | None
    installed: tuple[dict[str, Any], ...]
    needs_comfy_restart_and_reaudit: bool
    ready: bool
    manual_action_required: bool
    production_qualified: bool = False


async def prepare_renderer(
    settings: ArtifexSettings,
    *,
    sources: Path,
    comfy_root: Path | None = None,
    apply: bool = False,
    accept_licenses: bool = False,
    discover_nodes: bool = True,
) -> RendererPreparationReport:
    """Audit live ComfyUI and prepare *only explicitly bound* pinned model weights.

    Reading a source registry is not approval of arbitrary repositories: only
    the exact role/filename bindings already reviewed by the operator can
    produce a verified model manifest. Third-party node indexes only produce
    suggestions. No code is installed or executed by this entry point.
    """
    if apply:
        if not accept_licenses:
            raise ValueError("--apply requires --accept-licenses for reviewed model terms")
        if comfy_root is None:
            raise ValueError("--apply requires --comfy-root for an Artifex isolated ComfyUI")
        # Fail before contacting Hub/Manager when installation destination is
        # untrusted, e.g. an existing personal ComfyUI tree or symlink.
        await asyncio.to_thread(_safe_install_root, comfy_root)
    elif accept_licenses:
        raise ValueError("--accept-licenses requires --apply")

    saved = read_model_source_registry(sources, if_missing_empty=True)
    audit = await audit_workflows(settings)
    requested = tuple(sorted({
        f"{asset.label}:{asset.requested}"
        for entry in audit.entries for asset in entry.missing_assets
    }))
    needed_nodes = tuple(sorted({
        node for entry in audit.entries for node in entry.missing_node_types
    }))

    # No model source guessing. No Hub calls when there is nothing missing or
    # when none of the registered exact bindings match a missing loader choice.
    draft = await asyncio.to_thread(
        draft_hf_models, audit, repositories=(), bindings=saved.sources
    )

    candidates: tuple[dict[str, Any], ...] = ()
    node_error: str | None = None
    if needed_nodes and discover_nodes:
        try:
            resolution = await asyncio.to_thread(resolve_missing_dependencies, audit)
            candidates = tuple(c.model_dump(mode="json") for c in resolution.candidates)
        except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
            # A suggestion registry is not authoritative. Report its outage
            # without obstructing approved, independently pinned model assets.
            node_error = str(exc)
    elif needed_nodes:
        node_error = "Node discovery disabled; missing code requires separate review"

    installed: DependencyInstallResult | None = None
    if apply and draft.manifest.artifacts:
        assert comfy_root is not None
        # The existing installer enforces pinned source/SHA, size, approved
        # destination and no overwrite. Never grant third-party code consent.
        installed = await asyncio.to_thread(
            install_manifest,
            draft.manifest,
            comfy_root,
            accept_licenses=True,
            allow_custom_code=False,
        )

    installed_rows: tuple[dict[str, Any], ...] = ()
    if installed is not None:
        installed_rows = tuple(
            item.model_dump(mode="json") for item in installed.installed
        )
    pinned = tuple(item.model_dump(mode="json") for item in draft.manifest.artifacts)
    unresolved = tuple(item.model_dump(mode="json") for item in draft.unresolved)
    manual = bool(unresolved or needed_nodes)
    changed = bool(installed_rows)
    return RendererPreparationReport(
        mode="apply" if apply else "preview",
        audit_ready_before=audit.ready,
        models_requested=requested,
        approved_sources=len(saved.sources),
        pinned_models=pinned,
        unresolved_models=unresolved,
        missing_node_types=needed_nodes,
        node_candidates=candidates,
        node_discovery_error=node_error,
        installed=installed_rows,
        # ComfyUI may cache input model lists; even after a download its
        # previously captured /object_info is NOT reinterpreted as a PASS.
        needs_comfy_restart_and_reaudit=changed,
        ready=audit.ready and not changed and not manual,
        manual_action_required=manual,
    )
