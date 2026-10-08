from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.comfy.dependency_installer import (
    DependencyInstallResult,
    DependencyManifest,
    PinnedDependency,
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
    existing_verified: tuple[dict[str, Any], ...] = ()
    download_required: tuple[dict[str, Any], ...] = ()
    next_actions: tuple[str, ...] = ()
    needs_comfy_restart_and_reaudit: bool
    ready: bool
    manual_action_required: bool
    production_qualified: bool = False


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _existing_models(
    root: Path,
    manifest: DependencyManifest,
) -> tuple[DependencyManifest, tuple[dict[str, Any], ...]]:
    """Reuse only byte-verified models inside Artifex-owned isolated ComfyUI.

    A mismatched, nonregular or symlinked destination is a hard failure,
    never a reason to overwrite or download again. Preflight all targets
    before downloading even a single missing model.
    """
    pending: list[PinnedDependency] = []
    verified: list[dict[str, Any]] = []
    model_root = root / "models"
    if model_root.is_symlink():
        raise ValueError("Refusing symlinked ComfyUI model root")
    for item in manifest.artifacts:
        if item.kind != "model" or item.model_folder is None:
            raise ValueError("Renderer preparation never installs arbitrary custom code")
        folder = model_root / item.model_folder
        target = folder / item.name
        if folder.is_symlink() or target.is_symlink():
            raise ValueError(f"Refusing symlinked ComfyUI model destination: {target}")
        if not target.exists():
            pending.append(item)
            continue
        if not target.is_file():
            raise ValueError(f"Existing model destination is not a regular file: {target}")
        if target.stat().st_size != item.size_bytes:
            raise ValueError(f"Existing model size differs from pinned source: {target}")
        if _hash_file(target) != item.sha256.lower():
            raise ValueError(f"Existing model SHA-256 differs from pinned source: {target}")
        verified.append({
            "id": item.id,
            "destination": str(target),
            "sha256": item.sha256.lower(),
            "size_bytes": item.size_bytes,
        })
    return DependencyManifest(artifacts=tuple(pending)), tuple(verified)


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
    elif accept_licenses:
        raise ValueError("--accept-licenses requires --apply")
    verified_root: Path | None = None
    if comfy_root is not None:
        if comfy_root.expanduser().is_symlink():
            raise ValueError("Refusing symlinked ComfyUI installation root")
        # An unsafe root aborts before contacting ComfyUI, Hub or Manager.
        verified_root = await asyncio.to_thread(_safe_install_root, comfy_root)

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

    pending = draft.manifest
    existing_verified: tuple[dict[str, Any], ...] = ()
    if verified_root is not None and pending.artifacts:
        pending, existing_verified = await asyncio.to_thread(
            _existing_models, verified_root, pending
        )

    installed: DependencyInstallResult | None = None
    if apply and pending.artifacts:
        assert comfy_root is not None
        # The original installer still rechecks races and refuses overwrites.
        installed = await asyncio.to_thread(
            install_manifest,
            pending,
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
    download_required = tuple(item.model_dump(mode="json") for item in pending.artifacts)
    unresolved = tuple(item.model_dump(mode="json") for item in draft.unresolved)
    changed = bool(installed_rows)
    refresh_needed = changed or bool(existing_verified)
    actions: list[str] = []
    if unresolved:
        actions.append("review_and_register_exact_model_publisher_bindings")
    if needed_nodes:
        actions.append("review_and_install_missing_custom_nodes_separately")
    if pending.artifacts and not apply:
        actions.append("review_model_licenses_then_apply_pinned_download")
    if refresh_needed:
        actions.append("refresh_or_restart_comfyui_then_repeat_live_workflow_audit")
    if not audit.ready and not (
        unresolved or needed_nodes or pending.artifacts or refresh_needed
    ):
        actions.append("investigate_unverified_workflow_inputs")
    if audit.ready:
        actions.append("run_two_pc_deployment_verify")
    manual = bool(actions and actions != ["run_two_pc_deployment_verify"])
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
        existing_verified=existing_verified,
        download_required=download_required,
        next_actions=tuple(actions),
        # Files on disk alone are never runtime readiness evidence.
        needs_comfy_restart_and_reaudit=refresh_needed,
        ready=audit.ready and not refresh_needed and not manual,
        manual_action_required=manual,
    )
