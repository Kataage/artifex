from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy.dependency_installer import (
    DependencyInstallResult,
    DependencyManifest,
    InstalledArtifact,
    PinnedDependency,
)
from artifex.comfy.model_drafter import ModelManifestDraft
from artifex.comfy.model_sources import (
    ApprovedModelSource,
    ModelSourceRegistry,
    save_model_source_registry,
)
from artifex.comfy.preparation import (
    RendererPreparationReport,
    _existing_models,
    prepare_renderer,
)
from artifex.comfy.workflow_audit import (
    MissingWorkflowAsset,
    WorkflowAudit,
    WorkflowAuditEntry,
)
from artifex.config.models import ArtifexSettings


def _audit(*, missing: bool = True, with_nodes: bool = True) -> WorkflowAudit:
    asset = MissingWorkflowAsset(
        label="checkpoint", requested="illustration.safetensors",
        node_class="CheckpointLoaderSimple",
        input_name="ckpt_name", available=(),
    )
    entry = WorkflowAuditEntry(
        template_id="illust_main_v1", source_sha256="a" * 64,
        ready=not missing and not with_nodes,
        required_node_count=2, required_asset_count=1,
        missing_node_types=("MissingCustomNode",) if with_nodes else (),
        missing_assets=(asset,) if missing else (),
        unverifiable_assets=(),
    )
    return WorkflowAudit(
        ready=entry.ready, comfyui_url="http://localhost:8188",
        custom_node_folders=(), entries=(entry,), guidance=(),
    )


def _pinned() -> PinnedDependency:
    revision = "a" * 40
    return PinnedDependency(
        id="model-test-illustration", kind="model",
        url=f"https://huggingface.co/publisher/approved/resolve/{revision}/illustration.safetensors",
        sha256="b" * 64, size_bytes=10,
        license_id="apache-2.0",
        license_url=f"https://huggingface.co/publisher/approved/blob/{revision}/README.md",
        name="illustration.safetensors", model_folder="checkpoints",
    )


def _report(
    *,
    mode: str = "preview",
    ready: bool = False,
) -> RendererPreparationReport:
    return RendererPreparationReport(
        mode=mode, audit_ready_before=ready,
        models_requested=(), approved_sources=0, pinned_models=(),
        unresolved_models=(), missing_node_types=(), node_candidates=(),
        node_discovery_error=None, installed=(),
        needs_comfy_restart_and_reaudit=False,
        ready=ready, manual_action_required=False,
    )


@pytest.mark.asyncio
async def test_preview_with_unapproved_missing_model_does_not_query_hub_or_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.comfy.preparation as prep

    async def fake_audit(_: ArtifexSettings) -> WorkflowAudit:
        return _audit(with_nodes=False)

    monkeypatch.setattr(prep, "audit_workflows", fake_audit)
    monkeypatch.setattr(
        prep, "install_manifest",
        lambda *a, **kw: pytest.fail("preview must not install"),
    )
    report = await prepare_renderer(
        ArtifexSettings(), sources=tmp_path / "missing-sources.json",
        discover_nodes=False,
    )
    assert report.mode == "preview"
    assert report.approved_sources == 0
    assert report.pinned_models == ()
    assert report.installed == ()
    assert len(report.unresolved_models) == 1
    assert "No operator-approved source" in report.unresolved_models[0]["reason"]
    assert report.models_requested == ("checkpoint:illustration.safetensors",)
    assert report.manual_action_required and not report.ready
    assert report.production_qualified is False


@pytest.mark.asyncio
async def test_apply_requires_license_and_owned_install_root_before_any_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.comfy.preparation as prep

    async def never_audit(_: ArtifexSettings) -> WorkflowAudit:
        pytest.fail("invalid installation is rejected before contacting ComfyUI")

    monkeypatch.setattr(prep, "audit_workflows", never_audit)
    root = tmp_path / "unowned-ComfyUI"
    root.mkdir()
    (root / "main.py").touch()
    with pytest.raises(ValueError, match="--accept-licenses"):
        await prepare_renderer(
            ArtifexSettings(), sources=tmp_path / "sources.json",
            comfy_root=root, apply=True,
        )
    with pytest.raises(ValueError, match="isolated"):
        await prepare_renderer(
            ArtifexSettings(), sources=tmp_path / "sources.json",
            comfy_root=root, apply=True, accept_licenses=True,
        )
    with pytest.raises(ValueError, match="--comfy-root"):
        await prepare_renderer(
            ArtifexSettings(), sources=tmp_path / "sources.json",
            apply=True, accept_licenses=True,
        )
    with pytest.raises(ValueError, match="requires --apply"):
        await prepare_renderer(
            ArtifexSettings(), sources=tmp_path / "sources.json",
            accept_licenses=True,
        )


@pytest.mark.asyncio
async def test_apply_only_approved_pinned_weights_and_never_node_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.comfy.preparation as prep

    source = ApprovedModelSource(
        role="checkpoint", model_name="illustration.safetensors",
        repository="publisher/approved", file_path="illustration.safetensors",
        rationale="Publisher and license checked during approval",
    )
    sources = tmp_path / "sources.json"
    save_model_source_registry(sources, ModelSourceRegistry(sources=(source,)))
    root = tmp_path / "owned-comfy"
    root.mkdir()
    installs: list[dict[str, Any]] = []
    drafted: list[tuple[str, ...]] = []

    async def fake_audit(_: ArtifexSettings) -> WorkflowAudit:
        return _audit(with_nodes=True)

    def fake_draft(audit: WorkflowAudit, *, repositories: tuple[str, ...],
                   bindings: tuple[ApprovedModelSource, ...]) -> ModelManifestDraft:
        assert audit.entries[0].missing_assets
        assert repositories == ()
        drafted.append(tuple(src.repository for src in bindings))
        return ModelManifestDraft(
            manifest=DependencyManifest(artifacts=(_pinned(),)),
            evidence=(), unresolved=(), repositories_checked=("publisher/approved",),
        )

    def fake_install(manifest: DependencyManifest, path: Path,
                     **kwargs: Any) -> DependencyInstallResult:
        installs.append(kwargs)
        assert path == root
        assert len(manifest.artifacts) == 1
        assert manifest.artifacts[0].kind == "model"
        return DependencyInstallResult(
            comfyui_directory=root,
            installed=(InstalledArtifact(
                id="model-test-illustration",
                destination=root / "models" / "checkpoints" / "illustration.safetensors",
                sha256="b" * 64, kind="model",
            ),),
            restart_required=False,
        )

    monkeypatch.setattr(prep, "audit_workflows", fake_audit)
    monkeypatch.setattr(prep, "draft_hf_models", fake_draft)
    monkeypatch.setattr(prep, "install_manifest", fake_install)
    monkeypatch.setattr(prep, "_safe_install_root", lambda path: path)
    preview = await prepare_renderer(
        ArtifexSettings(), sources=sources, discover_nodes=False,
    )
    assert preview.mode == "preview" and not preview.installed
    assert len(preview.pinned_models) == 1
    applied = await prepare_renderer(
        ArtifexSettings(), sources=sources, comfy_root=root,
        apply=True, accept_licenses=True, discover_nodes=False,
    )
    assert drafted == [("publisher/approved",)] * 2
    assert installs == [{"accept_licenses": True, "allow_custom_code": False}]
    assert applied.installed[0]["kind"] == "model"
    assert applied.needs_comfy_restart_and_reaudit
    assert not applied.ready and applied.manual_action_required
    assert applied.missing_node_types == ("MissingCustomNode",)


@pytest.mark.asyncio
async def test_ready_only_from_live_runtime_audit_and_node_discovery_failure_is_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.comfy.preparation as prep

    async def healthy(_: ArtifexSettings) -> WorkflowAudit:
        return _audit(missing=False, with_nodes=False)

    monkeypatch.setattr(prep, "audit_workflows", healthy)
    report = await prepare_renderer(
        ArtifexSettings(), sources=tmp_path / "none.json",
    )
    assert report.ready and not report.needs_comfy_restart_and_reaudit
    assert report.pinned_models == () and report.production_qualified is False

    async def missing_nodes(_: ArtifexSettings) -> WorkflowAudit:
        return _audit(missing=False, with_nodes=True)

    monkeypatch.setattr(prep, "audit_workflows", missing_nodes)
    monkeypatch.setattr(
        prep, "resolve_missing_dependencies",
        lambda *_: (_ for _ in ()).throw(OSError("Node registry unavailable")),
    )
    pending = await prepare_renderer(
        ArtifexSettings(), sources=tmp_path / "none.json",
    )
    assert not pending.ready
    assert pending.node_discovery_error == "Node registry unavailable"
    assert pending.manual_action_required
    assert pending.installed == ()


def test_prepare_renderer_cli_preview_and_apply_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "render-node.yaml"
    config.write_text("agent: {}\n", encoding="utf-8")
    options: list[dict[str, Any]] = []

    async def mock_prepare(_: ArtifexSettings, **kwargs: Any) -> RendererPreparationReport:
        options.append(kwargs)
        return _report(mode="apply" if kwargs["apply"] else "preview")

    monkeypatch.setattr("artifex.cli.prepare_renderer", mock_prepare)
    runner = CliRunner()
    preview = runner.invoke(
        app, ["onboard", "prepare-renderer", "--config", str(config)]
    )
    assert preview.exit_code == 0, preview.output
    assert options[-1]["apply"] is False
    assert '"production_qualified": false' in preview.output

    applied = runner.invoke(
        app, [
            "onboard", "prepare-renderer", "--config", str(config),
            "--comfy-root", str(tmp_path / "isolated"),
            "--apply", "--accept-licenses", "--no-discover-nodes",
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert options[-1]["apply"] is True
    assert options[-1]["discover_nodes"] is False
    assert options[-1]["accept_licenses"] is True

    missing = runner.invoke(
        app, [
            "onboard", "prepare-renderer", "--config",
            str(tmp_path / "missing.yaml"), "--apply",
        ],
    )
    assert missing.exit_code != 0
    assert "existing non-symlink config required" in missing.output


def _actual_pinned(data: bytes, *, name: str = "illustration.safetensors") -> PinnedDependency:
    return _pinned().model_copy(
        update={
            "name": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    )


def _isolated_comfy(tmp_path: Path) -> Path:
    root = tmp_path / "isolated" / "ComfyUI"
    root.mkdir(parents=True)
    (root / "main.py").touch()
    (root.parent / ".artifex-comfy-install.json").write_text(
        '{"source":"Comfy-Org/ComfyUI"}', encoding="utf-8"
    )
    return root


def test_existing_pinned_models_are_reused_by_exact_sha_without_overwrite(
    tmp_path: Path,
) -> None:
    root = _isolated_comfy(tmp_path)
    data = b"trusted-verified-model-weights"
    target = root / "models" / "checkpoints" / "illustration.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    wanted = _actual_pinned(data)
    another = _actual_pinned(b"more", name="second.safetensors")
    pending, existing = _existing_models(
        root, DependencyManifest(artifacts=(wanted, another))
    )
    assert pending.artifacts == (another,)
    assert len(existing) == 1
    assert existing[0]["sha256"] == wanted.sha256
    assert existing[0]["destination"] == str(target)
    assert target.read_bytes() == data


def test_existing_model_mismatch_and_symlinks_fail_before_download(
    tmp_path: Path,
) -> None:
    root = _isolated_comfy(tmp_path)
    correct = b"expected source model"
    wanted = _actual_pinned(correct)
    dest = root / "models" / "checkpoints" / wanted.name
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"modified weights!!!!")
    with pytest.raises(ValueError, match="size differs|SHA-256 differs"):
        _existing_models(root, DependencyManifest(artifacts=(wanted,)))
    dest.write_bytes(b"x" * len(correct))
    with pytest.raises(ValueError, match="SHA-256 differs"):
        _existing_models(root, DependencyManifest(artifacts=(wanted,)))
    dest.unlink()
    dest.symlink_to(root / "main.py")
    with pytest.raises(ValueError, match="symlinked"):
        _existing_models(root, DependencyManifest(artifacts=(wanted,)))
    dest.unlink()
    dest.parent.rmdir()
    dest.parent.symlink_to(root)
    with pytest.raises(ValueError, match="symlinked"):
        _existing_models(root, DependencyManifest(artifacts=(wanted,)))


@pytest.mark.asyncio
async def test_repeat_apply_does_not_redownload_verified_model_and_requires_runtime_reaudit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.comfy.preparation as prep

    model_data = b"already-pinned-artifact"
    item = _actual_pinned(model_data)
    root = _isolated_comfy(tmp_path)
    destination = root / "models" / "checkpoints" / item.name
    destination.parent.mkdir(parents=True)
    destination.write_bytes(model_data)
    bind = ApprovedModelSource(
        role="checkpoint", model_name=item.name, repository="publisher/approved",
        rationale="Operator previously approved the publisher and license",
    )
    sources = tmp_path / "model-sources.json"
    save_model_source_registry(sources, ModelSourceRegistry(sources=(bind,)))

    async def runtime_missing(_: ArtifexSettings) -> WorkflowAudit:
        return _audit(with_nodes=False)

    def draft(*args: Any, **kwargs: Any) -> ModelManifestDraft:
        assert kwargs["bindings"] == (bind,)
        return ModelManifestDraft(
            manifest=DependencyManifest(artifacts=(item,)),
            evidence=(), unresolved=(), repositories_checked=("publisher/approved",),
        )

    monkeypatch.setattr(prep, "audit_workflows", runtime_missing)
    monkeypatch.setattr(prep, "draft_hf_models", draft)
    monkeypatch.setattr(
        prep, "install_manifest",
        lambda *a, **kw: pytest.fail("matching model must never be downloaded again"),
    )
    preview = await prepare_renderer(
        ArtifexSettings(), sources=sources, comfy_root=root,
        discover_nodes=False,
    )
    assert preview.mode == "preview" and preview.download_required == ()
    assert len(preview.existing_verified) == 1
    assert preview.needs_comfy_restart_and_reaudit
    assert preview.next_actions == (
        "refresh_or_restart_comfyui_then_repeat_live_workflow_audit",
    )
    assert not preview.ready

    applied = await prepare_renderer(
        ArtifexSettings(), sources=sources, comfy_root=root,
        apply=True, accept_licenses=True, discover_nodes=False,
    )
    assert not applied.installed
    assert applied.existing_verified == preview.existing_verified
    assert applied.download_required == ()
    assert not applied.ready
    assert applied.production_qualified is False
    assert destination.read_bytes() == model_data


@pytest.mark.asyncio
async def test_corrupt_existing_model_blocks_all_other_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifex.comfy.preparation as prep

    data = b"expected trusted bytes"
    item = _actual_pinned(data)
    other = _actual_pinned(b"other model", name="other.safetensors")
    root = _isolated_comfy(tmp_path)
    destination = root / "models" / "checkpoints" / item.name
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong but same length"[:len(data)].ljust(len(data), b"x"))
    bind = ApprovedModelSource(
        role="checkpoint", model_name=item.name, repository="publisher/approved",
        rationale="Operator already approved this model source",
    )
    sources = tmp_path / "model-sources.json"
    save_model_source_registry(sources, ModelSourceRegistry(sources=(bind,)))
    async def audit(_: ArtifexSettings) -> WorkflowAudit:
        return _audit(with_nodes=False)
    monkeypatch.setattr(prep, "audit_workflows", audit)
    monkeypatch.setattr(
        prep, "draft_hf_models",
        lambda *args, **kwargs: ModelManifestDraft(
            manifest=DependencyManifest(artifacts=(other, item)),
            evidence=(), unresolved=(), repositories_checked=("publisher/approved",),
        ),
    )
    monkeypatch.setattr(
        prep, "install_manifest",
        lambda *args, **kwargs: pytest.fail("must preflight entire manifest before any download"),
    )
    with pytest.raises(ValueError, match="SHA-256 differs"):
        await prepare_renderer(
            ArtifexSettings(), sources=sources, comfy_root=root,
            apply=True, accept_licenses=True, discover_nodes=False,
        )
    assert not (destination.parent / other.name).exists()
    assert destination.read_bytes() != data
