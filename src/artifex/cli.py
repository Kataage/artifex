from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, cast

import httpx
import typer
from sqlalchemy import select

from artifex.application import CoreServices, build_application, build_core, build_doctor
from artifex.characters import HololiveCatalog
from artifex.comfy.dependency_installer import install_manifest, read_manifest
from artifex.comfy.dependency_drafter import draft_missing_node_manifest
from artifex.comfy.dependency_resolver import resolve_missing_dependencies
from artifex.comfy.isolated_install import install_isolated_comfy
from artifex.comfy.workflow_audit import audit_workflows
from artifex.config import load_settings
from artifex.config.models import ArtifexSettings
from artifex.configuration_edit import write_override
from artifex.controller_preflight import check_controller
from artifex.db import Database
from artifex.db.models import PackInventoryRow
from artifex.deployment import DeploymentRole, verify_deployment
from artifex.discord import ArtifexRemoteOperations, CommandName, CommandRequest
from artifex.evaluation import (
    SemanticArchiveIndexer,
    SemanticCalibrator,
    SemanticEmbeddingRepository,
    SemanticIndex,
    SigLIP2EmbeddingProvider,
    SimilarityService,
    load_calibration_manifest,
    load_calibration_profile,
    save_calibration_profile,
)
from artifex.llm import (
    LlmCallRepository,
    LlmQualificationService,
    OpenAICompatibleClient,
    StructuredGenerator,
    bootstrap_llm,
)
from artifex.llm.release_install import install_official_llama, official_release_assets
from artifex.llm.server import ManagedLlmServer
from artifex.native_dependencies import DependencyRole, check_native_dependencies
from artifex.onboarding import (
    configure_discovered_renderer,
    discover_controller,
    discover_renderer,
)
from artifex.performance import (
    PatreonV2PublicationProvider,
    ingest_manual_performance,
    load_manual_performance,
)
from artifex.policy import PolicyDecisionRepository
from artifex.qualification import (
    REQUIRED_STAGES,
    QualificationService,
    QualificationStage,
    QualificationStatus,
)
from artifex.render_node import (
    build_attestation,
    check_render_node,
)
from artifex.render_node.comfy_process import serve_managed_renderer
from artifex.research import (
    ResearchIntent,
    ResearchProviderError,
    ResearchSearchRequest,
    SafeSearch,
    SearchSource,
)
from artifex.review import ReviewQueueRepository
from artifex.series import SeriesRepository
from artifex.setup import configure_two_pc
from artifex.setup_renderer import configure_renderer
from artifex.telemetry import EventSeverity
from artifex.windows_tasks import StartupRole, install_task, task_status, uninstall_task

app = typer.Typer(
    name="artifex",
    help="Autonomous Illustration Production System",
    no_args_is_help=True,
)
characters_app = typer.Typer(
    name="characters",
    help="Bootstrap and audit versioned character catalogs.",
    no_args_is_help=True,
)
app.add_typer(characters_app, name="characters")
research_app = typer.Typer(
    name="research",
    help="Search and inspect bounded external research evidence.",
    no_args_is_help=True,
)
app.add_typer(research_app, name="research")
llm_app = typer.Typer(
    name="llm",
    help="Local LLM diagnostics and qualification.",
    no_args_is_help=True,
)
app.add_typer(llm_app, name="llm")
signals_app = typer.Typer(
    name="signals",
    help="Inspect and refresh Trend/Seasonal signal sources.",
    no_args_is_help=True,
)
app.add_typer(signals_app, name="signals")
inventory_app = typer.Typer(
    name="inventory",
    help="Inspect and manage finalized Pack inventory lifecycle.",
    no_args_is_help=True,
)
app.add_typer(inventory_app, name="inventory")
semantic_app = typer.Typer(
    name="semantic",
    help="Calibrate and inspect local semantic similarity evaluation.",
    no_args_is_help=True,
)
app.add_typer(semantic_app, name="semantic")
performance_app = typer.Typer(
    name="performance",
    help="Import and inspect Patreon publication performance learning.",
    no_args_is_help=True,
)
app.add_typer(performance_app, name="performance")
qualify_app = typer.Typer(
    name="qualify",
    help="Collect and verify target-Windows production qualification evidence.",
    no_args_is_help=True,
)
app.add_typer(qualify_app, name="qualify")
render_node_app = typer.Typer(
    name="render-node",
    help="Serve and inspect a lightweight remote ComfyUI render-node attestation.",
    no_args_is_help=True,
)
app.add_typer(render_node_app, name="render-node")
startup_app = typer.Typer(
    name="startup",
    help="Manage opt-in Windows logon startup tasks for both Artifex PCs.",
    no_args_is_help=True,
)
app.add_typer(startup_app, name="startup")
deployment_app = typer.Typer(
    name="deployment",
    help="Verify two-PC rollout and optionally generate one real transport smoke image.",
    no_args_is_help=True,
)
app.add_typer(deployment_app, name="deployment")
onboard_app = typer.Typer(
    name="onboard",
    help="Discover native two-PC dependencies and prepare safe first-run configuration.",
    no_args_is_help=True,
)
app.add_typer(onboard_app, name="onboard")

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Optional user YAML configuration file."),
]


def _settings(config: Path | None) -> ArtifexSettings:
    return load_settings(user_config=config)


@app.command("preflight")
def preflight(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Check PC-A to PC-B ComfyUI and authenticated renderer, plus the local LLM."""
    settings = _settings(config)
    try:
        report = check_controller(settings)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        typer.echo(f"controller preflight error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_payload(report.model_dump(mode="json"), as_json=json_output)
    if not report.ready:
        raise typer.Exit(code=1)


def _startup_role(value: str) -> StartupRole:
    selected = value.strip().casefold()
    if selected not in {"controller", "renderer"}:
        raise ValueError("startup role must be controller or renderer")
    return cast(StartupRole, selected)


@startup_app.command("install")
def startup_install(
    role: Annotated[str, typer.Option("--role")] = "controller",
    config: ConfigOption = None,
    replace: Annotated[
        bool, typer.Option("--replace", help="Update only an existing Artifex-owned task."),
    ] = False,
    restart_count: Annotated[int, typer.Option("--restart-count", min=0, max=999)] = 10,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Explicitly register a non-admin, current-user Windows logon startup task."""
    try:
        selected = _startup_role(role)
        target = config or Path(
            "config/local.yaml" if selected == "controller" else "config/render-node.yaml"
        )
        # Reject malformed config before writing anything to Task Scheduler.
        _settings(target)
        result = install_task(
            selected, config=target, replace=replace, restart_count=restart_count
        )
        _print_payload(result.model_dump(mode="json"), as_json=json_output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        typer.echo(f"startup install error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@startup_app.command("status")
def startup_status(
    role: Annotated[str, typer.Option("--role")] = "controller",
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Read Windows Task Scheduler registration without modifying it."""
    try:
        result = task_status(_startup_role(role))
        _print_payload(result.model_dump(mode="json"), as_json=json_output)
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"startup status error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@startup_app.command("uninstall")
def startup_uninstall(
    role: Annotated[str, typer.Option("--role")] = "controller",
) -> None:
    """Remove only an Artifex-owned startup task; do not stop a running daemon."""
    try:
        selected = _startup_role(role)
        removed = uninstall_task(selected)
        typer.echo(
            f"{selected}: startup task removed"
            if removed else f"{selected}: startup task not installed"
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"startup uninstall error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@deployment_app.command("verify")
def deployment_verify(
    role: Annotated[
        str, typer.Option("--role", help="Local PC role: controller or renderer."),
    ] = "controller",
    config: ConfigOption = None,
    require_autostart: Annotated[
        bool,
        typer.Option(
            "--require-autostart",
            help="Treat missing Windows startup registration as a blocking failure.",
        ),
    ] = False,
    render_smoke: Annotated[
        bool,
        typer.Option(
            "--render-smoke",
            help="Explicitly queue one real production workflow and download its image.",
        ),
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Optional directory for smoke images on PC-A."),
    ] = None,
    width: Annotated[
        int | None, typer.Option("--width", min=64, max=8192),
    ] = None,
    height: Annotated[
        int | None, typer.Option("--height", min=64, max=8192),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """One read-only rollout check; --render-smoke explicitly runs one real image."""
    if role not in {"controller", "renderer"}:
        typer.echo("deployment verify error: --role must be controller or renderer", err=True)
        raise typer.Exit(code=1)
    selected = cast(DeploymentRole, role)
    if config is None:
        config = Path(
            "config/local.yaml" if selected == "controller" else "config/render-node.yaml"
        )
    if not config.is_file():
        typer.echo(f"deployment verify error: config does not exist: {config}", err=True)
        raise typer.Exit(code=1)
    try:
        settings = _settings(config)
        report = asyncio.run(
            verify_deployment(
                settings,
                role=selected,
                require_autostart=require_autostart,
                render_smoke=render_smoke,
                output_dir=output_dir,
                width=width,
                height=height,
            )
        )
        _print_payload(report.model_dump(mode="json"), as_json=json_output)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"deployment verify error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not report.ready:
        raise typer.Exit(code=1)


@onboard_app.command("inspect")
def onboard_inspect(
    role: Annotated[
        str, typer.Option("--role", help="Select controller (PC-A) or renderer (PC-B)."),
    ] = "controller",
    root: Annotated[
        Path | None,
        typer.Option("--comfy-root", help="ComfyUI project or portable distribution root on PC-B."),
    ] = None,
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Read-only bounded discovery of native binaries, models and setup gaps."""
    if role not in {"controller", "renderer"}:
        typer.echo("onboard error: --role must be controller or renderer", err=True)
        raise typer.Exit(code=1)
    try:
        if role == "renderer":
            if root is None:
                raise ValueError("--comfy-root is required for renderer inspection")
            payload = discover_renderer(root).model_dump(mode="json")
        else:
            if root is not None:
                raise ValueError("--comfy-root applies only to renderer inspection")
            payload = discover_controller(_settings(config)).model_dump(mode="json")
        _print_payload(payload, as_json=json_output)
    except (OSError, ValueError, TypeError) as exc:
        typer.echo(f"onboard inspect error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@onboard_app.command("renderer")
def onboard_renderer(
    comfy_root: Annotated[Path, typer.Option("--comfy-root", help="ComfyUI or portable root.")],
    checkpoint_path: Annotated[
        Path, typer.Option("--checkpoint-path", help="Explicit selected checkpoint; never guessed."),
    ],
    output: Annotated[Path, typer.Option("--output")] = Path("config/render-node.yaml"),
    node_id: Annotated[str | None, typer.Option("--node-id")] = None,
    bind_host: Annotated[str | None, typer.Option("--bind-host")] = None,
    port: Annotated[int | None, typer.Option("--port", min=1, max=65535)] = None,
    comfy_port: Annotated[int, typer.Option("--comfy-port", min=1, max=65535)] = 8188,
    comfy_exe: Annotated[
        Path | None, typer.Option("--comfy-exe", help="Choose Python explicitly if ambiguous."),
    ] = None,
    external_comfy: Annotated[
        bool, typer.Option("--external-comfy", help="Leave ComfyUI process outside Artifex."),
    ] = False,
    refiner_path: Annotated[Path | None, typer.Option("--refiner-path")] = None,
    vae_path: Annotated[Path | None, typer.Option("--vae-path")] = None,
    upscaler_path: Annotated[Path | None, typer.Option("--upscaler-path")] = None,
    lora_dir: Annotated[
        list[Path] | None, typer.Option("--lora-dir", help="Extra LoRA root; repeat if needed."),
    ] = None,
    update: Annotated[bool, typer.Option("--update")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Create PC-B YAML using detected ComfyUI Python and explicitly selected models."""
    try:
        settings = _settings(output) if update and output.is_file() else _settings(None)
        result = configure_discovered_renderer(
            settings,
            root=comfy_root,
            output_path=output,
            checkpoint_path=checkpoint_path,
            node_id=node_id,
            bind_host=bind_host,
            port=port,
            comfy_port=comfy_port,
            python_executable=comfy_exe,
            external_comfy=external_comfy,
            refiner_path=refiner_path,
            vae_path=vae_path,
            upscaler_path=upscaler_path,
            extra_lora_roots=tuple(lora_dir or ()),
            update=update,
            force=force,
        )
        _print_payload(result.model_dump(mode="json"), as_json=json_output)
    except (FileExistsError, OSError, ValueError, TypeError) as exc:
        typer.echo(f"onboard renderer error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@onboard_app.command("comfy-install")
def onboard_comfy_install(
    commit: Annotated[
        str, typer.Option("--commit", help="Exact 40-character Comfy-Org/ComfyUI commit SHA."),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Root for a new isolated ComfyUI folder."),
    ] = Path("tools/ComfyUI"),
    install_deps: Annotated[
        bool, typer.Option("--install-deps", help="Create a private venv and install torch/requirements."),
    ] = False,
    torch_backend: Annotated[
        str | None,
        typer.Option("--torch-backend", help="With --install-deps: cpu, cu126, cu128, cu130."),
    ] = None,
    python: Annotated[
        Path | None, typer.Option("--python", help="Existing Python executable for the isolated venv."),
    ] = None,
    archive_sha256: Annotated[
        str | None,
        typer.Option("--archive-sha256", help="Optional independent expected source ZIP SHA-256."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Install exact official ComfyUI source into a new directory without overwriting."""
    try:
        result = install_isolated_comfy(
            commit, output_dir, install_dependencies=install_deps,
            torch_backend=torch_backend, python=python,
            expected_archive_sha256=archive_sha256,
        )
        _print_payload(result.model_dump(mode="json"), as_json=json_output)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        typer.echo(f"onboard comfy-install error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@onboard_app.command("workflow-audit")
def onboard_workflow_audit(
    config: ConfigOption = None,
    custom_nodes_root: Annotated[
        Path | None, typer.Option("--custom-nodes-root", help="Optional local folder name inventory."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Read-only ComfyUI /object_info audit for all configured production workflows."""
    try:
        report = asyncio.run(
            audit_workflows(_settings(config), custom_nodes_root=custom_nodes_root)
        )
        _print_payload(report.model_dump(mode="json"), as_json=json_output)
    except (OSError, ValueError, RuntimeError, KeyError, httpx.HTTPError) as exc:
        typer.echo(f"onboard workflow-audit error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not report.ready:
        raise typer.Exit(code=1)


@onboard_app.command("deps-resolve")
def onboard_deps_resolve(
    config: ConfigOption = None,
    manager_commit: Annotated[
        str | None,
        typer.Option(
            "--manager-commit",
            help="Optional 40-character ComfyUI-Manager commit; default fetches HEAD and pins it.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Read-only live workflow audit plus pinned Manager source-candidate lookup."""
    try:
        audit = asyncio.run(audit_workflows(_settings(config)))
        resolution = resolve_missing_dependencies(
            audit, manager_commit=manager_commit
        )
        _print_payload(
            {
                "audit_ready": audit.ready,
                "resolution": resolution.model_dump(mode="json"),
            },
            as_json=json_output,
        )
    except (OSError, ValueError, RuntimeError, KeyError, httpx.HTTPError) as exc:
        typer.echo(f"onboard deps-resolve error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not audit.ready:
        raise typer.Exit(code=1)


@onboard_app.command("deps-draft")
def onboard_deps_draft(
    config: ConfigOption = None,
    output: Annotated[
        Path, typer.Option("--output", help="New output path for fully pinned node manifest."),
    ] = Path("config/generated-dependencies.json"),
    manager_commit: Annotated[
        str | None,
        typer.Option("--manager-commit", help="Optional full pinned ComfyUI-Manager commit."),
    ] = None,
    max_repositories: Annotated[
        int, typer.Option("--max-repositories", min=1, max=12),
    ] = 3,
    max_archive_mib: Annotated[
        int, typer.Option("--max-archive-mib", min=1, max=512),
    ] = 128,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Audit, find unique providers, verify pinned ZIP/license, write install draft."""
    if output.exists() or output.is_symlink():
        typer.echo(f"onboard deps-draft error: refusing to overwrite {output}", err=True)
        raise typer.Exit(code=1)
    try:
        audit = asyncio.run(audit_workflows(_settings(config)))
        resolution = resolve_missing_dependencies(
            audit, manager_commit=manager_commit
        )
        draft = draft_missing_node_manifest(
            resolution,
            max_repositories=max_repositories,
            max_archive_mib=max_archive_mib,
        )
        payload = draft.model_dump(mode="json")
        payload["output"] = None
        if draft.manifest.artifacts:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(draft.manifest.model_dump_json(indent=2) + "\n")
            payload["output"] = str(output)
        _print_payload(payload, as_json=json_output)
        if not draft.manifest.artifacts:
            raise typer.Exit(code=1)
    except (FileExistsError, OSError, ValueError, TypeError, RuntimeError, httpx.HTTPError) as exc:
        typer.echo(f"onboard deps-draft error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@onboard_app.command("deps-install")
def onboard_deps_install(
    manifest: Annotated[
        Path, typer.Option("--manifest", help="SHA-256-pinned reviewed JSON manifest."),
    ],
    comfy_root: Annotated[
        Path, typer.Option("--comfy-root", help="Artifex-owned isolated ComfyUI source directory."),
    ],
    apply: Annotated[
        bool, typer.Option("--apply", help="Explicitly download and place verified artifacts."),
    ] = False,
    accept_licenses: Annotated[
        bool, typer.Option("--accept-licenses", help="Acknowledge reviewed dependency licenses."),
    ] = False,
    allow_custom_code: Annotated[
        bool,
        typer.Option(
            "--allow-custom-code",
            help="Allow third-party custom-node Python to be installed for next restart.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Default read-only artifact plan; --apply is explicit and never overwrites."""
    try:
        selected = read_manifest(manifest)
        if not apply:
            _print_payload(
                {
                    "dry_run": True,
                    "manifest": str(manifest),
                    "comfy_root": str(comfy_root),
                    "artifacts": [item.model_dump(mode="json") for item in selected.artifacts],
                    "notice": (
                        "No download or disk changes performed. Review SHA-256 sources "
                        "and licenses, then use --apply --accept-licenses. "
                        "Custom nodes also require --allow-custom-code."
                    ),
                },
                as_json=json_output,
            )
            return
        result = install_manifest(
            selected, comfy_root,
            accept_licenses=accept_licenses,
            allow_custom_code=allow_custom_code,
        )
        _print_payload(result.model_dump(mode="json"), as_json=json_output)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        typer.echo(f"onboard deps-install error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@onboard_app.command("dependencies")
def onboard_dependencies(
    role: Annotated[str, typer.Option("--role")] = "controller",
    config: ConfigOption = None,
    comfy_root: Annotated[
        Path | None,
        typer.Option("--comfy-root", help="Required PC-B ComfyUI or portable directory."),
    ] = None,
    probe_torch: Annotated[
        bool,
        typer.Option("--probe-torch", help="Execute selected PC-B Python to inspect PyTorch CUDA."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Inspect native dependencies without installing or modifying anything."""
    if role not in {"controller", "renderer"}:
        typer.echo("onboard dependencies error: invalid role", err=True)
        raise typer.Exit(code=1)
    if role == "controller" and (comfy_root is not None or probe_torch):
        typer.echo("onboard dependencies error: PC-B options require --role renderer", err=True)
        raise typer.Exit(code=1)
    try:
        selected = cast(DependencyRole, role)
        settings = _settings(config)
        report = check_native_dependencies(
            settings, role=selected, comfy_root=comfy_root, probe_torch=probe_torch
        )
        _print_payload(report.model_dump(mode="json"), as_json=json_output)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        typer.echo(f"onboard dependencies error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not report.ready:
        raise typer.Exit(code=1)


@onboard_app.command("llama-assets")
def onboard_llama_assets(
    tag: Annotated[str, typer.Option("--tag", help="Exact upstream llama.cpp release tag.")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """List official Windows x64 ZIP assets that publish GitHub SHA-256 digests."""
    try:
        assets = official_release_assets(tag)
        _print_payload(
            {"tag": tag, "assets": [asset.model_dump(mode="json") for asset in assets]},
            as_json=json_output,
        )
        if not assets:
            raise typer.Exit(code=1)
    except (ValueError, OSError, httpx.HTTPError) as exc:
        typer.echo(f"onboard llama-assets error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@onboard_app.command("llama-install")
def onboard_llama_install(
    tag: Annotated[str, typer.Option("--tag", help="Exact pinned official release tag.")],
    asset: Annotated[
        str, typer.Option("--asset", help="Exact Windows x64 ZIP name from llama-assets."),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Install root (never overwrites existing folders)."),
    ] = Path("tools/llama.cpp"),
    configure: Annotated[
        bool,
        typer.Option("--configure", help="Enable managed local llama-server in existing PC-A YAML."),
    ] = False,
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Explicit official SHA-256 verified install; never execute downloaded code."""
    if configure and config is None:
        typer.echo("onboard llama-install error: --configure requires --config", err=True)
        raise typer.Exit(code=1)
    try:
        if configure:
            assert config is not None
            if not config.is_file():
                raise FileNotFoundError(f"Existing controller config required: {config}")
            settings = _settings(config)
            if settings.llm.backend != "llama_cpp":
                raise ValueError("Managed llama-server requires llm.backend=llama_cpp")
        result = install_official_llama(tag, asset, output_dir)
        payload = result.model_dump(mode="json")
        if configure:
            assert config is not None
            updated = write_override(
                config,
                {
                    "llm": {
                        "server": {
                            "enabled": True,
                            "executable": str(result.executable),
                        }
                    }
                },
                update=True,
            )
            payload["configured_path"] = str(updated)
        _print_payload(payload, as_json=json_output)
    except (FileExistsError, OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        typer.echo(f"onboard llama-install error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("setup")
def setup(
    comfy_url: Annotated[
        str | None,
        typer.Option(
            "--comfy-url",
            help="Reachable ComfyUI URL; optional when --update reuses saved settings.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Controller override YAML to create.",
        ),
    ] = Path("config/local.yaml"),
    render_node_id: Annotated[
        str | None,
        typer.Option("--render-node-id"),
    ] = None,
    attestation_url: Annotated[
        str | None,
        typer.Option(
            "--attestation-url",
            help="Render attestation URL; defaults to the ComfyUI host on port 8190.",
        ),
    ] = None,
    llm_models_dir: Annotated[
        Path | None,
        typer.Option(
            "--llm-models-dir",
            help="Optional controller directory for the selected GGUF.",
        ),
    ] = None,
    render_cache_dir: Annotated[
        Path | None,
        typer.Option("--render-cache-dir", help="Controller folder for retrieved images."),
    ] = None,
    production_checkpoint: Annotated[
        str | None,
        typer.Option("--production-checkpoint", help="ComfyUI checkpoint filename."),
    ] = None,
    semantic_model_path: Annotated[
        Path | None,
        typer.Option("--semantic-model-path", help="Controller-local semantic model asset."),
    ] = None,
    llm_url: Annotated[
        str | None,
        typer.Option("--llm-url", help="Local llama.cpp/OpenAI-compatible server URL."),
    ] = None,
    llama_server_exe: Annotated[
        str | None,
        typer.Option("--llama-server-exe", help="Local llama-server binary (enables lifecycle management)."),
    ] = None,
    llama_device: Annotated[
        str | None,
        typer.Option("--llama-device", help="Optional llama.cpp --device selector."),
    ] = None,
    llama_gpu_layers: Annotated[
        str | None,
        typer.Option("--llama-gpu-layers", help="GPU offload layers: auto, all, or a nonnegative integer."),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Write controller config without requiring running PC-B."),
    ] = False,
    download_llm: Annotated[
        bool,
        typer.Option("--download-llm/--skip-llm-download"),
    ] = True,
    update: Annotated[
        bool,
        typer.Option("--update", help="Change only supplied values and preserve other settings."),
    ] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Create or safely update the PC-A controller config."""
    settings = _settings(output) if update and output.is_file() else _settings(None)
    primary = settings.render_nodes.primary_node()
    node_id = render_node_id or (primary[0] if primary else "renderer")
    base_url = comfy_url or (primary[1].base_url if primary else None)
    if not base_url:
        typer.echo("setup error: --comfy-url is required for initial setup", err=True)
        raise typer.Exit(code=1)
    resolved_attestation_url = attestation_url
    if resolved_attestation_url is None and comfy_url is None and primary is not None:
        resolved_attestation_url = primary[1].attestation_url
    if render_cache_dir is not None:
        settings.comfyui.download_dir = render_cache_dir
    if llm_models_dir is not None:
        bootstrap = settings.llm.bootstrap.model_copy(
            update={"models_dir": llm_models_dir}
        )
        settings = settings.model_copy(
            update={
                "llm": settings.llm.model_copy(
                    update={"bootstrap": bootstrap}
                )
            }
        )
    try:
        result = configure_two_pc(
            settings,
            comfyui_base_url=base_url,
            output_path=output,
            render_node_id=node_id,
            attestation_url=resolved_attestation_url,
            production_checkpoint=production_checkpoint,
            semantic_model_path=semantic_model_path,
            llm_base_url=llm_url,
            llama_server_executable=llama_server_exe,
            llama_server_device=llama_device,
            llama_server_gpu_layers=llama_gpu_layers,
            probe_comfyui=not offline,
            update=update,
            force=force,
        )
        payload = result.model_dump(mode="json")
        if download_llm:
            generated = load_settings(user_config=result.config_path)
            model = bootstrap_llm(generated.llm)
            payload["llm_download"] = model.model_dump(mode="json")
        else:
            payload["llm_download"] = None
    except (FileExistsError, OSError, ValueError, httpx.HTTPError) as exc:
        typer.echo(f"setup error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_payload(payload, as_json=json_output)


async def _calibrate_semantic(
    settings: ArtifexSettings,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if settings.evaluation.semantic_provider != "siglip2":
        raise ValueError(
            "semantic calibration requires evaluation.semantic_provider=siglip2"
        )
    database = Database(settings.storage.database_url)
    database.migrate()
    provider = SigLIP2EmbeddingProvider(
        model=settings.evaluation.semantic_model,
        revision=settings.evaluation.semantic_revision,
        device=settings.evaluation.semantic_device,
        cache_dir=settings.evaluation.semantic_cache_dir,
        local_files_only=settings.evaluation.semantic_local_files_only,
    )
    try:
        index = SemanticIndex(
            SemanticEmbeddingRepository(database),
            provider,
        )
        similarity = SimilarityService(provider, index)
        manifest = load_calibration_manifest(manifest_path)
        profile = await SemanticCalibrator(
            similarity,
            lambda: provider.descriptor,
        ).calibrate(manifest)
        save_calibration_profile(profile, output_path)
        backfill = await SemanticArchiveIndexer(database, index).backfill()
        return {
            "output": str(output_path),
            "backfill": {
                "concepts_scanned": backfill.concepts_scanned,
                "concepts_indexed": backfill.concepts_indexed,
                "images_scanned": backfill.images_scanned,
                "images_indexed": backfill.images_indexed,
                "missing_image_paths": backfill.missing_image_paths,
            },
            **profile.model_dump(mode="json"),
        }
    finally:
        await provider.aclose()
        database.dispose()


@semantic_app.command("calibrate")
def semantic_calibrate(
    manifest: Annotated[
        Path,
        typer.Argument(
            help=(
                "Curated ILXL/Hololive calibration manifest with identity, duplicate, "
                "and paraphrase positive/negative pairs."
            )
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Calibration profile output. Defaults to evaluation.semantic_calibration_path.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
    config: ConfigOption = None,
) -> None:
    """Measure production semantic thresholds from a curated real-output corpus."""
    settings = _settings(config)
    output_path = output or settings.evaluation.semantic_calibration_path
    try:
        payload = asyncio.run(
            _calibrate_semantic(settings, manifest, output_path)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"semantic calibration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_payload(payload, as_json=json_output)
    if payload.get("validated") is not True:
        raise typer.Exit(code=2)


@semantic_app.command("status")
def semantic_status(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Inspect the measured semantic calibration required for production."""
    settings = _settings(config)
    path = settings.evaluation.semantic_calibration_path
    if not path.is_file():
        typer.echo(f"semantic calibration missing: {path}", err=True)
        raise typer.Exit(code=1)
    try:
        profile = load_calibration_profile(path)
    except (OSError, ValueError) as exc:
        typer.echo(f"invalid semantic calibration: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = profile.model_dump(mode="json")
    payload["path"] = str(path)
    payload["configured_model"] = settings.evaluation.semantic_model
    payload["configured_profile_id"] = settings.evaluation.semantic_calibration_profile
    if json_output:
        _print_payload(payload, as_json=True)
    else:
        typer.echo(
            f"{profile.profile_id}: validated={profile.validated} "
            f"model={profile.model}@{profile.revision} corpus={profile.corpus_id}"
        )
        typer.echo(
            f"identity hard/accept={profile.identity_hard_min:.4f}/"
            f"{profile.identity_accept_min:.4f} duplicate={profile.similarity_hard_max:.4f} "
            f"text={profile.planner_hard_similarity_threshold:.4f}"
        )

    compatible = (
        profile.validated
        and profile.profile_id == settings.evaluation.semantic_calibration_profile
        and profile.model == settings.evaluation.semantic_model
        and (
            settings.evaluation.semantic_revision is None
            or profile.revision == settings.evaluation.semantic_revision
        )
    )
    if not compatible:
        raise typer.Exit(code=1)


def _remote(core: CoreServices) -> ArtifexRemoteOperations:
    return ArtifexRemoteOperations(
        core.database,
        core.runtime,
        core.scheduler,
        ReviewQueueRepository(core.database),
        core.characters,
        SeriesRepository(core.database),
        PolicyDecisionRepository(core.database),
        signals=core.signals,
        loras=core.loras,
        performance=core.performance,
    )




@characters_app.command("bootstrap-hololive")
def characters_bootstrap_hololive(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Destination YAML. Defaults to <first profile dir>/hololive.yaml.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace an existing generated catalog."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    config: ConfigOption = None,
) -> None:
    """Materialize the packaged versioned Hololive roster into a runtime profile."""
    settings = _settings(config)
    catalog = HololiveCatalog.packaged()
    if output is None:
        if not settings.characters.profile_dirs:
            raise typer.BadParameter("characters.profile_dirs is empty")
        output = settings.characters.profile_dirs[0] / "hololive.yaml"
    try:
        path = catalog.materialize(output, overwrite=force)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = {
        "catalog_id": catalog.catalog_id,
        "catalog_version": catalog.catalog_version,
        "characters": len(catalog.profiles),
        "path": str(path),
    }
    if json_output:
        _print_payload(payload, as_json=True)
        return
    typer.echo(
        f"{catalog.catalog_id}@{catalog.catalog_version}: "
        f"{len(catalog.profiles)} character(s) -> {path}"
    )


@characters_app.command("audit")
def characters_audit(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Audit installed profiles against the packaged Hololive roster."""
    settings = _settings(config)
    core = build_core(settings)
    try:
        report = HololiveCatalog.packaged().audit(
            core.characters,
            settings.characters.profile_dirs,
            minimum_readiness=settings.characters.minimum_readiness,
        )
        if json_output:
            _print_payload(report.model_dump(mode="json"), as_json=True)
            return

        typer.echo(
            f"{report.catalog_id}@{report.catalog_version} "
            f"loaded={report.loaded_count}/{report.expected_count} "
            f"complete={report.catalog_complete} "
            f"production_ready={report.production_ready}"
        )
        for name in (
            "missing_ids",
            "unexpected_ids",
            "duplicate_profile_ids",
            "excluded_namespace_ids",
            "disabled_ids",
            "unready_ids",
            "missing_canonical_tag_ids",
            "missing_policy_profile_ids",
            "missing_provenance_ids",
            "missing_reference_slot_ids",
        ):
            values = getattr(report, name)
            if values:
                typer.echo(f"{name}: {', '.join(values)}")
    finally:
        asyncio.run(core.close())


@render_node_app.command("configure")
def render_node_configure(
    output: Annotated[
        Path,
        typer.Option("--output", help="PC-B renderer override YAML."),
    ] = Path("config/render-node.yaml"),
    node_id: Annotated[
        str | None, typer.Option("--node-id", help="Must match PC-A render node ID."),
    ] = None,
    bind_host: Annotated[
        str | None, typer.Option("--bind-host", help="PC-B listen address (restrict firewall to trusted LAN)."),
    ] = None,
    port: Annotated[
        int | None, typer.Option("--port", min=1, max=65535),
    ] = None,
    comfy_url: Annotated[
        str | None, typer.Option("--comfy-url", help="ComfyUI URL as seen from PC-B."),
    ] = None,
    checkpoint_path: Annotated[
        Path | None, typer.Option("--checkpoint-path", help="Full production checkpoint path on PC-B."),
    ] = None,
    refiner_path: Annotated[
        Path | None, typer.Option("--refiner-path"),
    ] = None,
    vae_path: Annotated[
        Path | None, typer.Option("--vae-path"),
    ] = None,
    upscaler_path: Annotated[
        Path | None, typer.Option("--upscaler-path"),
    ] = None,
    lora_dir: Annotated[
        list[Path] | None,
        typer.Option("--lora-dir", help="PC-B LoRA folder. Repeat for multiple roots."),
    ] = None,
    token_env: Annotated[
        str | None, typer.Option("--token-env", help="Environment variable NAME only, never the secret."),
    ] = None,
    comfy_exe: Annotated[
        Path | None, typer.Option("--comfy-exe", help="PC-B Python or ComfyUI executable."),
    ] = None,
    comfy_workdir: Annotated[
        Path | None, typer.Option("--comfy-workdir", help="PC-B ComfyUI project directory."),
    ] = None,
    comfy_arg: Annotated[
        list[str] | None,
        typer.Option("--comfy-arg", help="One ComfyUI launch argument; repeat as needed."),
    ] = None,
    disable_comfy_management: Annotated[
        bool, typer.Option("--disable-comfy-management"),
    ] = False,
    update: Annotated[
        bool, typer.Option("--update", help="Preserve other YAML settings and change supplied values."),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Replace the entire renderer override YAML."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Create/update PC-B renderer paths, IPs and ports without requiring a running ComfyUI."""
    settings = _settings(output) if update and output.is_file() else _settings(None)
    supplied_assets = {
        name: value
        for name, value in (
            ("production_checkpoint", checkpoint_path),
            ("refiner_checkpoint", refiner_path),
            ("vae", vae_path),
            ("upscale_model", upscaler_path),
        )
        if value is not None
    }
    try:
        result = configure_renderer(
            settings,
            output_path=output,
            node_id=node_id,
            bind_host=bind_host,
            port=port,
            comfyui_base_url=comfy_url,
            asset_paths=supplied_assets,
            lora_roots=tuple(lora_dir) if lora_dir is not None else None,
            token_env=token_env,
            comfy_executable=comfy_exe,
            comfy_working_directory=comfy_workdir,
            comfy_arguments=tuple(comfy_arg) if comfy_arg is not None else None,
            disable_comfy_management=disable_comfy_management,
            update=update,
            force=force,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        typer.echo(f"render-node configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_payload(result.model_dump(mode="json"), as_json=json_output)


@render_node_app.command("attest")
def render_node_attest(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Print the local render-node asset/GPU attestation without starting a server."""
    settings = _settings(config)
    try:
        payload = build_attestation(settings).model_dump(mode="json")
    except (OSError, ValueError) as exc:
        typer.echo(f"render-node attestation error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_payload(payload, as_json=json_output)


@render_node_app.command("preflight")
def render_node_preflight(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Check PC-B's GPU, model hashes, LoRAs, ComfyUI and LAN configuration."""
    settings = _settings(config)
    try:
        report = check_render_node(settings)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        typer.echo(f"render-node preflight error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_payload(report.model_dump(mode="json"), as_json=json_output)
    if not report.ready:
        raise typer.Exit(code=1)


@render_node_app.command("serve")
def render_node_serve(
    config: ConfigOption = None,
) -> None:
    """Serve authenticated model/LoRA attestations for the Artifex controller."""
    settings = _settings(config)
    typer.echo(
        "render-node "
        f"{settings.render_agent.node_id} listening on "
        f"{settings.render_agent.bind_host}:{settings.render_agent.port}"
    )
    try:
        serve_managed_renderer(settings)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        typer.echo(f"render-node server error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@llm_app.command("bootstrap")
def llm_bootstrap(
    force: Annotated[bool, typer.Option("--force")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = True,
    config: ConfigOption = None,
) -> None:
    """Ensure the selected local LLM model is present, resuming downloads when possible."""
    settings = _settings(config)
    try:
        result = bootstrap_llm(settings.llm, force=force)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        typer.echo(f"LLM bootstrap error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_payload(result.model_dump(mode="json"), as_json=json_output)


@llm_app.command("qualify")
def llm_qualify(
    samples: Annotated[int, typer.Option("--samples", min=1, max=100)] = 8,
    json_output: Annotated[bool, typer.Option("--json")] = True,
    config: ConfigOption = None,
) -> None:
    """Run a real structured-output/context qualification against the configured LLM."""
    settings = _settings(config)
    database = Database(settings.storage.database_url)
    database.migrate()
    provenance = LlmCallRepository(database)
    client = OpenAICompatibleClient(settings.llm, provenance=provenance)
    generator = StructuredGenerator(
        client,
        repair_attempts=settings.llm.structured_repair_attempts,
    )
    try:
        report = asyncio.run(
            LlmQualificationService(
                settings.llm,
                generator,
                provenance,
            ).run(samples=samples)
        )
        _print_payload(report.model_dump(mode="json"), as_json=json_output)
    finally:
        asyncio.run(client.aclose())
        database.dispose()


@signals_app.command("status")
def signals_status(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show current signal counts and persisted provider health."""
    core = build_core(_settings(config))
    try:
        snapshot = core.signals.snapshot()
        if json_output:
            _print_payload(snapshot.model_dump(mode="json"), as_json=True)
            return
        typer.echo(
            f"trends={len(snapshot.trends)} seasonal={len(snapshot.seasonal)}"
        )
        if not snapshot.sources:
            typer.echo("source-health: no collection run recorded yet")
        for source in snapshot.sources:
            typer.echo(
                f"{source.kind}/{source.provider}\t{source.state}\t"
                f"count={source.last_count}\t"
                f"failures={source.consecutive_failures}\t"
                f"error={source.last_error or '-'}"
            )
    finally:
        asyncio.run(core.close())


@signals_app.command("list")
def signals_list(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List active Trend and Seasonal signals."""
    core = build_core(_settings(config))
    try:
        snapshot = core.signals.snapshot()
        if json_output:
            _print_payload(snapshot.model_dump(mode="json"), as_json=True)
            return
        for signal in snapshot.trends:
            typer.echo(
                f"trend\t{signal.id}\t{signal.topic}\t"
                f"strength={signal.strength:.2f}\t"
                f"confidence={signal.confidence:.2f}\t"
                f"freshness={signal.freshness:.2f}"
            )
        for event in snapshot.seasonal:
            typer.echo(
                f"seasonal\t{event.id}\t{event.title}\t"
                f"relevance={event.relevance:.2f}\tprovider={event.provider}"
            )
    finally:
        asyncio.run(core.close())


@signals_app.command("refresh")
def signals_refresh(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Force an immediate Trend/Seasonal collection cycle."""
    core = build_core(_settings(config))
    try:
        report = asyncio.run(core.signals.refresh())
        _print_payload(report.model_dump(mode="json"), as_json=json_output)
    finally:
        asyncio.run(core.close())


@inventory_app.command("list")
def inventory_list(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List completed Pack inventory and lifecycle states."""
    core = build_core(_settings(config))
    try:
        counts = core.editorial.inventory_counts()
        with core.database.session() as session:
            rows = session.scalars(
                select(PackInventoryRow).order_by(
                    PackInventoryRow.updated_at.desc(),
                    PackInventoryRow.pack_id.asc(),
                )
            ).all()
            items: list[dict[str, object]] = [
                {
                    "pack_id": row.pack_id,
                    "state": row.state,
                    "reserved_at": (
                        row.reserved_at.isoformat()
                        if row.reserved_at is not None
                        else None
                    ),
                    "consumed_at": (
                        row.consumed_at.isoformat()
                        if row.consumed_at is not None
                        else None
                    ),
                    "expires_at": (
                        row.expires_at.isoformat()
                        if row.expires_at is not None
                        else None
                    ),
                    "metadata": dict(row.metadata_json),
                }
                for row in rows
            ]
        if json_output:
            _print_payload(
                {
                    "counts": counts.model_dump(mode="json"),
                    "items": items,
                },
                as_json=True,
            )
            return
        typer.echo(
            f"available={counts.available} reserved={counts.reserved} "
            f"consumed={counts.consumed} expired={counts.expired}"
        )
        for row in rows:
            expires = (
                row.expires_at.isoformat()
                if row.expires_at is not None
                else "-"
            )
            typer.echo(f"{row.pack_id}\t{row.state}\texpires={expires}")
    finally:
        asyncio.run(core.close())


def _inventory_transition(
    pack_id: str,
    action: str,
    config: Path | None,
) -> None:
    core = build_core(_settings(config))
    try:
        if action == "reserve":
            core.editorial.reserve_pack(pack_id)
        elif action == "release":
            core.editorial.release_pack(pack_id)
        elif action == "consume":
            core.editorial.consume_pack(pack_id)
        elif action == "expire":
            core.editorial.expire_pack(pack_id)
        else:
            raise ValueError(f"unknown inventory action: {action}")
        typer.echo(f"{pack_id}: {action}d")
    finally:
        asyncio.run(core.close())


@inventory_app.command("reserve")
def inventory_reserve(pack_id: str, config: ConfigOption = None) -> None:
    """Reserve an available finalized Pack for publication/delivery."""
    _inventory_transition(pack_id, "reserve", config)


@inventory_app.command("release")
def inventory_release(pack_id: str, config: ConfigOption = None) -> None:
    """Release a reserved Pack back to available inventory."""
    _inventory_transition(pack_id, "release", config)


@inventory_app.command("consume")
def inventory_consume(pack_id: str, config: ConfigOption = None) -> None:
    """Mark a finalized Pack as consumed/published."""
    _inventory_transition(pack_id, "consume", config)


@inventory_app.command("expire")
def inventory_expire(pack_id: str, config: ConfigOption = None) -> None:
    """Explicitly expire a finalized Pack."""
    _inventory_transition(pack_id, "expire", config)


def _timelimit_from_since(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().casefold()
    direct = {"d": "d", "w": "w", "m": "m", "y": "y"}
    if normalized in direct:
        return direct[normalized]
    if normalized.endswith("d") and normalized[:-1].isdigit():
        days = int(normalized[:-1])
        if days <= 1:
            return "d"
        if days <= 7:
            return "w"
        if days <= 31:
            return "m"
        return "y"
    raise typer.BadParameter("--since must be d/w/m/y or a value such as 1d, 7d, 30d")


def _print_payload(payload: object, *, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(str(payload))


def _qualification_service(
    core: CoreServices,
) -> QualificationService:
    return QualificationService(
        core.settings,
        core.database,
        core.characters,
        core.loras,
    )


def _qualification_details(values: list[str] | None) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(
                f"qualification detail must be key=value: {item}"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("qualification detail key must not be empty")
        try:
            value: object = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        result[key] = value
    return result


@qualify_app.command("start")
def qualify_start(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Capture doctor, machine, workflow, GPU and asset-hash baseline evidence."""
    core = build_core(_settings(config))
    try:
        doctor = asyncio.run(build_doctor(core).run())
        service = _qualification_service(core)
        session = service.start(doctor)
        payload = {
            "session_id": session.session_id,
            "doctor_ready": session.doctor_ready,
            "evidence_path": str(
                core.settings.qualification.evidence_dir
                / session.session_id
                / "qualification.json"
            ),
            "environment_requirements": session.environment.get(
                "requirements",
                {},
            ),
            "missing_asset_labels": session.environment.get(
                "missing_asset_labels",
                [],
            ),
        }
        _print_payload(payload, as_json=json_output)
    finally:
        asyncio.run(core.close())


@qualify_app.command("stages")
def qualify_stages(
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List the strict real-machine qualification ladder."""
    values = [stage.value for stage in REQUIRED_STAGES]
    if json_output:
        _print_payload(values, as_json=True)
        return
    for index, stage in enumerate(values, start=1):
        typer.echo(f"{index}. {stage}")


@qualify_app.command("record")
def qualify_record(
    session_id: Annotated[str, typer.Argument(help="Qualification session ID.")],
    stage: Annotated[str, typer.Argument(help="Qualification stage name.")],
    status: Annotated[
        str,
        typer.Option("--status", help="pass, fail, or skipped."),
    ],
    pack_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--pack-id",
            help="Finalized Pack ID. Repeat for multi-Pack stages.",
        ),
    ] = None,
    details: Annotated[
        list[str] | None,
        typer.Option(
            "--detail",
            help="Verified key=value observation. Repeat as needed.",
        ),
    ] = None,
    note: Annotated[str | None, typer.Option("--note")] = None,
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Record one real-machine stage after validating its persisted evidence."""
    core = build_core(_settings(config))
    try:
        service = _qualification_service(core)
        try:
            parsed_stage = QualificationStage(stage.casefold())
            parsed_status = QualificationStatus(status.casefold())
            if parsed_status is QualificationStatus.PENDING:
                raise ValueError("record status cannot be pending")
            session = service.record(
                session_id,
                parsed_stage,
                status=parsed_status,
                pack_ids=tuple(pack_ids or ()),
                note=note,
                details=_qualification_details(details),
            )
        except (KeyError, OSError, ValueError) as exc:
            typer.echo(f"qualification record error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        evidence = session.stage(parsed_stage)
        _print_payload(
            evidence.model_dump(mode="json"),
            as_json=json_output,
        )
    finally:
        asyncio.run(core.close())


@qualify_app.command("status")
def qualify_status(
    session_id: Annotated[str, typer.Argument(help="Qualification session ID.")],
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show captured qualification stages and baseline readiness."""
    core = build_core(_settings(config))
    try:
        service = _qualification_service(core)
        try:
            session = service.load(session_id)
        except (KeyError, OSError, ValueError) as exc:
            typer.echo(f"qualification status error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if json_output:
            _print_payload(session.model_dump(mode="json"), as_json=True)
            return
        typer.echo(
            f"session={session.session_id} doctor_ready={session.doctor_ready} "
            f"host={session.hostname}"
        )
        for stage in REQUIRED_STAGES:
            evidence = session.stage(stage)
            typer.echo(
                f"{stage.value}\t{evidence.status.value}\t"
                f"packs={','.join(evidence.pack_ids) or '-'}\t"
                f"{evidence.note or ''}"
            )
    finally:
        asyncio.run(core.close())


@qualify_app.command("verify")
def qualify_verify(
    session_id: Annotated[str, typer.Argument(help="Qualification session ID.")],
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Revalidate all evidence and fail until the strict ladder is complete."""
    core = build_core(_settings(config))
    try:
        service = _qualification_service(core)
        try:
            result = service.verify(session_id)
        except (KeyError, OSError, ValueError) as exc:
            typer.echo(f"qualification verify error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        _print_payload(result, as_json=json_output)
        if result["ready"] is not True:
            raise typer.Exit(code=1)
    finally:
        asyncio.run(core.close())


@performance_app.command("import")
def performance_import(
    path: Annotated[
        Path,
        typer.Argument(
            help="JSON, JSONL/NDJSON, or CSV performance export."
        ),
    ],
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Import persisted post metrics linked back to Artifex Pack/Scene IDs."""
    settings = _settings(config)
    core = build_core(settings)
    try:
        payload = load_manual_performance(path)
        publications, snapshots = ingest_manual_performance(
            core.performance_repository,
            payload,
            import_path=path,
        )
        effects = (
            core.performance.strongest_effects(limit=10)
            if core.performance is not None
            else ()
        )
        result = {
            "platform": payload.platform,
            "source": payload.source,
            "publications": publications,
            "snapshots": snapshots,
            "effects": [
                effect.model_dump(mode="json")
                for effect in effects
            ],
        }
        core.telemetry.record(
            "performance.ingested",
            EventSeverity.INFO,
            {
                "platform": payload.platform,
                "source": payload.source,
                "publications": publications,
                "snapshots": snapshots,
                "top_effects": [
                    {
                        "dimension": effect.dimension,
                        "key": effect.key,
                        "score": effect.score,
                        "confidence": effect.confidence,
                    }
                    for effect in effects[:5]
                ],
            },
        )
        _print_payload(result, as_json=json_output)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        typer.echo(f"performance import error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        asyncio.run(core.close())


@performance_app.command("status")
def performance_status(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show latest persisted publication evidence and learned effects."""
    core = build_core(_settings(config))
    try:
        latest = core.performance_repository.latest_snapshots(platform="patreon")
        effects = (
            core.performance.strongest_effects(limit=20)
            if core.performance is not None
            else ()
        )
        payload = {
            "enabled": core.performance is not None,
            "publication_count": len(latest),
            "effects": [
                effect.model_dump(mode="json")
                for effect in effects
            ],
        }
        if json_output:
            _print_payload(payload, as_json=True)
            return
        typer.echo(
            f"performance enabled={payload['enabled']} "
            f"publications={len(latest)} effects={len(effects)}"
        )
        for effect in effects:
            typer.echo(
                f"{effect.dimension}\t{effect.key}\t"
                f"score={effect.score:.3f}\tconfidence={effect.confidence:.3f}\t"
                f"n={effect.evidence_count}\t{effect.reason}"
            )
    finally:
        asyncio.run(core.close())


@performance_app.command("sync-post")
def performance_sync_post(
    post_id: Annotated[str, typer.Argument(help="Patreon v2 post ID.")],
    pack_id: Annotated[str, typer.Argument(help="Generating Artifex Pack ID.")],
    scene_id: Annotated[
        str | None,
        typer.Option("--scene-id", help="Optional generating Scene ID."),
    ] = None,
    publication_tier: Annotated[
        str | None,
        typer.Option("--tier", help="Optional public/member publication tier."),
    ] = None,
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Sync canonical Patreon v2 post metadata and bind it to Artifex."""
    settings = _settings(config)
    core = build_core(settings)
    provider = PatreonV2PublicationProvider(settings.patreon)
    try:
        link = asyncio.run(
            provider.sync_post(
                core.performance_repository,
                post_id=post_id,
                pack_id=pack_id,
                scene_id=scene_id,
                publication_tier=publication_tier,
            )
        )
        core.telemetry.record(
            "performance.publication_synced",
            EventSeverity.INFO,
            {
                "platform": "patreon",
                "external_post_id": link.external_post_id,
                "pack_id": link.pack_id,
                "scene_id": link.scene_id,
            },
        )
        _print_payload(link.model_dump(mode="json"), as_json=json_output)
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        typer.echo(f"Patreon sync error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        asyncio.run(provider.aclose())
        asyncio.run(core.close())


@research_app.command("status")
def research_status(
    config: ConfigOption = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Show configured research providers and health."""
    core = build_core(_settings(config))
    try:
        reports = asyncio.run(core.research.health())
        payload = [report.model_dump(mode="json") for report in reports]
        if json_output:
            _print_payload(payload, as_json=True)
            return
        for report in reports:
            capabilities = ",".join(item.value for item in report.capabilities) or "-"
            typer.echo(
                f"{report.provider}\t{report.state.value}\t"
                f"capabilities={capabilities}\t{report.detail}"
            )
    finally:
        asyncio.run(core.close())


@research_app.command("sources")
def research_sources(
    config: ConfigOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Alias for provider/capability status."""
    research_status(config=config, json_output=json_output)


@research_app.command("search")
def research_search(
    query: str,
    source: Annotated[str, typer.Option("--source")] = "web",
    intent: Annotated[str, typer.Option("--intent")] = "evergreen",
    limit: Annotated[int, typer.Option("--limit", min=1, max=50)] = 8,
    region: Annotated[str | None, typer.Option("--region")] = None,
    safe_search: Annotated[str | None, typer.Option("--safe-search")] = None,
    since: Annotated[str | None, typer.Option("--since")] = None,
    adult: Annotated[bool, typer.Option("--adult")] = False,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    config: ConfigOption = None,
) -> None:
    """Search web/images/news/videos/tags through the stable Artifex contract."""
    settings = _settings(config)
    core = build_core(settings)
    try:
        request = ResearchSearchRequest(
            query=query,
            source=SearchSource(source.casefold()),
            intent=ResearchIntent(intent.casefold()),
            max_results=limit,
            region=region or settings.research.default_region,
            safesearch=SafeSearch(
                (safe_search or settings.research.default_safesearch).casefold()
            ),
            timelimit=_timelimit_from_since(since),
            adult=adult,
            backend=backend,
        )
        response = asyncio.run(
            core.research.search(request, use_cache=not no_cache)
        )
        payload = response.model_dump(mode="json")
        if json_output:
            _print_payload(payload, as_json=True)
            return
        typer.echo(
            f"run={response.run_id} cached={response.cached} "
            f"degraded={response.degraded}"
        )
        for item in response.evidence:
            typer.echo(
                f"{item.id}\t[{item.provider}/{item.source.value}] "
                f"{item.title}\t{item.canonical_url}"
            )
    except (ResearchProviderError, RuntimeError, ValueError) as exc:
        typer.echo(f"research error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        asyncio.run(core.close())


@research_app.command("tags")
def research_tags(
    query: str,
    limit: Annotated[int, typer.Option("--limit", min=1, max=50)] = 10,
    adult: Annotated[bool, typer.Option("--adult")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    config: ConfigOption = None,
) -> None:
    """Search structured tag metadata using an approved specialist provider."""
    settings = _settings(config)
    core = build_core(settings)
    try:
        response = asyncio.run(
            core.research.search(
                ResearchSearchRequest(
                    query=query,
                    source=SearchSource.TAGS,
                    intent=(
                        ResearchIntent.ADULT
                        if adult
                        else ResearchIntent.EVERGREEN
                    ),
                    max_results=limit,
                    region=settings.research.default_region,
                    safesearch=SafeSearch(
                        settings.research.adult_safesearch
                        if adult
                        else settings.research.default_safesearch
                    ),
                    adult=adult,
                )
            )
        )
        payload = response.model_dump(mode="json")
        if json_output:
            _print_payload(payload, as_json=True)
            return
        for item in response.evidence:
            typer.echo(f"{item.id}\t{item.title}\t{item.snippet}")
    except (ResearchProviderError, RuntimeError, ValueError) as exc:
        typer.echo(f"research error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        asyncio.run(core.close())


@research_app.command("inspect")
def research_inspect(
    evidence_id: str,
    extract: Annotated[bool, typer.Option("--extract")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = True,
    config: ConfigOption = None,
) -> None:
    """Inspect persisted evidence; optionally fetch bounded untrusted page text."""
    core = build_core(_settings(config))
    try:
        payload = asyncio.run(
            core.research.inspect(evidence_id, extract=extract)
        )
        _print_payload(payload, as_json=json_output)
    finally:
        asyncio.run(core.close())


@research_app.command("brief")
def research_brief(
    topic: Annotated[str, typer.Option("--topic")],
    adult: Annotated[bool, typer.Option("--adult")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = True,
    config: ConfigOption = None,
) -> None:
    """Build a bounded evidence-backed ResearchBrief for an explicit topic."""
    settings = _settings(config)
    core = build_core(settings)
    try:
        requests = [
            ResearchSearchRequest(
                query=topic,
                source=SearchSource.WEB,
                intent=ResearchIntent.CURRENT,
                max_results=settings.research.max_results_per_query,
                region=settings.research.default_region,
                safesearch=SafeSearch(settings.research.default_safesearch),
                timelimit="m",
                adult=adult,
            ),
            ResearchSearchRequest(
                query=topic,
                source=SearchSource.IMAGES,
                intent=ResearchIntent.COMPOSITION,
                max_results=settings.research.max_results_per_query,
                region=settings.research.default_region,
                safesearch=SafeSearch(settings.research.default_safesearch),
                timelimit="m",
                adult=adult,
            ),
        ]
        if adult:
            requests.append(
                ResearchSearchRequest(
                    query=topic,
                    source=SearchSource.TAGS,
                    intent=ResearchIntent.ADULT,
                    max_results=settings.research.max_results_per_query,
                    region=settings.research.default_region,
                    safesearch=SafeSearch(settings.research.adult_safesearch),
                    adult=True,
                )
            )
        brief = asyncio.run(core.research.brief(topic, requests, adult=adult))
        _print_payload(brief.model_dump(mode="json"), as_json=json_output)
    except (ResearchProviderError, RuntimeError, ValueError) as exc:
        typer.echo(f"research error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        asyncio.run(core.close())


def _run_remote(name: CommandName, config: Path | None, *args: str) -> None:
    core = build_core(_settings(config))
    try:
        response = _remote(core).execute(CommandRequest(name=name, args=tuple(args)))
        typer.echo(response.message)
        if not response.ok:
            raise typer.Exit(code=1)
    finally:
        asyncio.run(core.close())


@app.command()
def status(config: ConfigOption = None) -> None:
    """Show daemon and inventory status."""
    _run_remote(CommandName.STATUS, config)


@app.command()
def pause(config: ConfigOption = None) -> None:
    """Pause autonomous scheduling without losing persisted work."""
    _run_remote(CommandName.PAUSE, config)


@app.command()
def resume(config: ConfigOption = None) -> None:
    """Resume autonomous scheduling."""
    _run_remote(CommandName.RESUME, config)


@app.command()
def queue(config: ConfigOption = None) -> None:
    """Show queued/non-terminal Packs."""
    _run_remote(CommandName.QUEUE, config)


@app.command()
def current(config: ConfigOption = None) -> None:
    """Show the current active Pack and Scene states."""
    _run_remote(CommandName.CURRENT, config)


@app.command()
def recent(config: ConfigOption = None) -> None:
    """Show recently finalized/failed/blocked Packs."""
    _run_remote(CommandName.RECENT, config)


@app.command()
def retry(
    target_id: str,
    config: ConfigOption = None,
) -> None:
    """Retry a persisted review item or Scene."""
    _run_remote(CommandName.RETRY, config, target_id)


@app.command()
def characters(config: ConfigOption = None) -> None:
    """List enabled Character Registry profiles."""
    core = build_core(_settings(config))
    try:
        profiles = core.characters.list(enabled_only=True)
        if not profiles:
            typer.echo("No enabled character profiles.")
            return
        for profile in profiles:
            typer.echo(
                f"{profile.id}\t{profile.display_name}\t"
                f"readiness={profile.readiness:.2f}\t"
                f"lora={profile.lora_policy.value}"
            )
    finally:
        asyncio.run(core.close())


@app.command()
def loras(config: ConfigOption = None) -> None:
    """List discovered LoRAs and validation/readiness state."""
    core = build_core(_settings(config))
    try:
        profiles = core.loras.list()
        if not profiles:
            typer.echo("No LoRAs discovered.")
            return
        for profile in profiles:
            targets = ",".join(profile.target_character_ids) or "-"
            typer.echo(
                f"{profile.id}\t{profile.state.value}\t"
                f"readiness={profile.readiness:.2f}\t"
                f"targets={targets}\t{profile.path}"
            )
    finally:
        asyncio.run(core.close())


@app.command()
def doctor(config: ConfigOption = None) -> None:
    """Check LLM, ComfyUI, DB, storage, Discord and production readiness."""
    core = build_core(_settings(config))
    try:
        report = asyncio.run(build_doctor(core).run())
        for component in report.health.components:
            typer.echo(
                f"[{component.state.value.upper()}] "
                f"{component.name}: {component.detail}"
            )
        for check in report.checks:
            state = "READY" if check.ready else "NOT READY"
            typer.echo(f"[{state}] {check.name}: {check.detail}")
        if not report.ready:
            raise typer.Exit(code=1)
    finally:
        asyncio.run(core.close())


async def _run_daemon_with_llm_manager(settings: ArtifexSettings) -> None:
    manager = ManagedLlmServer(settings.llm)
    try:
        await manager.start()
        application = build_application(settings)
        if not settings.llm.server.enabled:
            await application.run()
            return
        async with asyncio.TaskGroup() as group:
            watcher = group.create_task(manager.watch())
            try:
                await application.run()
            finally:
                watcher.cancel()
    finally:
        await manager.close()


@app.command()
def daemon(config: ConfigOption = None) -> None:
    """Start the long-running autonomous Artifex daemon."""
    settings = _settings(config)
    if (
        settings.llm.backend == "llama_cpp"
        and settings.llm.bootstrap.enabled
        and settings.llm.bootstrap.auto_download
    ):
        try:
            model = bootstrap_llm(settings.llm)
        except (OSError, ValueError, httpx.HTTPError) as exc:
            typer.echo(f"LLM bootstrap error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"LLM model ready: {model.path}")
    try:
        asyncio.run(_run_daemon_with_llm_manager(settings))
    except KeyboardInterrupt:
        typer.echo("Artifex stopped.")


if __name__ == "__main__":
    app()
