"""Read-only, native two-PC first-run guide, including missing-config states.

Never initialize the DB, launch a renderer, install tasks, fetch a model,
submit a GPU job or write a controller/renderer config. Preview commands are
structured argv, not executable shells. Configuration writes require the
operator to run an explicit opt-in command separately.
"""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from artifex.config import load_settings
from artifex.onboarding import ControllerDiscovery, discover_controller
from artifex.onboarding_pair import PairConfigurationPreview, discover_controller_pair
from artifex.onboarding_renderer_auto import RendererAutoPlan, plan_renderer_auto

FirstRunRole = Literal["controller", "renderer"]


class FirstRunGuide(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    role: FirstRunRole
    config_path: Path
    configuration_exists: bool
    native_windows: bool
    status: Literal["needs_input", "needs_review", "preview_ready", "configured"]
    needed_inputs: tuple[str, ...]
    blockers: tuple[str, ...]
    next_action: str
    next_safe_preview_argv: tuple[str, ...] | None
    # Explicitly suggested to a human, NEVER run by the guide.
    optional_config_write_argv: tuple[str, ...] | None
    controller_discovery: ControllerDiscovery | None = None
    renderer_plan: RendererAutoPlan | None = None
    authenticated_pair_preview: PairConfigurationPreview | None = None
    checked_live_pc_b: bool = False
    config_written: Literal[False] = False
    service_started: Literal[False] = False
    subprocess_commands_executed: Literal[False] = False
    gpu_jobs_submitted: Literal[False] = False
    production_qualified: Literal[False] = False


def _protected_path(path: Path) -> Path:
    selected = path.expanduser().absolute()
    if any(part.is_symlink() for part in (selected, *selected.parents)):
        raise ValueError("Refusing symlinked first-run input path")
    return selected


def compile_first_run_guide(
    *,
    role: FirstRunRole,
    config: Path,
    comfy_root: Path | None = None,
    attestation_url: str | None = None,
    gateway_port: int = 8191,
    token_env: str = "ARTIFEX_RENDER_NODE_TOKEN",
    client: httpx.Client | None = None,
    system_name: str | None = None,
) -> FirstRunGuide:
    """Check only known local facts plus an explicitly requested PC-B GET.

    Never infer a LAN IP, PC-B root, ownership or qualification from defaults.
    """
    config = _protected_path(config)
    if config.exists() and not config.is_file():
        raise ValueError("PC configuration path is not a regular file")
    has_config = config.is_file()
    native = (system_name or platform.system()) == "Windows"
    if role == "controller":
        if comfy_root is not None:
            raise ValueError("--comfy-root is only valid on PC-B")
        settings = load_settings(user_config=config if has_config else None)
        discovered = discover_controller(settings)
        preview = None
        blocks: list[str] = []
        inputs: list[str] = []
        safe: tuple[str, ...] | None
        opt_in: tuple[str, ...] | None = None
        if not native:
            blocks.append("First real-machine PC-A qualification requires native Windows")
        if attestation_url is not None:
            # The existing pairing API validates bounded URL, shared token,
            # fresh protected attestation, real gateway and checkpoint identity.
            preview = discover_controller_pair(
                attestation_url, gateway_port=gateway_port,
                token_env=token_env, client=client,
            )
            safe = (
                "uv", "run", "artifex", "onboard", "pair-sync",
                "--attestation-url", preview.attestation_url,
                "--gateway-port", str(gateway_port),
                "--token-env", token_env,
                "--output", str(config), "--json",
            )
            if native:
                opt_in = (*safe[:-1], "--apply", *("--update",) if has_config else (), "--json")
            status: Literal["needs_input", "needs_review", "preview_ready", "configured"] = (
                "preview_ready" if native else "needs_review"
            )
            next_action = (
                "Authenticated PC-B configuration preview is available. Review it, "
                "then separately approve the explicit config-only --apply step."
            )
        elif not has_config:
            inputs.append("PC-B authenticated attestation LAN URL (do not guess IP/port)")
            inputs.append("Shared token environment variable on PC-A and PC-B")
            safe = (
                "uv", "run", "artifex", "onboard", "inspect",
                "--role", "controller", "--json",
            )
            status = "needs_input"
            next_action = (
                "First identify the actual PC-B attestation URL and token environment "
                "variable; then rerun this guide with --attestation-url."
            )
        else:
            safe = (
                "uv", "run", "artifex", "qualify", "overview",
                "--config", str(config), "--json",
            )
            status = "configured" if native else "needs_review"
            next_action = (
                "PC-A config exists. Run the suggested read-only two-PC live "
                "overview to inspect remaining blockers; it never submits work."
            )
        if discovered.uv_executable is None:
            blocks.append("uv not found on PC-A PATH")
        if has_config and settings.render_nodes.primary_node() is None:
            blocks.append("PC-A config has no selected primary renderer")
        if not discovered.selected_llm_model_exists:
            blocks.append("Selected local GGUF is absent; opt-in model bootstrap is separate")
        if discovered.llama_server_executable is None and settings.llm.backend == "llama_cpp":
            blocks.append("Selected llama-server binary is absent; no automatic install")
        return FirstRunGuide(
            role=role, config_path=config, configuration_exists=has_config,
            native_windows=native, status=status, needed_inputs=tuple(inputs),
            blockers=tuple(blocks), next_action=next_action,
            next_safe_preview_argv=safe, optional_config_write_argv=opt_in,
            controller_discovery=discovered,
            authenticated_pair_preview=preview,
            checked_live_pc_b=preview is not None,
        )

    if role != "renderer":
        raise ValueError("PC role must be controller or renderer")
    if attestation_url is not None:
        raise ValueError("--attestation-url is only valid on PC-A")
    if client is not None:
        raise ValueError("HTTP client is only meaningful for explicit PC-A pairing")
    blocks = []
    inputs = []
    if not native:
        blocks.append("First real-machine PC-B qualification requires native Windows")
    plan = None
    safe = None
    opt_in = None
    if comfy_root is not None:
        comfy_root = _protected_path(comfy_root)
        # Reuse the authoritative no-GPU ComfyUI discovery and ambiguity
        # checks; do not pick an arbitrary checkpoint or Python.
        plan = plan_renderer_auto(comfy_root)
        blocks.extend(plan.blockers)
        safe = (
            "uv", "run", "artifex", "onboard", "renderer-auto",
            "--comfy-root", str(comfy_root),
            "--output", str(config), "--json",
        )
        if not has_config and native and plan.ready_to_write:
            opt_in = (*safe[:-1], "--apply", "--json")
    elif not has_config:
        inputs.append("Actual local PC-B ComfyUI project or portable root")
    if has_config:
        # Do not treat a PC-B YAML as ownership or permission to restart a
        # previously running ComfyUI instance. Local safety is separate.
        load_settings(user_config=config)
        safe = (
            "uv", "run", "artifex", "onboard", "renderer-safety",
            "--config", str(config), "--json",
        )
        status = "configured" if native else "needs_review"
        action = (
            "PC-B YAML exists. Inspect the actual native listener/PID, gateway "
            "and Task Scheduler ownership without changing any process."
        )
    elif plan is not None and plan.ready_to_write and native:
        status = "preview_ready"
        action = (
            "PC-B found one unambiguous proposed model and Python. Review the "
            "read-only preview before separately approving config-only --apply."
        )
    elif plan is not None:
        status = "needs_review"
        action = (
            "Review the ambiguous ComfyUI Python, checkpoint, or external model "
            "paths; choose exact files before configuring PC-B."
        )
    else:
        status = "needs_input"
        action = "Supply --comfy-root pointing to the actual local PC-B ComfyUI tree."
    return FirstRunGuide(
        role=role, config_path=config, configuration_exists=has_config,
        native_windows=native, status=status, needed_inputs=tuple(inputs),
        blockers=tuple(blocks), next_action=action,
        next_safe_preview_argv=safe, optional_config_write_argv=opt_in,
        renderer_plan=plan,
    )
