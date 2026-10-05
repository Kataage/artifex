from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from artifex.application import build_application, build_core, build_doctor
from artifex.config import load_settings
from artifex.discord import ArtifexRemoteOperations, CommandName, CommandRequest
from artifex.policy import PolicyDecisionRepository
from artifex.review import ReviewQueueRepository
from artifex.series import SeriesRepository

app = typer.Typer(
    name="artifex",
    help="Autonomous Illustration Production System",
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Optional user YAML configuration file."),
]


def _settings(config: Path | None):
    return load_settings(user_config=config)


def _remote(core):
    return ArtifexRemoteOperations(
        core.database,
        core.runtime,
        core.scheduler,
        ReviewQueueRepository(core.database),
        core.characters,
        SeriesRepository(core.database),
        PolicyDecisionRepository(core.database),
    )


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


@app.command()
def daemon(config: ConfigOption = None) -> None:
    """Start the long-running autonomous Artifex daemon."""
    settings = _settings(config)
    application = build_application(settings)
    try:
        asyncio.run(application.run())
    except KeyboardInterrupt:
        typer.echo("Artifex stopped.")


if __name__ == "__main__":
    app()
