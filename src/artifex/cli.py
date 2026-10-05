from __future__ import annotations

import typer

app = typer.Typer(
    name="artifex",
    help="Autonomous Illustration Production System",
    no_args_is_help=True,
)


@app.command()
def status() -> None:
    """Show the current Artifex service status."""
    typer.echo("Artifex repository bootstrap complete; runtime is not implemented yet.")


@app.command()
def daemon() -> None:
    """Start the long-running Artifex production daemon."""
    raise typer.Exit("Artifex daemon is not implemented yet.")
