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
    typer.echo("Artifex daemon is not implemented yet.", err=True)
    raise typer.Exit(code=1)
