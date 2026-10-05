from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from artifex.application import CoreServices, build_application, build_core, build_doctor
from artifex.config import load_settings
from artifex.config.models import ArtifexSettings
from artifex.db import Database
from artifex.discord import ArtifexRemoteOperations, CommandName, CommandRequest
from artifex.llm import (
    LlmCallRepository,
    LlmQualificationService,
    OpenAICompatibleClient,
    StructuredGenerator,
)
from artifex.policy import PolicyDecisionRepository
from artifex.research import (
    ResearchIntent,
    ResearchProviderError,
    ResearchSearchRequest,
    SafeSearch,
    SearchSource,
)
from artifex.review import ReviewQueueRepository
from artifex.series import SeriesRepository

app = typer.Typer(
    name="artifex",
    help="Autonomous Illustration Production System",
    no_args_is_help=True,
)
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

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Optional user YAML configuration file."),
]


def _settings(config: Path | None) -> ArtifexSettings:
    return load_settings(user_config=config)


def _remote(core: CoreServices) -> ArtifexRemoteOperations:
    return ArtifexRemoteOperations(
        core.database,
        core.runtime,
        core.scheduler,
        ReviewQueueRepository(core.database),
        core.characters,
        SeriesRepository(core.database),
        PolicyDecisionRepository(core.database),
    )




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
