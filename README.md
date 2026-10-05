# Artifex

**Artifex — Autonomous Illustration Production System**

Artifex is a local-first autonomous illustration production agent built around ComfyUI and a local LLM.

The target workflow is not “generate one image on request”. Artifex continuously plans content, resolves character/LoRA requirements, builds multi-image content packs, submits production workflows to ComfyUI, evaluates results, retries failures, archives provenance, and reports review items through Discord.

## Initial Complete scope

The first complete release includes the scope previously split across v1, v1.5, and v2:

- autonomous planner and idea queue
- trend / evergreen / seasonal / exploration idea sources
- character registry
- LoRA registry, discovery, selection, validation, and readiness scoring
- single / duo / group planning
- multi-scene Content Packs and Series continuity
- ILXL-oriented prompt compiler with pluggable model-family adapters
- ComfyUI API integration using stable production workflow templates
- image and concept similarity checks
- quality evaluation, selection, and reason-aware retries
- SQLite persistence and checkpoints
- long-running daemon with watchdog/recovery behavior
- Discord bot notifications, review, approval, retry, reject, pause/resume and status controls
- publication / rights gates as independent policy layers
- daily summaries and operational telemetry

## Design principles

1. **Autonomous by default** — the system keeps working without requiring a new user instruction for every pack.
2. **Pack-first production** — one job represents a coherent multi-image content pack, not a single disconnected image.
3. **Stable workflows** — the LLM does not rewrite ComfyUI graphs at runtime; Artifex patches validated templates.
4. **Character-aware generation** — character identity and LoRA requirements are explicit data, not prompt-time guesses.
5. **Recoverable operation** — every long-running job is checkpointed and can resume after process or machine failure.
6. **Human review by exception** — Discord should surface decisions only when automation cannot meet confidence thresholds.
7. **Policy separation** — generation capability, publication classification, and rights/commercial-use decisions are separate layers.

## Planned local topology

```text
Local LLM / llama.cpp
        |
        v
Artifex daemon
  |-- Planner / Idea Director
  |-- Character + LoRA Registry
  |-- Pack / Series Engine
  |-- Prompt Compiler
  |-- Evaluator / Retry Policy
  |-- SQLite / Archive
  |-- Discord Bot
        |
        v
ComfyUI API
        |
        v
Production outputs
```

The primary development environment is Windows with `uv` for Python dependency management.

## Development status

Repository bootstrap is in progress. The implementation specification is tracked under `docs/` and GitHub Issues.


## Initial Complete acceptance

The normative acceptance criteria and automated evidence matrix are documented in [`docs/acceptance.md`](docs/acceptance.md). Real local backend qualification is tracked separately from the reproducible CI acceptance tier.


## Research/Search

Artifex uses a native Windows research layer before autonomous ideation. Docker,
WSL, Podman and a local SearXNG instance are **not required**.

The default general-search provider is the embedded Python `ddgs` dependency.
Optional remote SearXNG and specialist metadata providers are routed behind the
same Artifex contract.

```powershell
uv sync
uv run artifex research status --json
uv run artifex research search "illustration composition" --source web --json
uv run artifex research search "illustration composition" --source images --since 30d --json
uv run artifex research tags "example_tag%" --json
```

Normal autonomous idea generation is research-gated: a fresh cached or newly
collected `ResearchBrief` must exist before the production `IdeaDirector`
can generate candidates. Evidence/run IDs are persisted with the Concept for
auditability.

Adult-rated research is disabled by default (`research.adult_enabled: false`).
When deliberately enabled, it remains a separate mode; adult evidence is
metadata-only for page inspection and does not enable raw explicit page
extraction.

## Hololive character catalog

The requested Hololive roster is maintained as a versioned, provenance-bearing
catalog rather than hand-authored one-off profiles. It covers current and historical
Hololive JP/ID/EN/DEV_IS/ASOBI talents plus historical hololive China while
deliberately excluding HOLOSTARS, holoAN and office staff.

```powershell
uv run artifex characters bootstrap-hololive --json
uv run artifex characters audit --json
```

The catalog keeps lifecycle status, canonical identity tags, aliases, wardrobe
records, local reference-image slots and source provenance. See
[`docs/hololive-catalog.md`](docs/hololive-catalog.md) for update policy and scope.

