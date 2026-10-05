# Artifex — Complete Implementation Specification

Status: **Normative**  
Target: **Initial Complete**  
Scope rule: the scope formerly discussed as v1, v1.5, and v2 is delivered together as the first complete implementation.

## 1. Product definition

Artifex is a local-first autonomous illustration production system. It continuously plans coherent multi-image content packs, resolves character and LoRA requirements, compiles prompts for a validated image-generation workflow, submits work to ComfyUI, evaluates results, retries failures by reason, archives provenance, and surfaces only exception/review work to the operator through Discord.

Artifex is not a one-shot prompt wrapper and not a system that asks the operator for a new instruction after every generation. Once enabled, the daemon maintains its own queue and continues until paused, blocked by policy/health conditions, or constrained by inventory limits.

The core engine is generic. Hololive is represented as a character/profile data set rather than hard-coded product logic, so other characters and original IP can be added without redesigning the runtime.

## 2. Initial Complete goals

Initial Complete MUST provide:

- autonomous long-running daemon
- idea inventory and work queue
- trend, evergreen, seasonal, exploration, and series-continuation idea sources
- character registry covering enabled character profiles
- LoRA registry with required/preferred/optional/none policies
- LoRA discovery, metadata inspection, validation state, and readiness scoring
- single-character, duo, and group planning
- coherent multi-scene Content Packs
- Series continuity and prior-pack memory
- structured LLM outputs validated by schemas
- model-family prompt compiler, initially optimized for ILXL/Danbooru-style prompting
- stable ComfyUI workflow templates patched at runtime
- concept and image similarity detection
- quality evaluation and automatic selection
- reason-aware retry policy
- accepted/review/rejected classification
- publication and rights policy gates as independent layers
- SQLite persistence and checkpoints
- process health monitoring and recovery
- Discord bot for status, review, approval, rejection, retry, pause/resume, skip, and queue inspection
- daily operational summary
- tests and CI sufficient to prevent schema, migration, state-machine, and adapter regressions

## 3. Non-goals for Initial Complete

The following are explicitly outside the first completion gate:

- training LoRAs automatically
- downloading third-party LoRAs automatically without operator-approved source policy
- autonomous publishing to external paid platforms
- video generation
- allowing the LLM to synthesize arbitrary ComfyUI graphs at runtime
- allowing policy decisions to be bypassed by prompt text

These may be added later without changing the core domain model.

## 4. Runtime topology

Recommended local topology:

```text
llama.cpp / local LLM
        |
        v
+----------------------------+
| Artifex daemon             |
|                            |
| Scheduler / Queue          |
| Idea Director / Planner    |
| Trend Collector            |
| Character Registry         |
| LoRA Registry / Resolver   |
| Pack / Series Engine       |
| Prompt Compiler            |
| ComfyUI Adapter            |
| Evaluator / Selector       |
| Retry Policy               |
| Rights / Publication Gates |
| SQLite / Archive           |
| Discord Bot                |
| Watchdogs                  |
+-------------+--------------+
              |
              v
          ComfyUI API
              |
              v
        production outputs
```

Primary development environment: Windows.  
Python environment/dependency management: `uv`.  
Reference LLM transport: OpenAI-compatible endpoint exposed by `llama-server`.  
Reference planning model: Spark-X2.5-4B-Heretic-jp or a compatible replacement.  
Reference image backend: ComfyUI.

The system MUST not depend on one exact LLM or image model implementation. Both are adapters.

## 5. Repository architecture

Target module boundaries:

```text
src/artifex/
  cli.py
  config/
  domain/
  db/
  scheduler/
  planner/
  trends/
  characters/
  loras/
  packs/
  series/
  prompts/
  comfy/
  evaluation/
  retry/
  policy/
  archive/
  discord/
  health/
  telemetry/
```

Runtime-owned data is excluded from git and stored under the configured data root.

## 6. Core domain objects

### 6.1 Character

Required fields:

- stable `id`
- display names
- franchise/profile namespace
- enabled flag
- branch/group metadata where applicable
- canonical tags
- aliases
- model-family compatibility
- LoRA policy
- preferred/default LoRA candidates
- readiness score
- policy profile reference
- generation notes/constraints
- timestamps and usage counters

Character data MUST be configuration/data driven.

### 6.2 LoRA

Required fields:

- stable `id`
- local file path / file identity
- checksum where practical
- type: character/style/outfit/utility/other
- target characters
- compatible model families
- trigger tags if known
- recommended weight
- validated min/max weight
- identity/flexibility/quality scores
- lifecycle state: discovered, pending, validated, production, disabled, failed
- provenance/source notes
- last validation timestamp

Character LoRA policy values:

- `none`
- `optional`
- `preferred`
- `required`

A character with `required` policy MUST NOT enter generation without a production-ready compatible LoRA.

### 6.3 Concept

A Concept represents the creative idea before prompt compilation.

Minimum structured fields:

- concept id
- character set
- concept family
- format
- theme
- setting
- mood
- visual hook
- continuity requirements
- target scene count
- source signals
- candidate score components
- similarity score
- selected/rejected reason

### 6.4 Content Pack

A Pack is the primary work unit.

It contains:

- pack id
- one or more characters
- format type
- concept
- scenes
- publication tiers
- series metadata
- lifecycle state
- generated assets
- evaluations
- review items
- provenance
- final metadata

The engine MUST plan the entire Pack before generating the first scene so that scenes remain coherent.

### 6.5 Scene

A Scene contains:

- scene id
- pack id
- ordering
- narrative/visual purpose
- participating characters
- continuity constraints
- structured visual specification
- publication tier intent
- compiled prompt provenance
- workflow template reference
- generation attempts
- selected output

### 6.6 GenerationAttempt

Must record enough information to reproduce a result:

- prompt and negative prompt after compilation
- seed
- workflow template/version
- workflow patch values
- checkpoint/model identifiers
- LoRA identifiers and weights
- dimensions and sampler settings
- output file identities
- timestamps
- backend status/error
- evaluation scores
- retry reason / parent attempt

### 6.7 Series

A Series maintains continuity across Packs:

- series id
- title/key
- current episode
- prior pack ids
- characters
- continuity state
- unresolved hooks
- preferred format
- lifecycle state

## 7. Planning and Idea Director

The planner MUST generate multiple candidate Concepts and select among them rather than accepting the first LLM answer.

Default candidate count: 8.

Primary idea sources:

- evergreen
- trend
- seasonal
- exploration
- series continuation

Initial mix target:

```yaml
evergreen: 0.40
trend: 0.30
seasonal: 0.20
exploration: 0.10
```

Series continuation is scheduled separately and can override the mix when continuity policy says a series is due.

Candidate scoring SHOULD include:

- trend strength
- evergreen strength
- character fit
- novelty
- visual strength
- seasonal relevance
- series potential
- historical performance when available
- readiness of required assets/LoRAs
- recent-character overuse penalty
- concept similarity penalty

A candidate above the configured hard similarity threshold MUST be rejected even if its aggregate score is high.

The planner MUST emit validated structured data. Invalid output is repaired/retried within a bounded schema-repair policy; raw free-form text is never passed directly into production.

## 8. Trend Collector

Trend collection is an independent subsystem.

Responsibilities:

- collect approved public trend signals
- normalize source-specific data
- attach observed timestamp and freshness
- deduplicate signals
- score strength/confidence
- expire stale signals
- provide concise structured summaries to the planner

The planner consumes TrendSignals, not unbounded raw social content.

Trend collection failure MUST NOT stop evergreen/seasonal/exploration production.

## 9. Character scheduling

Character selection MUST account for:

- enabled/readiness state
- required LoRA availability
- recent-use penalty
- series obligations
- trend/seasonal fit
- desired branch/group diversity
- historical success once sufficient data exists

All enabled profiles can participate; there is no hard-coded “main character” assumption.

Characters whose generation prerequisites are unresolved are skipped with an actionable status rather than repeatedly failing the queue.

## 10. LoRA discovery and resolution

### Discovery

The service scans configured LoRA roots and records additions/removals/changes.

Discovery SHOULD inspect:

- filename
- embedded metadata where available
- checksum/file identity
- known trigger/tag metadata
- configured aliases

Unknown files enter `discovered/pending`, never production automatically unless policy explicitly permits it.

### Resolution

Given Scene + characters + model family, the resolver returns a deterministic LoRA plan.

For multi-character scenes, the resolver MUST detect:

- incompatible base families
- excessive/conflicting LoRA stacks
- missing required LoRAs
- known pair/group conflicts

Unsafe/unvalidated combinations go to review or an alternate plan rather than being forced.

### Validation

Validation records quality observations and can promote a LoRA to `production`.

Readiness is numeric and separate from lifecycle state.

## 11. Pack formats

Initial Complete supports at least:

- `single_feature`
- `continuation`
- `mini_story`
- `variation_pack`
- `outfit_feature`
- `seasonal`
- `trend`
- `evergreen`
- `experimental`
- `duo`
- `group`

Duo/group are production features, not future placeholders.

A format defines scene roles and constraints, not exact prompt wording.

## 12. Publication tiers

The engine supports configurable tiers such as:

- `public`
- `member`
- `private_review`
- `blocked`

Pack planning can create a progression across tiers.

Publication classification is performed after generation as well as during planning, because generated output can differ from intent.

Policy modules MUST be configurable and independently testable. They must not rely only on the LLM.

## 13. Rights gate

Rights/commercial-use policy is a separate policy layer.

A profile references a rights-policy configuration. The engine does not hard-code legal conclusions into planner logic.

The gate can:

- allow generation
- allow only configured use classes
- block publication tiers
- require operator review
- block the Pack

Policy decisions and the policy version used MUST be recorded in provenance.

## 14. Prompt compiler

The LLM produces a structured Scene Specification.

The Prompt Compiler converts that specification into the target model-family representation.

The initial compiler is optimized for ILXL / Danbooru-style tags and MUST handle:

- canonical character tags
- required LoRA triggers
- composition
- camera/view
- pose
- expression
- clothing
- setting/background
- lighting
- atmosphere
- model-specific ordering rules
- negative constraints
- duplicate/conflict removal

Prompt compilation is deterministic given the same structured Scene Specification and registry version, except where a configured controlled-randomization policy is explicitly used.

The compiler interface MUST support future model families.

## 15. ComfyUI integration

Artifex uses stable validated workflow templates.

The LLM MUST NOT write or mutate arbitrary graph structure at runtime.

Typical runtime patches include:

- positive prompt
- negative prompt
- seed
- checkpoint/model
- LoRA file(s)
- LoRA weights
- width/height
- batch count where supported
- output prefix/path
- optional repair/detail workflow parameters

Workflow templates are versioned and referenced by GenerationAttempt.

The adapter MUST support:

- health check
- queue submission
- execution tracking
- timeout/error classification
- output discovery
- cancellation where supported
- reconnect after temporary loss

## 16. Evaluation

Evaluation MUST be multi-signal rather than a single aesthetic score.

Target score families:

- character identity
- prompt/scene alignment
- face quality
- anatomy/technical quality
- aesthetic quality
- similarity/novelty
- continuity with adjacent scenes
- output integrity

Thresholds are configurable by format and generation mode.

A strong global score MUST NOT override a hard character-identity failure.

Output state:

- accepted
- review
- rejected

Selected outputs and rejected attempts remain traceable in provenance.

## 17. Retry policy

Retry decisions are reason-aware.

Examples:

- low identity -> alternate LoRA / weight / identity-focused prompt correction
- poor composition -> structured scene/prompt revision
- anatomy/technical problem -> seed change and/or repair workflow
- excessive similarity -> concept/scene variation
- backend failure -> infrastructure retry without creative mutation
- missing asset -> block/skip and create actionable review state

Retries are bounded. Default maximum: 3 production retries per scene unless format-specific policy overrides it.

After exhaustion, the Scene enters review or Pack-level failure according to severity.

## 18. Similarity and diversity

Two distinct similarity systems are required:

### Concept similarity

Compare new Concepts against recent/historical Concept embeddings and structured attributes.

### Image similarity

Compare generated outputs against recent/historical image embeddings.

The system uses similarity for:

- duplicate rejection
- monotony penalties
- avoiding repeated camera/background/pose families
- detecting “different prompt, same-looking result”

Embedding provider is pluggable.

## 19. Persistence

SQLite is the reference datastore for local single-node operation.

Required logical entities:

- characters
- loras
- concepts
- packs
- scenes
- generation_attempts
- evaluations
- series
- trend_signals
- review_queue
- policy_decisions
- agent_events
- settings / migrations

Database migrations are mandatory.

Filesystem stores media and large artifacts; SQLite stores metadata and paths/identifiers.

## 20. Pack archive

Reference layout:

```text
data/
  artifex.sqlite3
  packs/
    YYYY/
      MM/
        PACK-xxxxxx/
          public/
          member/
          review/
          rejected/
          metadata/
          source/
  cache/
    embeddings/
    trends/
  checkpoints/
  logs/
```

Every finalized Pack writes machine-readable metadata sufficient for audit/reproduction.

## 21. Daemon state machine

Top-level service states:

- starting
- running
- paused
- degraded
- blocked
- stopping
- stopped

Pack lifecycle SHOULD include:

- idea
- planned
- policy_check
- generating
- evaluating
- review
- finalized
- failed
- blocked

Scene lifecycle SHOULD be explicit and persisted.

The daemon MUST NOT hold critical progress only in memory.

## 22. Autonomous inventory

The daemon maintains target inventories rather than requiring commands for every job.

Reference defaults:

```yaml
inventory:
  ideas: 30
  planned: 10
  completed: 7
```

When completed inventory is at/above target, expensive generation can idle while low-cost maintenance/trend work continues.

An explicit unlimited-production mode MAY exist but MUST not be the safety default.

## 23. Health and recovery

Watch:

- ComfyUI endpoint
- LLM endpoint
- database availability
- disk free space
- writeability of archive
- Discord connection when enabled
- worker heartbeat

Recovery behavior:

- transient backend outage -> backoff/reconnect
- repeated backend failure -> degraded/blocked + Discord alert
- process restart -> resume from persisted checkpoint
- machine restart -> no Pack corruption; recover incomplete work
- low disk -> stop new generation before archive integrity is threatened

Infrastructure retry and creative retry MUST be distinguished.

## 24. Discord bot

Discord is the primary remote observation/review interface.

Required commands/actions include:

- status
- pause
- resume
- current
- queue
- recent
- approve
- reject
- retry
- skip
- next
- character inspection
- series inspection

Pack review messages SHOULD include concise summary, representative previews where permitted, scores/reasons, and buttons such as:

- approve
- regenerate
- alternate idea
- reject
- details

Notification policy should avoid noise.

Notify by default for:

- Pack completed
- review required
- critical/repeated error
- backend unavailable
- low disk / blocked state
- daily summary

Do not notify for every single generated image unless debug mode is enabled.

Discord credentials are secrets and MUST never be stored in the repository.

## 25. CLI

Minimum target commands:

```text
artifex daemon
artifex status
artifex pause
artifex resume
artifex queue
artifex current
artifex recent
artifex doctor
artifex characters
artifex loras
artifex retry
```

CLI and Discord operate on the same persisted service state.

## 26. Configuration

Configuration precedence:

1. packaged defaults
2. user config file
3. environment overrides
4. explicit CLI override

Secrets use environment variables or an external secret mechanism.

Configuration is validated on startup and errors are actionable.

## 27. Security and operational constraints

- bind local services to loopback/private interfaces by default
- do not expose ComfyUI/LLM endpoints publicly by default
- never log Discord tokens or other secrets
- sanitize filesystem paths
- validate LLM JSON against schemas
- treat all external trend content as untrusted data
- policy gates cannot be disabled by model output
- destructive cleanup requires explicit retention policy

## 28. Observability

Structured events SHOULD capture:

- pack transitions
- scene transitions
- generation latency
- retry counts/reasons
- backend availability
- accepted/review/rejected counts
- character/LoRA readiness
- queue depth
- disk use
- daily totals

Daily Discord summary SHOULD emphasize exceptions and completed Packs rather than raw debug logs.

## 29. Testing strategy

Required automated coverage:

- schema validation
- configuration precedence
- DB migrations
- Pack/Scene state transitions
- planner candidate selection with deterministic fixtures
- LoRA resolver rules
- prompt compiler conflict handling
- policy gates
- retry routing
- ComfyUI adapter with mocked protocol
- daemon recovery/checkpoint scenarios
- Discord action authorization/routing
- similarity threshold behavior
- duo/group prerequisite failures

Integration tests SHOULD run without real GPUs by using fake adapters.

Real-backend qualification is a separate test tier.

## 30. CI quality gate

Every pull request must pass:

- Ruff
- mypy
- pytest

Later additions may include:

- migration validation
- coverage threshold
- dependency/security checks

No implementation issue is considered complete while its required CI is failing.

## 31. Initial Complete acceptance criteria

Initial Complete is achieved only when all of the following are demonstrated:

1. A clean Windows installation can install Artifex with `uv`.
2. `artifex doctor` correctly reports LLM, ComfyUI, DB, storage, and Discord readiness.
3. The daemon can start, stop, pause, and resume without corrupting state.
4. The planner automatically replenishes idea inventory.
5. Character selection respects readiness and recent-use rules.
6. Required LoRA policy is enforced.
7. LoRA discovery detects local additions and records validation state.
8. A single-character Pack can run end-to-end.
9. A duo Pack can run through prerequisite resolution and generation.
10. A group Pack can run through prerequisite resolution and generation.
11. Multi-scene Pack continuity data is maintained.
12. Series continuation can reference prior Packs.
13. Runtime prompt compilation produces target-model prompts from structured scenes.
14. ComfyUI uses versioned stable templates, not LLM-authored graphs.
15. Generation failures are classified and retried by reason.
16. Evaluation can accept, review, and reject outputs.
17. Concept and image similarity affect planning/selection.
18. Pack progress survives Artifex process restart.
19. Backend temporary outage recovers automatically.
20. Persistent backend failure surfaces an actionable Discord alert.
21. Discord can approve, reject, retry, pause, resume, and inspect status.
22. Policy gates can independently block publication/use without breaking generation metadata.
23. A finalized Pack has complete provenance and archive metadata.
24. The daemon can autonomously proceed to the next Pack without operator instruction.
25. A sustained-run test completes multiple Packs without unbounded state/resource growth.
26. All required CI tests pass.

## 32. Development workflow

Implementation is issue-driven.

Recommended order:

1. foundations: config/domain/db/migrations
2. service state machine and scheduler
3. LLM adapter + structured planner
4. character/LoRA registries and resolver
5. Pack/Series domain
6. prompt compiler
7. ComfyUI adapter
8. evaluation/similarity/retry
9. trend subsystem
10. policy gates
11. Discord bot
12. watchdog/recovery/telemetry
13. duo/group hardening
14. sustained-run production qualification

Each issue should be implemented on a branch, validated by CI, reviewed, and merged before dependent work relies on it.

## 33. Source of truth

This document is the normative implementation specification for Initial Complete.

If README summaries, issue descriptions, or implementation comments conflict with this specification, update the specification deliberately or treat this document as authoritative.
