# Two-PC Artifex topology

Artifex supports a split native-Windows deployment where the controller and renderer are separate machines.

- **PC-A / controller**: Artifex, local LLM, research/planning, SQLite, archive, evaluation, Discord/Patreon integration.
- **PC-B / render node**: ComfyUI, production checkpoint, refiner, VAE, upscaler, LoRAs and the lightweight Artifex render-node attestation server.

No Docker, WSL, shared SMB output folder, or duplicate LoRA/model copy on PC-A is required.

## Native dependency diagnosis and verified llama.cpp binary installation

Artifex separates read-only dependency checks from opt-in installation. No
Docker, package-manager administrator permissions, local MCP tunnel, or
automatic replacement of ComfyUI/custom nodes is required.

**PC-A — inspect Windows, Python, uv, NVIDIA driver, chosen GGUF and
llama-server binary:**

```powershell
uv run artifex onboard dependencies --role controller `
  --config .\\config\\local.yaml --json
```

**PC-B — inspect the local ComfyUI Python/GPU/CUDA pairing:**

```powershell
uv run artifex onboard dependencies --role renderer `
  --config .\\config\\render-node.yaml `
  --comfy-root "E:/ComfyUI_windows_portable" `
  --probe-torch --json
```

`--probe-torch` intentionally invokes the **selected local ComfyUI Python**
with a bounded-time, read-only `import torch`/CUDA availability check.
Omit it to inspect dependencies without running the selected Python.
The report includes detected PyTorch version, built CUDA version, NVIDIA
driver version and discovered devices. This checks presence/availability;
it cannot guarantee that every installed custom node or model runs properly.

**PC-A — install one explicitly selected official llama.cpp Windows release:**

```powershell
uv run artifex onboard llama-assets --tag b12345 --json
uv run artifex onboard llama-install `
  --tag b12345 `
  --asset "EXACT-ASSET-NAME-FROM-LIST.zip" `
  --output-dir "D:/AI/Artifex/tools/llama.cpp" `
  --config .\\config\\local.yaml --configure
```

`b12345` and the ZIP name are **illustrative**: choose a currently
published tag and a matching Windows x64 CPU/CUDA backend from
<https://github.com/ggml-org/llama.cpp/releases>. CUDA 12, CUDA 13,
Vulkan and CPU distributions may have different driver/runtime
requirements. Do not guess the correct backend from an RTX model alone.

`llama-assets` enumerates **only official Windows x64 ZIP assets that
publish a GitHub SHA-256 digest**. Installation refuses a wrong tag,
unrecognized URL, absent digest, hash/size mismatch, ZIP path escape,
symlinks, duplicate file names or an unexpected archive layout.
Downloads are staged and verified, then moved to a versioned directory
atomically; an unrelated preexisting directory is never overwritten.
A matching installation can be reused without downloading again.
This verifies bytes against GitHub's published asset digest; it is not
an independent digital signature or a software security certification.

`--configure` is optional and only updates the explicitly supplied
**existing** PC-A YAML, setting `llm.server.executable` to the verified
binary and enabling lifecycle management. Without it the command
only installs the binary, so manual settings are unchanged. The install
command does **not** launch the binary and does **not** modify the local
GGUF selection. Use `artifex llm bootstrap` to fetch an absent selected
GGUF with the existing pinned download/verification flow. Re-run
`onboard dependencies` and then `deployment verify` to inspect
services after starting them.

ComfyUI itself and its CUDA/PyTorch/custom-node dependencies are **not**
auto-installed or upgraded by this step. Official ComfyUI distributions
have GPU-generation-specific variants, and replacing a working complex
ComfyUI tree can break custom nodes. Use an existing ComfyUI root with
`onboard renderer` for now, rather than installing into or modifying
the operator's current ComfyUI directory.

## Coordinated two-PC admission maintenance (PC-A)

After configuring the **same authenticated PC-B gateway URL** for both
`comfyui.base_url` and `render_nodes.primary`, and setting
`comfyui.gateway_token_env: ARTIFEX_RENDER_NODE_TOKEN` (with the actual
secret present in PC-A's environment), you can operate both durable
admission barriers from **one PC-A command**:

```powershell
# Read-only comparison; never pauses or changes either machine.
uv run artifex maintenance pair --config .\\config\\local.yaml

# Stop Artifex scheduling, drain existing production + ComfyUI queue,
# seal PC-A first, then seal and verify the authenticated PC-B gateway.
uv run artifex maintenance pair --config .\\config\\local.yaml `
  --apply --wait-seconds 600 --poll-seconds 5
```

The new command checks the actual PC-B `/v1/gateway/status` before
mutating anything on PC-A. An unavailable gateway, invalid authentication,
unexpected attestation or mismatched URL fails closed. It waits for the
existing controller's persisted in-flight Packs and generation attempts,
uses the same cross-process SQLite lock as Artifex `POST /prompt`, then
asks PC-B to atomically seal its gateway and reads its status again.

**Partial failures remain sealed on PC-A.** If the PC-B gateway times
out or its queue becomes busy after the PC-A fence closes, the command
reports `partial` with explicit recovery actions. It does NOT secretly
undo the controller's maintenance pause, discard GPU work, retry a restart,
or resume rendering. Once PC-B is repaired and its queue is empty, rerun
the exact `--apply` command to finish sealing.

After any separately performed, safe renderer maintenance, use the guarded
release command:

```powershell
uv run artifex maintenance pair --config .\\config\\local.yaml `
  --apply --release
```

Release requires BOTH fences sealed, controller still PAUSED, and a
**fresh passing real** production/repair workflow audit plus verified
ComfyUI idle queue. It opens PC-B first and verifies the status before
opening PC-A. If PC-B release fails, PC-A remains sealed. If PC-A release
fails, it reports `partial` for manual reconciliation. It never resumes
the controller automatically.

**Security boundary:** `both_sealed` means the Artifex controller
and the authenticated PC-B gateway are sealed, **not** that all local
PC-B applications are unable to write to ComfyUI's loopback socket.
Neither this tool nor a successful live audit can prove that unrelated
PC-B-local apps have stopped using that socket. Therefore reports retain
`restart_authorized=false` and
`external_loopback_clients_fenced=false`. No service restart is
performed. Also, the LAN gateway uses HTTP unless you intentionally
provide an encrypted transport; protect the private LAN or tunnel.

## Opt-in PC-B authenticated ComfyUI gateway (managed renderer only)

For a more controlled two-PC setup, Artifex can expose only an authenticated,
allowlisted ComfyUI **API** on PC-B. Unlike connecting directly to ComfyUI's
unprotected LAN port, this gateway forwards only the APIs Artifex needs:
`GET /system_stats`, `/object_info`, `/queue`, `/history`,
`/history/<prompt-id>`, `/view`; and authenticated
`POST /prompt`, `/free`, `/queue`, `/interrupt`.
It deliberately does not proxy arbitrary custom-node routes, a web UI,
WebSocket or other unaudited APIs.

Use the **same randomly generated strong token** in the environment of both
PCs, named `ARTIFEX_RENDER_NODE_TOKEN` (minimum 24 characters). Never
place it directly in YAML, logs or source control. The default HTTP protocol
does **not encrypt credentials**; restrict this to a trusted private LAN
with firewall rules, preferably an encrypted private tunnel if your network
is not trusted. Never expose the gateway port to the public internet.

PC-B sample override (paths and Python invocation are examples and must
match the actual installed Artifex-owned isolated ComfyUI):

```yaml
render_agent:
  bind_host: 0.0.0.0
  require_token: true
  token_env: ARTIFEX_RENDER_NODE_TOKEN
  comfyui_process:
    enabled: true
    executable: D:/AI/Artifex/isolated/.venv/Scripts/python.exe
    working_directory: D:/AI/Artifex/isolated/ComfyUI
    arguments: ["main.py", "--listen", "127.0.0.1", "--port", "8188"]
  gateway:
    enabled: true
    bind_host: 0.0.0.0
    port: 8191
    admission_path: data/render-gateway-fence.sqlite3
comfyui:
  base_url: http://127.0.0.1:8188
```

The backend must be an **Artifex-owned ComfyUI subprocess** and must have
exactly one explicit `--listen 127.0.0.1` argument. When an existing
ComfyUI is already healthy on that port, the protected configuration fails
closed rather than attaching to it. The gateway is part of
`uv run artifex render-node serve --config .\\config\\render-node.yaml`;
there is no separate Docker service. Do **not** open port 8188 to the LAN:
only allow the trusted PC-A to reach ports 8191 (gateway) and 8190
(attestation). Update Windows firewall rules accordingly. The gateway does
not configure Windows firewall rules itself.

PC-A must use the **gateway URL** for every ComfyUI API call, including the
configured primary render node. Example:

```yaml
comfyui:
  base_url: http://PC-B-LAN-IP:8191
  gateway_token_env: ARTIFEX_RENDER_NODE_TOKEN
render_nodes:
  primary: main
  nodes:
    main:
      base_url: http://PC-B-LAN-IP:8191
      attestation_url: http://PC-B-LAN-IP:8190
      attestation_token_env: ARTIFEX_RENDER_NODE_TOKEN
```

Authenticate to `GET http://PC-B-LAN-IP:8191/v1/gateway/status` with
`Authorization: Bearer <token>` to inspect the durable PC-B acceptance
state. PC-B also supports authenticated
`POST /v1/gateway/seal` (only when the real ComfyUI running/pending queues
are explicitly empty) and `POST /v1/gateway/release`.
The seal is stored on PC-B under its own SQLite transaction lock and survives
gateway restarts; a previously admitted mutation finishes before the seal
can commit. During sealing, other API mutations return HTTP 423.

**Limit:** Local applications running on PC-B can still send requests
directly to loopback `127.0.0.1:8188`, and Windows firewall/network
isolation has not been verified by the software. The gateway reports
`external_loopback_clients_fenced=false` and
`restart_authorized=false` **even when sealed**. Do not automatically
stop/restart ComfyUI on this evidence alone. The PC-A
`maintenance fence` is a separate, mandatory barrier for Artifex
submissions; an end-to-end restart sequence must coordinate both and
verify external isolation on the actual PCs.

## Atomic Artifex ComfyUI submission fence on PC-A

Artifex's controller now shares a durable **admission lock** across all
`ComfyUIClient.submit()` invocations. Its default SQLite barrier lives at
`data/comfy-submission-fence.sqlite3`, and can be moved by setting
`comfyui.submission_fence_path` in the controller's YAML. The daemon, CLI
commands and any other Artifex controller processes **must use the same
filesystem path** (including working directory) or they will not share this
barrier. The file is separate from the main state database, so a slow ComfyUI
submission does not lock unrelated Artifex DB updates.

After the current Pack completes, use this PC-A command for the **actual
atomic** Artifex admission change:

```powershell
# Read-only inspection; does not pause or change anything
uv run artifex maintenance fence --config .\\config\\local.yaml

# Pause controller production, observe DB + ComfyUI queue, double-check both
# under the same lock as every Artifex /prompt, then persist a closed fence
uv run artifex maintenance fence --config .\\config\\local.yaml `
  --apply --wait-seconds 600 --poll-seconds 5
```

The final check happens **after all previously started Artifex /prompt HTTP
requests finish**, while no *new Artifex requests* can pass the fence's
cross-process SQLite write lock. The sealed state persists over controller
restarts, so future submissions fail before any network POST. An active Pack,
nonterminal attempt, missing queue schema, offline ComfyUI or a late-appearing
prompt prevents sealing. A failed seal leaves the existing operator pause in
effect and reports why. It never cancels a GPU job.

Once the **actual** renderer and model/workflow checks succeed after any
maintenance, reopen admissions explicitly:

```powershell
uv run artifex maintenance fence --config .\\config\\local.yaml `
  --apply --release
uv run artifex resume --config .\\config\\local.yaml
```

Releasing admission does **not** resume the controller automatically and
requires a paused state; always verify renderer recovery before executing
either command.

**Not an external network firewall:** the fence protects Artifex clients
sharing its PC-A database path, not independent applications or personal
ComfyUI clients that POST directly to PC-B. Consequently
`external_comfyui_clients_fenced=false` and `restart_authorized=false`
remain hard-coded regardless of observed idle state. Never use this tool as
authorization to kill/restart an unowned or externally accessible ComfyUI.
Full automatic zero-interruption restart requires routing or blocking all
other writers to PC-B, plus owning the renderer child process.

## Controller-side maintenance drain — no interrupted GPU jobs

A future coordinated restart must first stop new Artifex production, let the
already-started Pack finish, and observe both persistent in-flight work and
ComfyUI's **real** pending/running queue. On **PC-A**, after configuring
the primary renderer endpoint and before maintenance:

```powershell
# Read-only inspection; leaves scheduler state untouched.
uv run artifex maintenance drain --config .\\config\\local.yaml

# Explicitly PAUSE new daemon work. Observe for up to 10 minutes and require
# two consecutive idle samples; this NEVER cancels a ComfyUI prompt.
uv run artifex maintenance drain --config .\\config\\local.yaml `
  --apply --wait-seconds 600 --poll-seconds 5
```

The command preserves an already-paused operator state and refuses to silently
override degraded/blocked/stopped state. It uses the configured
`render_nodes.primary` endpoint on PC-B, rather than any stale PC-A
`comfyui.base_url`. It counts persisted non-terminal generation attempts
and Packs in policy/generation/evaluation, plus ComfyUI
`queue_running`/`queue_pending`. If any source is unknown, busy or
persistently unfinished, the report stays blocked; it never starts or stops
Windows processes. If no work is observed in two consecutive samples, it
reports `observed_quiescent` but **still**
`restart_authorized=false`: external clients may submit directly to PC-B
after the check. A proper restart requires exclusive admission fencing of
*all* ComfyUI clients, not a queue snapshot or controller pause alone.

The `--apply` pause is intentionally durable: when maintenance is
finished, use `uv run artifex resume --config .\\config\\local.yaml`
only after the real PC-B service has recovered and deployment checks pass.
This command does not reset model settings, delete work, terminate or restart
ComfyUI, initiate a GPU generation or certify production. It is the first
step toward a safe unattended maintenance window.

## Safe runtime reconciliation after installing models or custom nodes

**PC-B can automatically wait for a fresh ComfyUI workflow inspection** rather
than guessing that an installed file is already available to the runtime.
After the actual renderer has refreshed its model choices or restarted through
its regular owner-managed lifecycle, run:

```powershell
# Read-only single check of live ComfyUI queue and production/repair workflows
uv run artifex onboard reconcile-renderer `
  --config .\\config\\render-node.yaml

# Read-only bounded observation of an external refresh/restart, up to 10 minutes
uv run artifex onboard reconcile-renderer `
  --config .\\config\\render-node.yaml `
  --wait-seconds 600 --poll-seconds 5
```

This command sends only ComfyUI `GET /queue` and `GET /object_info`
requests, never a GPU `/prompt`, `/interrupt`, `/queue` deletion, system
process signal, or Windows Task Scheduler restart. To be considered
`ready`, the current configured production **and** repair workflows must
pass the actual runtime audit and ComfyUI must explicitly return empty
`queue_running` and `queue_pending` lists. An unreachable endpoint,
malformed/missing queue fields, busy GPU, missing model/node or unknown loader
choices is reported with structured `next_actions` and a nonzero exit code.
Use `--allow-busy` only when you need to inspect workflow readiness while
other rendering work is in progress; it does **not** make a destructive restart
safe.

This is **automatic observation and verification**, not automatic process
restarting. A queue-empty snapshot is not a lock against new remote prompts:
Artifex cannot yet guarantee a zero-interruption restart from another process
without coordinated submission fencing. It consequently leaves that
operation to the already configured renderer owner, rather than unexpectedly
interrupting personal ComfyUI jobs. `production_qualified` remains false
until the separate real-machine qualification ladder succeeds.

## One-command safe renderer model preparation

After the initial PC-B configuration and a running ComfyUI endpoint, use the
new workflow-aware preparation command **on PC-B**. It audits the real
production/repair templates against ComfyUI `/object_info`, reads the
existing explicit model source bindings, resolves only exact missing model
filenames, and reports missing custom-node classes:

```powershell
# Read-only audit and preparation plan. Nothing is installed or rewritten.
uv run artifex onboard prepare-renderer `
  --config .\\config\\render-node.yaml `
  --sources .\\config\\model-sources.json
```

Model bindings in `--sources` are explicit role/filename/publisher choices
approved once through `onboard model-sources-add --approve` (see below).
No publisher is inferred from the filename. Missing model choices without
an approved binding remain **unresolved** and are shown in the report.
For approved choices, the tool verifies the Hub repository card/license,
locks the source revision and file path and creates a SHA-256/size-pinned
download plan without downloading in preview mode.

To install the approved, pinned **model files only** to a separate
Artifex-owned isolated ComfyUI directory, after reviewing the recorded terms:

```powershell
uv run artifex onboard prepare-renderer `
  --config .\\config\\render-node.yaml `
  --sources .\\config\\model-sources.json `
  --comfy-root "D:/AI/Artifex/isolated/ComfyUI" `
  --apply --accept-licenses
```

The path is illustrative; use the actual source directory inside a
verified `onboard comfy-install` installation. The existing installer
enforces that ownership receipt, exact fixed-commit source, expected file
size and SHA-256 and **never overwrites** an existing model or modifies an
unowned ComfyUI tree. Specify `--comfy-root` even without `--apply` to inspect
the Artifex-owned local model folder read-only. Already-installed, exact-SHA-256
verified weights appear under `existing_verified`, are removed from
`download_required`, and will **not be downloaded twice**. A corrupted,
non-file or symlinked destination is rejected rather than replaced. This
prevents a stale ComfyUI loader cache from triggering repeat multi-GB downloads. Installing models is not proof that the running
ComfyUI has loaded them: restart or rescan the renderer as appropriate,
then rerun the preview to verify actual loader availability. The tool's
`ready` is based only on the *live pre-install audit*, and output
`production_qualified` stays false.

Missing custom-node classes trigger **read-only** mapping suggestions from
the ComfyUI-Manager index by default; no suggested Python code is ever
downloaded, installed, pip-installed or executed by this command.
Suggestions are not endorsements or evidence of compatible, licensed code.
The index lookup may be disabled with `--no-discover-nodes` when offline.
Independently review and explicitly approve required third-party nodes
using the existing `onboard deps-draft` / `onboard deps-install`
controls; do not implicitly grant `--allow-custom-code`. This separation
avoids silently replacing working custom nodes and preserves the user's
existing ComfyUI installation.

## One-time operator-approved model source bindings (no repeated --repo)

Previously, `models-draft` required a manual list of Hugging Face
repositories every run. Artifex now persists an explicit *model role +
loader filename* to publisher mapping in a local JSON file, so the
renderer can make the same decision consistently without manual input.
Nothing is pre-trusted by default.

After checking the intended publisher and model card, **register each
known source once**:

```powershell
uv run artifex onboard model-sources-add `
  --role checkpoint `
  --name "your-ILXL.safetensors" `
  --repo "reviewed-publisher/reviewed-model" `
  --source-file "weights/your-ILXL.safetensors" `
  --rationale "Publisher and license checked by operator" `
  --approve
```

Those values are illustrative; they are **not** asserted to represent
a real model. To guard against a re-upload or changed filename mapping,
optionally register an independently known LFS digest with
`--sha256 64_HEX_DIGITS`. The `--source-file` option locks the
precise path within that publisher's repository, avoiding collisions
between multiple same-named files.

Saved bindings default to `config/model-sources.json`. Inspect or
revoke only the exact recorded role, name and repository:

```powershell
uv run artifex onboard model-sources-list --json
uv run artifex onboard model-sources-remove `
  --role checkpoint --name "your-ILXL.safetensors" `
  --repo "reviewed-publisher/reviewed-model" --confirm
```

**Next runs no longer need `--repo`**:

```powershell
uv run artifex onboard models-draft `
  --config .\\config\\local.yaml `
  --output .\\config\\generated-models.json
```

With no `--repo`, Artifex reads the saved source registry by default.
An alternative file can be supplied with `--sources PATH`, but
`--repo` and `--sources` cannot be combined. The previous
`--repo` workflow remains supported when manually reviewing a new
unregistered model.

Only an exact registered *role/filename* may use that publisher, even
when other registered publishers carry files with the same name.
Optional exact file path and expected SHA-256 are checked against the
fixed-commit Hugging Face LFS metadata, and mismatches are reported
instead of falling back to another file. Missing registration is
reported as unresolved, never silently substituted. Unknown loader
availability does not trigger a download. The registry is written
atomically, rejects duplicate mappings and avoids symlinked files.

This registry records an operator's **source-selection policy**, not
proof of upstream rights, licensing, safety, or model quality. Final
weight downloads still require explicit reviewed installation into
an Artifex-created isolated ComfyUI environment; the existing
`deps-install` safeguards and offline model hashing are unchanged.

## Start existing Windows services with one command per PC

A safe native Windows activation command combines scheduled-task ownership
checks, configuration/path validation and the existing *live* deployment
checks. It runs **locally** on each PC; it does not execute commands remotely,
alter the other machine, silently install packages, or generate GPU images.

After configuring both machines, on **PC-B (renderer)** run:

```powershell
# Preview: inspect registered task, current config and live connectivity
uv run artifex deployment activate --role renderer `
  --config .\\config\\render-node.yaml

# Opt-in: create the current-user scheduled task if absent, start it, then
# check live renderer readiness (10-minute maximum; default 90 seconds)
uv run artifex deployment activate --role renderer `
  --config .\\config\\render-node.yaml --apply --install-missing `
  --wait-seconds 600
```

Once PC-B is reachable, run the same two operations on **PC-A (controller)**:

```powershell
uv run artifex deployment activate --role controller `
  --config .\\config\\local.yaml
uv run artifex deployment activate --role controller `
  --config .\\config\\local.yaml --apply --install-missing `
  --wait-seconds 600
```

The preview does not register or start anything. `--apply` explicitly
activates only an **Artifex-owned** Windows Task Scheduler task, using the
Python environment, current directory and YAML file selected on that PC.
An already-running matching task is left alone, rather than spawning a second
daemon. If the installed task still refers to an old Python executable,
directory, or config, the command refuses to launch it. Use `--apply --replace`
to replace only an **Artifex-owned stopped task**; for a running task,
stop it safely before replacement. An unrelated scheduled task can never be
taken over. A missing task is only registered with `--install-missing --apply`;
the existing Windows logon autostart and restart policy is preserved.

After activation Artifex polls the **real** existing deployment checks:
LLM/ComfyUI, authenticated renderer, workflow and model dependencies for
PC-A; local renderer process readiness and config for PC-B. A Task Scheduler
`Running` status alone is **not proof of healthy production**. The command
returns a nonzero exit code and actual failing checks if readiness is not
established within `--wait-seconds` (0–600, default 90). Output always
sets `production_qualified=false`. Unlike
`deployment pair-smoke --confirm-render` or
`qualify archive-reproduce --confirm-render`, activation never submits
a GPU job. Model downloads, permission approvals, the 8-hour soak, and
the remaining strict qualification steps are still separate workflows.

## Automatic collection of completed Pack qualification

When the two Windows PCs are already operating and you have a qualification
session ID, `uv run artifex qualify collect <SESSION> --config .\\config\\local.yaml`
shows which genuine completed Packs qualify the required production tests,
without manually gathering IDs. Run the same command with `--apply` to
register only validated post-session evidence, without generating or restarting
anything. Repeat after a real retry, reboot recovery, backend outage recovery,
or verified Discord actions: pending stages can pass automatically when their
actual telemetry exists. Dedicated overnight and real archive replay evidence
remain separate, strict gates; this collector does **not** start services,
run GPU jobs or declare production-ready. See
[real-machine-qualification.md](real-machine-qualification.md).

## Unattended 8-hour evidence observer (first qualification gate)

Artifex can **observe an already-running production controller** for
an extended real interval from PC-A, without starting/stopping any
services, sending GPU jobs, altering existing ComfyUI or requiring
Docker. First qualify the two hosts with `deployment pair-check`
and execute `deployment pair-smoke --confirm-render`, then start
the normal Artifex production daemon using your existing setup.

**Recommended zero-copy workflow (PC-A, one command):** once
Artifex, the local LLM, PC-B render-node and ComfyUI are running,
start the bound 8-hour qualification with:

```powershell
uv run artifex qualify soak-run --config .\\config\\local.yaml
```

This command runs live Doctor checks, automatically creates a new
qualification session, generates a collision-resistant JSONL evidence
path under `qualification.evidence_dir`, logs progress samples to
stderr, and registers the `overnight_soak` stage PASS **only if** the
real observation lasts at least 8 hours (or a longer configured minimum)
and passes every evidence check. It emits a
machine-readable final JSON summary containing the session ID,
observation metrics, remaining qualification stages, and
`production_qualified=false`. On failure or interruption it never
registers a successful soak. It does **not** start the Artifex daemon,
send image-generation work, reboot Windows, restart services or
qualify the other 13 stages. The production daemon must already be
generating Packs. To resume qualification in an existing session
instead of creating a new one, add `--session-id <SESSION>`. A
configuration mismatch or unhealthy baseline is rejected *before*
waiting overnight. `--no-progress` suppresses periodic stderr logs,
and `--output` can select a new path within the configured evidence
directory. This command never overwrites evidence.

**Advanced/manual observer flow** (kept for forensic operation):

On **PC-A**, first create the production qualification session with
`uv run artifex qualify start --config .\\config\\local.yaml --json` and save
the returned `session_id`. Then open a separate terminal and launch the
read-only observer alongside the production process. The `--session-id`
argument binds its JSONL trace to that exact qualification run:

```powershell
uv run artifex qualify soak-observe `
  --config .\\config\\local.yaml `
  --output .\\data\\qualification\\soak-2026-10-08.jsonl `
  --session-id <SESSION> `
  --hours 8 --sample-seconds 300
```

A new file is required for each run; the observer refuses overwrite.
The default duration is `qualification.minimum_soak_hours` (8
hours), so `--hours` is optional. The defaults sample approximately
every five minutes. The process **really waits the configured time**
using monotonic elapsed timing. The observer does not fake or accelerate
time in production. On each sample it checks the LLM endpoint,
ComfyUI `/system_stats`, authenticated PC-B render-node attestation
(with required checkpoint/refiner/VAE/upscale hashes and NVIDIA GPU),
current GPU allocated/free VRAM metrics (when reported by ComfyUI),
local RAM usage and free disk space, and **read-only** SQLite counts of
stored agent events, error/critical events and finalized Packs.

Each measurement is fsynced as JSONL. On application crash, Ctrl+C,
Windows reboot or unexpected termination the file is **incomplete**
(no final record), and validation will never say the soak succeeded.
If the first sample fails, the observer stops immediately rather than
spending 8 hours in a broken environment. The monitoring process itself
does **not restart any services or automatically create Packs**;
the production controller must already be running.

Later, on PC-A, independently analyze a saved record:

```powershell
uv run artifex qualify soak-check `
  --config .\\config\\local.yaml `
  --evidence .\\data\\qualification\\soak-2026-10-08.jsonl
```

An acceptable record must contain 8 real elapsed hours, continuous
sampling with bounded gaps, matching monotonic and UTC clock times,
healthy LLM/ComfyUI/authenticated renderer checks, unchanged model
hashes, observed GPU/RAM/disk and SQLite counters, **at least three
new finalized Packs**, fresh agent telemetry, and **zero new error or
critical events**. Failure and diagnostic data are always retained,
with an actionable nonzero command exit.

The results include the recorded metrics and evidence-file SHA-256,
plus `ready_for_soak_review`. This flag **is not the same as a
qualification PASS**: `production_qualified` is always false.
When the independent review reports `ready_for_soak_review=true`,
register this trace to its original qualification session:

```powershell
uv run artifex qualify record <SESSION> overnight_soak `
  --status pass `
  --soak-evidence .\\data\\qualification\\soak-2026-10-08.jsonl `
  --config .\\config\\local.yaml
```

This shortcut calculates the SHA-256 rather than asking an operator to
copy it. Qualification **requires** the observed trace, matching session
ID, controller hostname, renderer ID, session start time, configuration
hash, pinned asset hashes and 8-hour live checks. It stores an absolute
evidence path, its SHA-256 and measured metrics; `qualify verify` checks
these again. A file that was moved, edited or replaced invalidates
the stage. Manually entered elapsed time, resource usage or fault counts
cannot produce PASS. Standalone `soak-observe` without `--session-id`
is still available for diagnostics, but its trace cannot qualify a stage.
Evidence must be saved under the configured `qualification.evidence_dir`.
The current metric observer uses a SQLite backend; if a non-SQLite
database is configured, it fails closed until a dedicated read-only
adapter is implemented.

**Operational caveat:** PC-B attestation includes model and LoRA
inventories. Collecting their hashes can increase disk I/O on PC-B,
so avoid very short sampling periods with large LoRA collections.
ComfyUI must expose numeric `vram_total` and `vram_free` per device
for a complete VRAM profile; unknown counters do not pass silently.

## Explicit gated real-image test after two-PC readiness

After PC-B `deployment pair-export` and PC-A `deployment pair-check`
succeed, PC-A can request **one actual production-workflow test image**:

```powershell
uv run artifex deployment pair-smoke `
  --config .\\config\\local.yaml `
  --renderer-report .\\data\\pair\\pc-b-evidence.json `
  --output .\\data\\pair\\pair-smoke-proof.json `
  --image-dir .\\data\\pair\\images `
  --confirm-render
```

`--confirm-render` is required to submit GPU work; without it, **nothing
is queued**. PC-A performs the full live, authenticated pair preflight
again, checks model/LoRA/GPU evidence, and fails closed **before**
submitting any generation if one prerequisite fails. When it passes,
the existing `deployment verify --render-smoke` runs the configured
production workflow on PC-B, inspects ComfyUI's queue/history, downloads
its actual output, checks that Pillow can fully decode the image,
and records the prompt ID, output location, image size, file size and
SHA-256 in an evidence JSON. The primary render-node configuration
controls API image transport and remote address even if old
`comfyui.base_url` or `comfyui.output_mode` settings remain local.

After **every attempted** render (including generation errors), PC-A
requests **fresh authenticated** PC-B attestation. The result checks
model checkpoint/refiner/VAE/upscale SHA-256 **and byte sizes**,
host/node identity, GPU presence, and inventory error/LoRA count
against the PC-B snapshot. Any post-render mismatch prevents
`ready_for_qualification=true`; an image test can succeed but
asset-stability verification can independently fail.

The summary reports `preflight_ready`, `render_attempted`,
`actual_render_verified`, `asset_stability_verified` and
`ready_for_qualification`. The full JSON evidence file is created
exclusively, **never overwritten**; even failed preflights and
render attempts produce a diagnostic record with a nonzero exit code.
An output from a previous run must not be reused as current evidence.

**This is not production acceptance.** The output always explicitly
sets `production_qualified=false`. Artifex still requires the separate
14-stage production qualification ladder, genuine multi-Pack testing,
restart/recovery checks, Discord controls when enabled and
the unattended overnight/8-hour run for Issue #40.

## Integrated two-PC readiness evidence (read-only, no image queue)

After completing initial per-host setup, verify **PC-B locally** and then
validate the evidence **from PC-A against the actual authenticated running
renderer**. This avoids mistaking a passing local mock or a report from a
different machine for two-host production readiness.

**PC-B** (renderer with ComfyUI and the GPU): ensure ComfyUI and the
authenticated Artifex render-node attestation service are already running.
From the Artifex repo on PC-B, run:

```powershell
uv run artifex deployment pair-export `
  --config .\\config\\render-node.yaml `
  --comfy-root "D:/AI/ComfyUI_windows_portable" `
  --output .\\data\\pair\\pc-b-evidence.json
```

The command checks native Windows, uv, NVIDIA GPU, selected ComfyUI
Python, actual `torch.cuda.is_available()`, configured production assets,
LoRA inventory, ComfyUI `/system_stats`, PC-B bind address and token
presence. It saves a small JSON report containing node ID, host,
GPU count, snapshot time, model SHA-256/size and both diagnostic results.
A GPU/asset inventory may take time because the existing render-node
attestation computes checksums. The report has no bearer-token values.
If some check fails, the file is still written but the exit code is nonzero,
so you can inspect the failure; an existing report is NEVER overwritten.

Copy the generated PC-B JSON report to **PC-A** (the LLM/controller PC)
using your preferred trusted file transfer. On PC-A, with ComfyUI's
LAN address and authenticated attestation URL set in
`config/local.yaml` and the bearer-token environment available, run:

```powershell
uv run artifex deployment pair-check `
  --config .\\config\\local.yaml `
  --renderer-report .\\data\\pair\\pc-b-evidence.json `
  --max-age-minutes 60
```

This combines the PC-A native Python/uv/LLM binary/GGUF inspection,
existing PC-A live `deployment verify` (ComfyUI REST, LLM health,
authenticated remote attestation and both actual workflow template
requirements) with a **fresh extra authenticated render-node request**.
It cross-checks that the physical host differs from PC-A, the
PC-B report is recent, node identity/host/OS and GPU count agree,
and model **SHA-256 and exact sizes** match the live renderer for the
production checkpoint, refiner, VAE and upscaler. It compares LoRA
counts and any remote inventory errors and requires an actual Torch/CUDA
probe in PC-B evidence. Mismatches and missing prerequisites appear as
separate JSON checks with concrete `next_action` suggestions and a
nonzero exit code. The report's timestamp is checked against both
staleness and clock skew. No model installation, daemon control,
configuration changes, or rendered image request occurs.

**Caveat:** this is a *preflight readiness* report only. It does **not**
prove that ComfyUI can actually generate the configured illustration or
that the two machines will remain stable unattended. Only the separate
explicit `artifex deployment verify --role controller --render-smoke`
(and the full issue #40 qualification ladder and endurance test) can
progress toward production qualification. A saved PC-B report can be
edited by its holder; therefore PC-A does not trust it alone but
independently obtains authenticated current renderer state.

## Draft missing checkpoint/VAE/upscaler/LoRA models from trusted Hub publishers

Artifex can prepare **checksum-pinned model manifests** for missing model
files without downloading multi-gigabyte weights during the discovery phase.
For each model, select one or more repositories whose publisher and license
you have reason to trust:

```powershell
uv run artifex onboard models-draft `
  --config .\\config\\local.yaml `
  --repo "publisher/illustration-models" `
  --repo "publisher/upscalers" `
  --output .\\config\\generated-models.json --json
```

The repository identifiers above are **examples**, not verified upstream
publishers. Use actual Hugging Face `owner/repository` IDs.
The repeated `--repo` option establishes an operator-provided **source
shortlist**; it is intentionally not a speculative filename search across
every uploader's models.

The command reads the current Artifex ComfyUI workflow audit, checks an
*exact* missing loader filename in the selected repositories, freezes each
model repository to the full immutable 40-character commit from Hugging
Face, queries `paths-info` at that commit, and only accepts a single
unambiguous file with LFS **SHA-256 and matching byte size**. It reads the
model-card license identifier and links to the pinned README. A mismatched
repo identity, missing license/card, gated repository, ambiguous filename,
missing/conflicting LFS metadata or unsupported model role is reported
as unresolved. No model weights are downloaded, no existing JSON is
overwritten, and no files are written unless at least one verified model
candidate is found.

For safety, ComfyUI loader choices that cannot be enumerated by the
server are reported as *unverifiable* and never automatically treated as
a missing file requiring download. Runtime paths needing special
subdirectories or unsupported detector loader conventions are also
left unresolved, not silently guessed.

The result uses the **same schema_version=1 JSON manifest** as custom
nodes. First preview the generated manifest:

```powershell
uv run artifex onboard deps-install `
  --manifest .\\config\\generated-models.json `
  --comfy-root "D:/AI/Artifex/tools/ComfyUI/comfyui-0123456789ab/ComfyUI"
```

To actually download and checksum-verify the pinned weights into the
Artifex-created *isolated* ComfyUI, explicitly add
`--apply --accept-licenses`; there is no
`--allow-custom-code` requirement for model-only manifests.
The installer re-verifies the complete downloaded payload against LFS
SHA-256 and size and refuses to overwrite any existing weight file.

**Security/rights limitation:** matching filenames and Hub LFS hashes prove
identity of Hub-hosted bytes, not that the shortlisted publisher is the
original rights holder, that training data is licensed, or that the
weights are safe. Git LFS metadata and model-card license declarations
must still be assessed as part of selecting trusted repositories. Also,
ComfyUI user-customized external model directories are not automatically
rewritten; use `workflow-audit` again after installation and server
reload to verify loader visibility.

## Automatically prepare a verified custom-node manifest draft

Artifex can now turn **unambiguous missing custom-node provider candidates**
into a prepared, checksum-pinned manifest without you collecting individual
GitHub commit hashes or ZIP checksums by hand:

```powershell
uv run artifex onboard deps-draft `
  --config .\\config\\local.yaml `
  --output .\\config\\generated-dependencies.json `
  --max-repositories 3 --json
```

This performs a read-only ComfyUI workflow audit, obtains ComfyUI-Manager's
node-provider candidates from one immutable manager commit, and looks up
each uniquely mapped provider's *actual GitHub default-branch commit*.
It then checks GitHub's license at that exact commit, downloads its
commit-pinned source ZIP **to a temporary folder** (never executing it),
validates archive structure and safe paths, and verifies that the license
file bytes in the archive match the pinned GitHub license response.
It records the ZIP's actual SHA-256, byte size, fixed commit, SPDX license
classification and immutable license URL in a manifest compatible with
`onboard deps-install`. No source code is installed by this command.

Candidates without a reliably recognized license identifier, matching
license bytes, safe Python node archive or unambiguous owner are
**excluded with a reason**. The command will not generate a misleading
empty manifest when no safe candidate is available. Existing output
files are not overwritten. Use `--manager-commit FULL_SHA` to pin a
specific ComfyUI-Manager index. Limits on repository count and archive
size bound network and disk usage.

The generated manifest is an **installation proposal**, *not*
automatic legal or security clearance. GitHub's SPDX classification
does not replace reviewing the actual license conditions, and a
published Python package can still contain malicious or incompatible
code. The final installation still requires
`onboard deps-install --apply --accept-licenses --allow-custom-code`
on an Artifex-owned isolated ComfyUI environment and must be followed
by a real ComfyUI restart and `workflow-audit`.

Model weights are intentionally separate: Artifex cannot safely infer
an authoritative checkpoint/VAE/LoRA download URL from a filename
alone. Unknown models remain in the draft's unresolved report rather
than causing automatic downloads from unverified third-party links.

## Resolve missing node packages, then install reviewed pinned dependencies

`workflow-audit` identifies what the running ComfyUI is missing.
`deps-resolve` adds a **read-only candidate lookup** using two
ComfyUI-Manager index files from the **same pinned Git commit**:

```powershell
uv run artifex onboard deps-resolve `
  --config .\\config\\local.yaml --json
```

Artifex automatically requests the current ComfyUI-Manager main commit SHA
and fetches `extension-node-map.json` and `custom-node-list.json` at
that immutable revision. You can supply `--manager-commit <FULL_SHA>`
for reproducibility. The output shows missing class types, candidate GitHub
repositories with matches corroborated by the same pinned index, and
unresolved classes. A suggested repository is **not** necessarily safe,
licensed, compatible with your ComfyUI version, or the unique provider of
a node class. The planner never executes third-party code.

For **installation**, use a reviewed `schema_version: 1` JSON manifest
with exact source URLs, SHA-256 hashes, byte sizes and license details.
The following model record is an illustrative structure—not a real
downloadable asset or a license recommendation:

```json
{
  "schema_version": 1,
  "artifacts": [
    {
      "id": "chosen-base-model",
      "kind": "model",
      "name": "model.safetensors",
      "model_folder": "checkpoints",
      "url": "https://huggingface.co/owner/repository/resolve/0123456789abcdef0123456789abcdef01234567/model.safetensors",
      "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "size_bytes": 123456,
      "license_id": "CHECK-ACTUAL-LICENSE",
      "license_url": "https://huggingface.co/owner/repository"
    }
  ]
}
```

For a custom node, use `"kind": "custom_node"`, a simple destination
`"name"`, `"repository": "owner/repository"`,
`"commit": "<FULL_40_CHARACTER_SHA>"`, and a matching exact
`"url": "https://codeload.github.com/owner/repository/zip/<FULL_SHA>"`
plus the exact archive size, SHA-256 and license information.
**Never treat illustrative hash values as verified hashes.** Artifex
does not fabricate those facts from a registry suggestion. The
operator must review and pin dependency source and license metadata
before introducing third-party code.

Preview the manifest safely (the default performs **no download**):

```powershell
uv run artifex onboard deps-install `
  --manifest .\\config\\approved-dependencies.json `
  --comfy-root "D:/AI/Artifex/tools/ComfyUI/comfyui-0123456789ab/ComfyUI"
```

To perform installation, explicitly accept the licenses you reviewed:

```powershell
uv run artifex onboard deps-install `
  --manifest .\\config\\approved-dependencies.json `
  --comfy-root "D:/AI/Artifex/tools/ComfyUI/comfyui-0123456789ab/ComfyUI" `
  --apply --accept-licenses
```

Installing a custom-node ZIP also requires `--allow-custom-code`,
because its Python is imported by ComfyUI on a later restart.
The installer only writes into **Artifex-created isolated ComfyUI
directories**, never into an existing user ComfyUI install. It refuses
to overwrite any model or node. Model files require a full GitHub
release or pinned Hugging Face revision and expected SHA-256/size;
node ZIPs require a full GitHub commit URL and the same verification.
All bytes are downloaded and staged before placement. No arbitrary
`install.py`, pip requirements, or shell scripts are executed.
Third-party custom nodes may still require explicitly managed Python
dependencies; their mere placement does not guarantee functionality.
Once they are placed, restart the isolated ComfyUI separately and
repeat `workflow-audit` and `deployment verify` to ensure runtime
dependencies truly load.

**Current limit:** `deps-resolve` automatically finds likely GitHub
origins but does not generate a trusted SHA-256/size/license manifest
from ambiguous community index data. Artifex must not convert an
unverified suggestion into silent executable installation. Model
sources are not inferred from filenames, which are ambiguous.

## Isolated ComfyUI setup and exact workflow dependency audit

To **avoid changing your existing ComfyUI installation**, Artifex can download
a pinned Comfy-Org/ComfyUI source snapshot into a separate, versioned directory.
Use a full 40-character commit SHA from the official
[ComfyUI repository](https://github.com/Comfy-Org/ComfyUI).
First, download source only (no Python packages executed):

```powershell
uv run artifex onboard comfy-install `
  --commit <FULL_40_CHARACTER_COMMIT_SHA> `
  --output-dir "D:/AI/Artifex/tools/ComfyUI"
```

The command never overwrites an existing target, never touches your original
ComfyUI tree, and refuses unsafe ZIP paths, symlinks and oversized archives.
The content SHA-256 is reported and saved with its requested commit.
GitHub's pinned ZIP URL over HTTPS does **not** independently prove archive
authenticity. If you have independently verified the exact ZIP SHA-256, supply
`--archive-sha256 <SHA256>` to enforce it.

For an **entirely new environment**, add `--install-deps` plus an explicit
PyTorch backend compatible with your NVIDIA driver, for example:

```powershell
uv run artifex onboard comfy-install `
  --commit <FULL_40_CHARACTER_COMMIT_SHA> `
  --output-dir "D:/AI/Artifex/tools/ComfyUI" `
  --install-deps --torch-backend cu130
```

Supported choices are `cpu`, `cu126`, `cu128`, and `cu130`.
This uses `uv` to create an isolated ComfyUI `.venv` and installs PyTorch,
TorchVision, TorchAudio and the pinned source's `requirements.txt` within
the **new** folder only. The selected Python interpreter defaults to the one
running Artifex, or you can provide `--python PATH`. PyTorch releases are
resolved when installing, not reproducibly locked in this slice.
The command performs **network installation of third-party Python packages**
only with `--install-deps`, so use a trusted pinned ComfyUI commit.

After installation, verify the Python/CUDA compatibility with
`onboard dependencies --role renderer --comfy-root PATH --probe-torch`.
An installed source tree alone is **not** a fully working ComfyUI; CUDA
availability, model files, custom nodes and extensions must still be verified.
No existing ComfyUI directories, weights, custom nodes or Python environments
are replaced, copied or mutated.

### Audit exactly what the actual Artifex workflows need

After starting your existing or newly installed ComfyUI instance, run a
read-only audit against its `/object_info` API:

```powershell
uv run artifex onboard workflow-audit `
  --config .\\config\\local.yaml --json
```

The report separately identifies missing **ComfyUI node class types**, models
not present in the loader's available choice list, and **unverifiable models**
when ComfyUI exposes no choice list. Unknown is never reported as a confirmed
model match. Both default production and repair workflow requirements are
audited, including configured checkpoint, VAE, refiner, upscaler and any
referenced detector and LoRA requirements. You may supply
`--custom-nodes-root PATH` to list local custom-node directories, but
folder names are **not** authoritative mappings of runtime node classes.

The audit does not queue image jobs and does not automatically install unknown
third-party custom-node repositories or download weight files: those require
specific provenance, version compatibility and license information. It returns
a nonzero exit code if any required node/model is missing or unverifiable.
Once resolved, run `deployment verify` and then explicitly
`deployment verify --render-smoke` for the real image generation check.

## Native first-run discovery (no network or background changes)

You can inspect each machine **before starting ComfyUI, llama.cpp or the
other PC**. These commands only read bounded local paths, selected GGUF
metadata and environment-variable *presence*, not secret values:

**On PC-B**, provide the existing ComfyUI or Windows Portable root:

```powershell
uv run artifex onboard inspect --role renderer `
  --comfy-root "E:/ComfyUI_windows_portable" --json
```

The report lists the actual ComfyUI `main.py` directory, detected embedded
or virtual-environment Python executables, checkpoint/VAE/upscale filenames,
and the local LoRA directory. It deliberately does not scan entire drives,
follow `extra_model_paths.yaml` references, download assets, or guess which
of several Python installations or checkpoints is correct.

Once a specific checkpoint is chosen, **one command** can generate PC-B's
renderer YAML without having to type Python and working-directory paths:

```powershell
uv run artifex onboard renderer `
  --comfy-root "E:/ComfyUI_windows_portable" `
  --checkpoint-path "E:/ComfyUI_windows_portable/ComfyUI/models/checkpoints/your-ILXL.safetensors" `
  --node-id rtx3060 `
  --bind-host 0.0.0.0 `
  --output .\\config\\render-node.yaml
```

The command automatically configures managed ComfyUI startup only when
exactly **one** local Python environment is detected. With multiple candidates,
supply `--comfy-exe`; with an independently managed ComfyUI process, pass
`--external-comfy` instead. You may also specify `--refiner-path`,
`--vae-path`, `--upscaler-path`, `--lora-dir` (repeatable),
`--comfy-port`, `--port`, `--update` and `--force`.
Model roles are **never** assigned based on guesses about filenames.
Additional external model roots remain operator-configurable.

**On PC-A**, inspect the local LLM executable, chosen GGUF, `uv`, Python,
and render-node token presence:

```powershell
uv run artifex onboard inspect --role controller --json
```

You can create the controller YAML **while PC-B is turned off**, avoiding
the earlier setup dependency on an already-running ComfyUI:

```powershell
uv run artifex setup `
  --comfy-url http://192.168.1.50:8188 `
  --render-node-id rtx3060 `
  --production-checkpoint your-ILXL.safetensors `
  --offline --skip-llm-download `
  --output .\\config\\local.yaml
```

The `--offline` flag only skips the ComfyUI probe; the output reports
`comfyui_probed: false`, not a fabricated successful connection.
`--skip-llm-download` is optional and shown to keep this initial step
network-free; remove it to use the existing resumable pinned GGUF bootstrap.
No installer or command registers Windows startup tasks implicitly. Run
`artifex deployment verify` when services are online and opt into the
real `--render-smoke` when ready to allocate GPU resources.

## PC-B: renderer

ComfyUI must listen on an address reachable from PC-A. Keep the ComfyUI port restricted to the trusted LAN/firewall scope.

You can generate the PC-B configuration without hand-editing YAML or having
ComfyUI running yet. **Replace these sample paths/IPs with the actual PC values**:

```powershell
uv run artifex render-node configure `
  --node-id rtx3060 `
  --bind-host 0.0.0.0 `
  --port 8190 `
  --comfy-url http://127.0.0.1:8188 `
  --checkpoint-path "E:/ComfyUI/models/checkpoints/my-model.safetensors" `
  --refiner-path "E:/ComfyUI/models/checkpoints/my-refiner.safetensors" `
  --vae-path "E:/ComfyUI/models/vae/my-vae.safetensors" `
  --upscaler-path "E:/ComfyUI/models/upscale_models/my-upscaler.pt" `
  --lora-dir "E:/ComfyUI/models/loras" `
  --output .\\config\\render-node.yaml
```

Later you can update only the changed settings **without losing the rest**:

```powershell
uv run artifex render-node configure `
  --output .\\config\\render-node.yaml `
  --checkpoint-path "F:/AI/checkpoints/new-model.safetensors" `
  --lora-dir "F:/AI/loras" `
  --bind-host 192.168.1.60 `
  --update
```

Repeat `--lora-dir` to configure multiple folders. With `--update`, providing
this flag replaces the whole LoRA roots list; omitting it preserves the list.
`--force` replaces the complete file, whereas `--update` preserves unrelated
settings. Set the token value through an environment variable, not these flags.
Restrict the ComfyUI and attestation ports to the trusted LAN.

Alternatively, edit a renderer config YAML directly, for example:

```yaml
render_agent:
  node_id: rtx3060
  bind_host: 0.0.0.0
  port: 8190
  token_env: ARTIFEX_RENDER_NODE_TOKEN
  require_token: true
  asset_paths:
    production_checkpoint: D:/AI/ComfyUI/models/checkpoints/production.safetensors
    refiner_checkpoint: D:/AI/ComfyUI/models/checkpoints/anime-refiner-beta1.1.safetensors
    vae: D:/AI/ComfyUI/models/vae/pppanimixVAE_ilxl.safetensors
    upscale_model: D:/AI/ComfyUI/models/upscale_models/4xRealisticrescaler_100000G.pt
  lora_roots:
    - D:/AI/ComfyUI/models/loras

comfyui:
  base_url: http://127.0.0.1:8188
```

### Optional: manage ComfyUI itself on PC-B

PC-B's `render-node serve` can now start and supervise the **real ComfyUI
application**, rather than requiring you to open it manually after login.
This mode is **off by default**, so existing external ComfyUI installations
are not disrupted. Run the following example on PC-B after replacing the paths:

```powershell
uv run artifex render-node configure `
  --output .\\config\\render-node.yaml `
  --comfy-exe "E:/ComfyUI/.venv/Scripts/python.exe" `
  --comfy-workdir "E:/ComfyUI" `
  --comfy-arg=main.py `
  --comfy-arg=--listen `
  --comfy-arg=0.0.0.0 `
  --comfy-arg=--port `
  --comfy-arg=8188 `
  --update
```

For the Windows portable ComfyUI distribution, instead point `--comfy-exe`
to its embedded Python executable and `--comfy-workdir` to the corresponding
ComfyUI working directory; adjust arguments to match that distribution.
The CLI accepts any number of `--comfy-arg=VALUE` arguments. No drive letter,
Python path, ComfyUI version, checkpoint or GPU device is hard-coded.
The `comfyui.base_url` in the PC-B YAML should refer to the **local**
`http://127.0.0.1:8188` endpoint; PC-A separately uses PC-B's LAN IP,
for example `http://192.168.1.50:8188`. Keep ComfyUI reachable on the
trusted LAN by setting its `--listen` appropriately and restricting access
with Windows Firewall.

After configuration, the usual command is unchanged:

```powershell
uv run artifex render-node serve --config .\\config\\render-node.yaml
```

When managed mode is enabled, the command first checks whether ComfyUI is
already healthy. If so, it uses that instance **without killing it**. Otherwise
it starts the configured executable and waits for `/system_stats` readiness.
It monitors only the child it owns, applies a bounded restart policy after
crashes/sustained failures, writes logs to `data/logs/comfyui.log`, and
stops its owned child when the render-node service shuts down normally.
A fatal supervisor error also shuts down the attestation endpoint so Windows
Task Scheduler can retry the entire task within its own bounded restart policy.
If ComfyUI is externally owned and stops responding, Artifex fails with a
diagnostic instead of killing/replacing that external process.

`render-node preflight` checks **running** ComfyUI. On first-time managed
installations, start `render-node serve` before running preflight in a second
terminal. If management is disabled, start ComfyUI yourself first.

To turn managed mode back off without replacing the other settings:

```powershell
uv run artifex render-node configure `
  --output .\\config\\render-node.yaml `
  --disable-comfy-management --update
```

The Windows `startup install --role renderer` task created previously still
launches `render-node serve` and therefore now starts managed ComfyUI
automatically **on user logon**, with no task re-registration needed when
only the YAML content changes. The configured executable and arguments are
operator-owned; Artifex does not install Python or ComfyUI binaries for you.

Set the same strong token on PC-A and PC-B through the `ARTIFEX_RENDER_NODE_TOKEN` environment variable. Do not put the token in YAML.

After ComfyUI is running, run the **read-only native-Windows preflight** on PC-B
(or start `render-node serve` first when ComfyUI is Artifex-managed).
It checks NVIDIA GPU detection, required checkpoint/refiner/VAE/upscaler hashes,
LoRA inventory completeness, the bearer-token environment variable,
LAN binding and the locally running ComfyUI endpoint. Failures produce
`ready: false` with a nonzero exit code:

```powershell
uv run artifex render-node preflight --config .\\config\\render-node.yaml --json
```

Inspect the full local evidence when troubleshooting:

```powershell
uv run artifex render-node attest --config .\config\render-node.yaml --json
```

Then keep the lightweight attestation endpoint running:

```powershell
uv run artifex render-node serve --config .\config\render-node.yaml
```

The agent exposes only health and authenticated asset/LoRA attestation data. Image generation remains on ComfyUI.

Normal attestation queries are cached for 60 seconds to avoid repeatedly hashing large
model libraries. Strict qualification uses `?fresh=1` and bypasses that cache. If
the LoRA scan is incomplete or a root becomes temporarily inaccessible, Artifex
records the error but does **not** interpret missing inventory entries as deletions.
Keep ports 8188/8190 restricted to the trusted LAN; the bearer token is not
encrypted over plain HTTP. Use a TLS tunnel or reverse proxy outside that scope.

## PC-A: controller

The recommended first-run path is the setup command. It probes the renderer's
ComfyUI endpoint, writes only the controller-specific overrides, configures API
output retrieval and the render attestation endpoint, and downloads the selected
LLM unless explicitly skipped:

```powershell
uv run artifex setup `
  --comfy-url http://192.168.1.50:8188 `
  --render-node-id rtx3060 `
  --llm-models-dir D:/AI/models/llm `
  --output .\config\local.yaml
```

To reconfigure PC-A without replacing unrelated settings, supply `--update`.
You can change its ComfyUI/attestation IP, checkpoint name, model directory,
LLM endpoint, semantic model location or downloaded image directory:

```powershell
uv run artifex setup `
  --output .\\config\\local.yaml `
  --comfy-url http://192.168.1.60:8188 `
  --attestation-url http://192.168.1.60:8190 `
  --production-checkpoint my-model.safetensors `
  --render-cache-dir "E:/Artifex/render-cache" `
  --llm-models-dir "E:/Models/GGUF" `
  --llm-url http://127.0.0.1:8899 `
  --skip-llm-download `
  --update
```

If no IP has changed, you may omit `--comfy-url` and `--render-node-id`
with `--update`; the saved node and URL will be reused. You can add
`--semantic-model-path` if qualification requires a separately stored local
semantic model. Changing PC-B's LAN IP or port also requires updating PC-A's
matching endpoint and restarting the affected services.

The default attestation URL is derived from the same host on port 8190. Override
it with `--attestation-url` when PC-B uses another port. Existing config files
are never silently overwritten; use `--update` to merge changes safely or
`--force` to replace the full YAML. Use
`--skip-llm-download` when the model is already managed separately.

The generated config is intentionally a small override, not a copy of every
Artifex default. A production checkpoint still needs to be selected explicitly
because automatically choosing one from a machine with multiple ILXL
checkpoints would be unsafe. Setup is safe to re-run with identical options if
the LLM download was interrupted: an identical generated YAML is reused;
an operator-modified configuration is not overwritten without `--update` or `--force`.

Configure the primary renderer manually when needed:


```yaml
render_nodes:
  primary: rtx3060
  nodes:
    rtx3060:
      type: comfyui
      enabled: true
      base_url: http://192.168.1.50:8188
      output_mode: api
      download_dir: data/render-cache
      attestation_url: http://192.168.1.50:8190
      attestation_token_env: ARTIFEX_RENDER_NODE_TOKEN

production:
  checkpoint: production.safetensors
```

`output_mode: api` makes PC-A retrieve completed images through ComfyUI's `/view` API. The renderer's output directory therefore does not need to be mounted on PC-A.

The legacy `comfyui:` configuration remains supported. When `render_nodes.primary` is configured, the primary node overlays the connection/output transport fields while existing sampling/workflow settings remain in `comfyui:`.

## LLM bootstrap on PC-A

The default local model profile is currently Spark-X2.5-4B-Heretic-jp Q8_0. It is a default, not a hard-coded lock.

```yaml
llm:
  bootstrap:
    enabled: true
    auto_download: true
    profile: spark-x2.5-4b-heretic-jp-q8_0
    models_dir: D:/AI/models/llm
```

Check or download it explicitly with:

```powershell
uv run artifex llm bootstrap --config .\config\local.yaml --json
```

Downloads use a `.part` file and HTTP Range resume. For the default pinned
Hugging Face model, bootstrap obtains the expected SHA-256 from Hugging Face's
resolve headers, checks the complete model against it, and stores a local
source-bound integrity receipt for offline reuse. A custom direct URL profile
must specify `sha256`. A corrupt existing GGUF is rejected rather than reused;
`artifex llm bootstrap --force` can replace it while preserving the last
working copy until the new download passes verification.

`artifex daemon` also runs the bootstrap automatically before production when
the local llama.cpp backend has `auto_download: true`. Bootstrap installs only
the GGUF; to automatically manage the server executable itself, enable
`llm.server.enabled` using the procedure below. Otherwise continue to
run the server separately at the configured local endpoint.

A different model can be selected by adding another bootstrap profile or by using a local profile; Artifex is not tied permanently to Spark.

## Managed llama.cpp on PC-A (optional, recommended for unattended operation)

Previously the GGUF download did not start `llama-server`. Artifex can now
**launch, check and supervise** a local llama.cpp server as part of
`artifex daemon`. Choose the `llama-server.exe` installed on PC-A; the
program deliberately does **not** download or execute an arbitrary binary.

```powershell
uv run artifex setup `
  --output .\\config\\local.yaml `
  --llama-server-exe "E:/tools/llama.cpp/llama-server.exe" `
  --llama-device CUDA0 `
  --llama-gpu-layers auto `
  --skip-llm-download `
  --update
```

Passing `--llama-server-exe` enables managed mode. `--llama-device` uses
the device identifier accepted by your installed llama.cpp release; check
`llama-server --list-devices` on that machine. `--llama-gpu-layers`
accepts `auto`, `all` or a nonnegative number. All values remain editable
under `llm.server` in the operator YAML, including disabling the feature.
The inference model is still selected through the independent `llm.bootstrap`
profile, so model choice is not locked to Spark.

With management enabled, `artifex daemon` uses the selected local GGUF,
`llm.model` as the served model alias, the configured context length and
the port from `llm.base_url`. It binds **only to 127.0.0.1/localhost or
IPv6 loopback**; remote/cloud LLMs are never launched or killed. If a
compatible server with the expected model alias is already healthy at that
address, Artifex uses it **without taking process ownership**. Otherwise
it launches its own process, waits for HTTP readiness, and retries bounded
crashes or sustained unhealthiness. The manager stops only the child it
started when Artifex exits; it never kills a third-party process.

By default `llm.server.enabled: false` protects users who manage llama.cpp
separately, and allows third-party OpenAI-compatible providers. You must
install the llama.cpp binary yourself or point the executable field to one
already on PATH. Startup failures surface the configured log path (default
`data/logs/llama-server.log`). If the target port is held by an incompatible
server, change `llm.base_url` or align the model alias; Artifex will not
terminate that existing server. On PC-A choose GPU offload according to
available VRAM; PC-B's ComfyUI GPU remains independent.

## Remote LoRAs

PC-B's attestation scans the configured LoRA roots and returns filename, relative ComfyUI asset name, safetensors metadata and SHA-256. PC-A stores only the inventory/provenance.

New or changed remote LoRAs return to discovery/validation. Missing LoRAs are disabled after a successful inventory refresh. A network outage alone does not mark remote LoRAs as deleted.

Generation submits the renderer-relative LoRA name to ComfyUI, so the LoRA file itself does not need to exist on PC-A. While scheduling remains primary-only, the resolver excludes LoRAs that are present only on a secondary renderer or on the controller.

## More render nodes

The configuration is intentionally a node map rather than one fixed ComfyUI URL:

```yaml
render_nodes:
  primary: rtx3060
  nodes:
    rtx3060:
      base_url: http://192.168.1.50:8188
      attestation_url: http://192.168.1.50:8190
    future-gpu:
      enabled: false
      base_url: http://192.168.1.60:8188
      attestation_url: http://192.168.1.60:8190
```

Current production dispatch uses the configured primary node. The node model deliberately leaves room for later scheduling/failover across multiple renderers without changing Pack, workflow or provenance formats.

## PC-A to PC-B end-to-end preflight

After configuring both PCs, starting ComfyUI and the PC-B render-node service,
and starting your local LLM server on PC-A, run the **read-only controller
preflight**:

```powershell
uv run artifex preflight --config .\\config\\local.yaml --json
```

It reads the saved URLs and paths (you do not have to reenter any IP addresses)
and checks:

- PC-A can connect to ComfyUI's `/system_stats` on PC-B
- PC-A has a bearer token and can obtain authenticated renderer attestation
- PC-B's node ID, Windows/NVIDIA GPU evidence, inventory status and timestamps
- PC-A's checkpoint/refiner/VAE/upscaler filenames match the PC-B inventory
- The local llama.cpp/OpenAI-compatible server is running and, when bootstrap
  is enabled, the selected GGUF file exists on PC-A

The response lists each check with a `ready` flag. The command exits nonzero
if any check fails, but **does not download models, change configuration, start
ComfyUI or submit image generation jobs**. It is a connectivity/prerequisite
diagnostic, not the full quality/long-running qualification. Token values are
never printed. Missing files must be fixed with the respective PC-specific
`setup --update` / `render-node configure --update` commands.

## Windows autostart after sign-in

On each PC, install its own **opt-in per-user Windows Task Scheduler task**.
No Docker, Windows service account, administrator privileges, or saved Windows
password is required. Run these commands on the actual PC, from the Artifex
project/working directory, using the same `uv` environment in which Artifex
was installed.

**PC-A:**
```powershell
uv run artifex startup install --role controller --config .\\config\\local.yaml
uv run artifex startup status --role controller --json
```

**PC-B:**
```powershell
uv run artifex startup install --role renderer --config .\\config\\render-node.yaml
uv run artifex startup status --role renderer --json
```

The tasks launch the existing project's Python environment directly
(`python -m artifex.cli daemon` on PC-A and `python -m artifex.cli render-node serve`
on PC-B), using absolute configuration paths and the installation-time
working directory. User-specific IPs, local paths and environment settings
remain configurable; no machine-specific address or GPU index is baked into
the task implementation.

Task Scheduler starts the task **on that Windows user's next sign-in**, not
before sign-in or immediately upon registration. A failed task is retried at
one-minute intervals, up to ten retries by default (configure
`--restart-count`). Concurrent starts of the same scheduled task are
ignored, and there is no three-day run-time timeout. The restart policy is
bounded; persistent errors still require operator attention.

Use `--replace` to update a previously installed Artifex task after moving
the repository, changing Python environments, or changing its config path.
Normal changes *inside* the existing YAML do not require re-registering a
task; restart the process to apply them. Example:

```powershell
uv run artifex startup install --role controller --config .\\config\\local.yaml --replace
uv run artifex startup uninstall --role controller
```

The task registration never stores Discord or render-node tokens in command
arguments or XML. **Environment variables set only in the current PowerShell
window do not automatically become available to a future login task.**
Provision credentials to the intended user through an appropriate persistent
secret/environment mechanism, and restrict access accordingly. PC-A and PC-B
must each have their required credentials available to the scheduled process.
The CLI refuses to overwrite or remove unrelated tasks with the same name.

The controller task supervises PC-A's Artifex daemon; the renderer task
runs the render-node attestation endpoint and, **when explicitly configured**,
also launches and supervises ComfyUI itself using `render_agent.comfyui_process`.
When managed ComfyUI is disabled, configure its own startup separately.
This logon-triggered configuration also does not guarantee execution
before anyone signs in; a pre-login service deployment has different Windows
account/GPU-session and secret-handling requirements.

## One-command deployment verification on each PC

The `deployment verify` command aggregates the existing role-specific
preflight and (on the controller) validates the configured generation and
repair workflow's required ComfyUI node types and model choices. It also
inspects the Windows user-logon startup task. **Normal verification is
read-only:** it does not submit a prompt, download files, register tasks,
start services or mark real-machine qualification stages complete.

With ComfyUI and render-node already started on PC-B, run on **PC-B**:

```powershell
uv run artifex deployment verify --role renderer --config .\\config\\render-node.yaml --json
```

With the local LLM endpoint available, run on **PC-A**:

```powershell
uv run artifex deployment verify --role controller --config .\\config\\local.yaml --json
```

Both commands return a structured list of passing/failing checks and a
nonzero exit code when an essential prerequisite fails. Windows startup
registration is reported as an optional observation unless you add
`--require-autostart`. This allows first-run service setup to be verified
before you register a persistent startup task.

### Explicit real image transport smoke (PC-A only)

After the read-only deployment verification passes, a **single actual
ComfyUI workflow** can be submitted from PC-A:

```powershell
uv run artifex deployment verify --role controller `
  --config .\\config\\local.yaml `
  --render-smoke `
  --json
```

This step exercises the selected `comfyui.default_template` with the
configured production checkpoint, refiner, VAE, upscaler and sampling
parameters. It submits to PC-B's ComfyUI `/prompt`, waits for `/history`,
downloads an image to PC-A using `/view`, and verifies the downloaded
image can be completely decoded. The report includes the prompt ID,
download path, image dimensions, byte count and SHA-256. Evidence defaults
to a unique folder under `data/render-cache/deployment-smoke/`, or below
the configured render-cache folder. No existing operator output is
overwritten. `--output-dir` can choose another location; optional
`--width` and `--height` can request a different initial canvas size.
Without them the production dimensions are used. Full production
workflows may upscale the output beyond the initial canvas size.

The smoke probe does **not** test character-LoRA selection, full
Artifex Pack production, editorial/Discord publication, continuity or
eight-hour unattended recovery. Those remain in the strict qualification
ladder in `docs/real-machine-qualification.md`. Smoke images are not
automatically published, nor is smoke success recorded as a qualification
pass. It is intentionally opt-in because it consumes GPU memory and time.
If network/model/workflow prerequisites fail, **no GPU prompt is queued**.
Use `--render-smoke` only when PC-B is ready for a real job.

## Qualification

Complete the PC-B preflight first; then on PC-A run `artifex setup`,
configure the explicit production checkpoint and controller-local semantic
assets, start the local LLM server and run `artifex doctor`. PC-B preflight
does not prove network reachability from PC-A; the controller must check LAN
connectivity and the remote token separately.

For the split topology, qualification records controller-local hashes and primary-renderer attestations in one session. Final verification re-fetches the remote attestation and rejects model drift.

The controller no longer needs a local filesystem path to the renderer checkpoint/VAE/upscaler. The primary renderer must expose authenticated attestation evidence for those assets.
