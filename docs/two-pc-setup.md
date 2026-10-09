# Two-PC Artifex topology

Artifex supports a split native-Windows deployment where the controller and renderer are separate machines.

- **PC-A / controller**: Artifex, local LLM, research/planning, SQLite, archive, evaluation, Discord/Patreon integration.
- **PC-B / render node**: ComfyUI, production checkpoint, refiner, VAE, upscaler, LoRAs and the lightweight Artifex render-node attestation server.

No Docker, WSL, shared SMB output folder, or duplicate LoRA/model copy on PC-A is required.

## First-run guide before either PC has a YAML (read-only)

You can inspect **PC-A or PC-B without creating a config file, starting a
service, changing a task or generating an image**. All paths, URLs, model
locations and machine addresses remain configurable; the guide never guesses
the IP address of PC-B, ComfyUI root, GPU ownership or a checkpoint.

**On PC-B, with the existing ComfyUI path:**

```powershell
uv run artifex onboard first-run --role renderer --comfy-root "D:/AI/ComfyUI_windows_portable" --json
```

This uses the existing `renderer-auto` preview's strict selection rules. A
single checkpoint/Python is a *proposal*, not an authorized renderer launch.
Ambiguous checkpoints, `extra_model_paths.yaml` or multiple Python binaries
require an explicit operator selection. The returned
`next_safe_preview_argv` remains read-only; any
`optional_config_write_argv` is **advice for an explicitly approved
config-only `--apply`**, not executed by this command.

**On PC-A, before `config/local.yaml` exists:**

```powershell
uv run artifex onboard first-run --role controller --json
```

It checks local `uv`, the configured Spark GGUF and llama.cpp binary presence,
and lists the missing inputs without downloading anything. Once PC-B's
**actual LAN attestation URL** and shared token environment variable are known,
use the authenticated preview:

```powershell
uv run artifex onboard first-run --role controller `
  --attestation-url "http://192.168.1.20:8190" --json
```

The URL above is only an example: use the PC-B address from the real network.
Set `ARTIFEX_RENDER_NODE_TOKEN` in both PCs' environments, **never in the
command line or a saved JSON file**. The controller preview performs at most
the two existing protected, bounded authenticated GETs to PC-B
(`/v1/attestation?fresh=1` and gateway `/system_stats`). It does not call
`/prompt`, use a direct 8188 bypass, or launch ComfyUI.

If the YAML already exists, the guide points to existing
`qualify overview` (PC-A) or `onboard renderer-safety` (PC-B) instead of
rewriting anything. For an audit trail, `--report-path` creates **only a new
JSON report** and refuses to overwrite/symlink a previous one.

An opt-in config-write argv is shown only for native Windows, a protected
authenticated pair preview or an unambiguous renderer model/Python preview.
It is **never executed**. A report can be `configured` or `preview_ready`
without being qualified; the separate 14-stage actual Windows/GPU/soak
verification is mandatory.

## PC-A qualification handoff: protected readiness + actual workflow nodes

Once PC-B has a protected gateway and PC-A has a non-symlinked
`config/local.yaml`, run **one read-only handoff** on PC-A:

```powershell
uv run artifex qualify handoff --config .\\config\\local.yaml --json
```

This runs the existing PC-A/PC-B authenticated `qualify overview` snapshot,
then validates both **configured ComfyUI production and repair workflows**
through the existing `deployment verify` **without** `--render-smoke`.
The node checks use the configured protected PC-B gateway, not a direct
unmanaged ComfyUI port. No new stage is registered, workflow submitted,
Task Scheduler job changed or model downloaded.

The response distinguishes `blocked`, `session_start_candidate`, and
`saved_session_review`. The first is returned as structured JSON and a
nonzero CLI exit; missing PC-A/PC-B live readiness, stale samples, incomplete
real node dependencies or a contradictory deployment result **block** the
suggestion to begin qualification. A `session_start_candidate` includes an
**unexecuted** `qualify start` argv; a saved session is **not assumed active**
and suggests only a preview-only continuation. An optional
`--output data/qualification/handoff.json` creates a new evidence report
but never overwrites another one. Always use the actual configured paths.

Even if the handoff is ready, `renderer_start_authorized=false`,
`gpu_jobs_submitted=false` and `production_qualified=false` remain
explicit. This **does not prove** Windows virtualenv process survival
(Issue #93), active GPU image generation, the eight-hour soak, or all
14 required real-machine stages (Issue #40). No live ComfyUI service may
be stopped, restarted or adopted to satisfy these checks.

## One-command PC-A qualification overview (read-only)

From **PC-A**, get current authenticated PC-B safety, PC-A preflight,
a grouped action plan and all 14 **saved** stage statuses in one report:

```powershell
uv run artifex qualify overview --config .\\config\\local.yaml --json
```

The command selects the newest properly named local qualification session if
`--session-id` is omitted. This is labelled `latest_saved`, **not**
a verified active auto-collection binding. Supply `--session-id ID` to
choose a specific session; if the newest session is corrupt, foreign or
symlinked, the command refuses to silently reuse an earlier one.

The response includes `recorded_pass_count`, `unresolved_stage_count`,
stage-specific next actions, `action_plan` grouped by PC, and a
`remediation_triage` for each actionable step. The triage distinguishes
`observed_live` (a PC-A check was **sampled**, not necessarily passed),
`unavailable` (no usable observation), `requires_local_pc_b` (do not
execute the PC-B command remotely), `operator_review`, and
`real_machine_evidence`. Separate `already_sampled_read_only`,
`unavailable_read_only`, `pc_b_local_checks_required`, and
`operator_review_required` lists make it clear what still needs a person.

When the environment is blocked, `next_priority` favors PC-A
configuration/manual review or a necessary native PC-B check rather than
telling the user to **repeat read-only checks just performed**.
`next_safe_command` is therefore null for unresolved current readiness.
When PC-A is ready and only ordinary persisted Packs remain, the
`qualify collect` recommendation is preview-only (no `--apply`).
Suggested commands are **never executed**, and no new runtime/GPU operation
is triggered.

`production_qualified=false`, `commands_executed=false` and
`mutated_services=false` are explicit: **stored PASS is not independent
revalidation**. Use `qualify verify SESSION_ID` for authoritative acceptance
after actual real-machine 14-stage verification and eight-hour soak. The
overview never launches ComfyUI, restarts PC-B or submits GPU work.

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

## PC-B Task Scheduler hard-termination safety

Artifex now installs the managed renderer scheduled task with
`-DisallowHardTerminate`, `MultipleInstances=IgnoreNew` and
`ExecutionTimeLimit=0`. Microsoft documents that the scheduler may
otherwise call `TerminateProcess` when a task does not respond.

The following PC-B command reads the **actual registered task**, checks
its Artifex ownership marker, one expected action, configured executable,
arguments, working directory and all three safety options, but makes no
changes or stops:

```powershell
uv run artifex startup audit --role renderer --config .\\config\\render-node.yaml
```

The existing `startup status` output now includes the scheduler's actual
`action_count`, `allow_hard_terminate`, `multiple_instances` and
`execution_time_limit_seconds`. Missing policy properties remain unknown
(`null`) rather than silently converting to safe `false` or `0`. Unknown
or older unsafe registrations fail closed.

During an explicit stopped-renderer maintenance window, an older Artifex-owned
task may be upgraded with `startup install --role renderer --replace`.
**Task Scheduler's `Ready` state does not prove ComfyUI has exited.**
Replacement first acquires the same exclusive supervisor lease, checks the
persisted ComfyUI ownership receipt for a surviving process, and requires
the configured upstream TCP port to have no listener. The native Windows
registration script rechecks the listener and running-task state immediately
before registration. An invalid receipt, failed socket inventory, a live receipt-recorded child,
or an occupied lease block replacement. A missing receipt is not sufficient
proof of child absence; the listener probe and exclusive lease are additional
checks, not a guarantee against independently started processes. No running ComfyUI
child is terminated. All launch paths for the same renderer must share the
same ownership receipt/lease location. Non-Artifex clients cannot be
atomically fenced by this check, so use a controlled maintenance window.

**Limit:** `DisallowHardTerminate` does not guarantee that Windows
logout/shutdown, Task Scheduler job-object behavior, or other processes
cannot stop ComfyUI. Do not issue an uncontrolled task stop while the GPU
is busy. The audit reports `restart_authorized=false` and
`child_survival_qualified=false`; the real two-PC qualification remains
required.

## PC-A: inspect live PC-B safety without touching the renderer

After updating both PCs to this revision and configuring the existing
authenticated PC-B attestation endpoint, run the following from **PC-A**:

```powershell
uv run artifex qualify readiness --config .\\config\\local.yaml --json
uv run artifex qualify plan --config .\\config\\local.yaml --json
```

PC-A uses the configured primary renderer ID, attestation URL and existing
Bearer token environment variable to make **two uncached, read-only** remote
observations: `/v1/owner-audit` and `/v1/renderer-safety`. The second
endpoint includes current native Windows TCP inventory for ComfyUI, protected
attestation and gateway, plus the local process/scheduler/receipt audit.
No extra PC-B command is needed for a routine remote observation while the
attestation service is already running. Do not pass tokens as command-line
arguments or paste them into saved reports.

The readiness output now independently checks that both observations are
fresh (within 120 seconds, with at most 30 seconds of future clock skew), the
reported ComfyUI execution PIDs agree, both sample times are reasonably
close, all required owner checks pass, and the protected service ports have
coherent listener ownership. An unavailable/older PC-B endpoint, missing
credentials, a changing PID, a missing gateway, or a blocked safety report
leaves `environment_ready=false`. Re-run from PC-A first. The action plan
gives read-only remediation instructions; it never executes them.

**Limit:** this is topology and process identity observation, not
authentication proof for the gateway and never launch, reattach, restart or
production permission. A healthy readiness result cannot replace native
14-stage real-machine validation or the eight-hour GPU soak. On an older
PC-B version without `/v1/renderer-safety`, the remote check is reported as
unavailable rather than silently bypassed. Updating the service should
happen only in an appropriate maintenance window; do not stop live ComfyUI
to force a green result.

## PC-B: inspect existing ComfyUI before any startup or reattachment

After `onboard renderer-auto` has created a proposed PC-B configuration,
run this **read-only live safety check** on actual native Windows PC-B:

```powershell
uv run artifex onboard renderer-safety --config .\config\render-node.yaml --json

# Optional: save new evidence; never overwrite old evidence.
uv run artifex onboard renderer-safety --config .\config\render-node.yaml --output .\renderer-safety.json --json
```

The command samples the native Windows TCP listener inventory **twice** for
the configured ComfyUI upstream (normally 127.0.0.1:8188), attestation
(8190) and protected gateway (8191). It detects absent, exposed, foreign
or changing listener PIDs and addresses. When a ComfyUI listener exists,
it reuses the established owner audit: native Windows CIM process identity,
launcher ancestry, persistent receipt, TCP ownership and Task Scheduler
policy. The result reports blockers and safe next actions, without
raw command lines, auth tokens or changing any Windows services.

The diagnostic also requires the **attestation and protected gateway**
listener ports to be present and stable with one coherent supervisor PID,
distinct from the original ComfyUI execution PID. Missing services, divergent
PIDs, ambiguous ownership, or duplicate/mismatched `--listen` and `--port`
arguments block `owned_observed` and produce a read-only remediation message.
The shared supervisor PID is only a TCP topology observation: it does **not**
prove that a service enforces its Bearer token or that Windows has authorized
a start. A blocked result should not be resolved by restarting an active
ComfyUI outside a controlled maintenance window.

A previously verified process is reported as `owned_observed` and should be
**left running**. Unknown or foreign listeners must not be adopted. Even if
a port is empty at one instant, that observation **does not authorize**
a new process to start: it could become occupied later or a stale task or
receipt might exist. The fields `launch_authorized`, `restart_authorized`,
`reattach_authorized`, `mutated_services`, and `production_qualified` are
unconditionally false. No background process, GPU job, download, Task
Scheduler change, or ComfyUI service manipulation is performed.

Actual original PID/launcher survival under reboot, a sustained eight-hour
GPU soak and all 14 real-machine production qualification stages remain
separate requirements. Use a safe maintenance window for any change to
the existing ComfyUI process or its listen address.

### Protected live launch gate (separate from diagnostic reports)

A **live process spawn** is permitted only inside the managed supervisor's
exclusive PC-B lease, after rechecking both a missing ownership receipt and
a demonstrably unoccupied local upstream TCP port immediately at the launch
boundary. Neither a free port in a `renderer-safety` report nor a passing
owner audit authorizes a separate manual launch. If a receipt appears after
the first check, or the supervisor lease is missing, the managed launcher
refuses to spawn and does not erase that receipt.

On native Windows, Artifex captures the original venv launcher CIM identity
immediately after Popen where possible. If `python.exe` acts as a shim and
exits before the real ComfyUI process becomes HTTP-ready, the *original*
listener may be accepted only when the captured launcher PID, creation time,
full executable and exact command line establish direct child ancestry,
the real interpreter matches, and subsequent TCP and receipt checks pass.
An uncaptured, replaced or unverifiable launcher instead fails closed; a
responding HTTP endpoint or matching port is never sufficient. No unrelated
ComfyUI process is terminated or automatically adopted.

These checks protect the in-process launch/reattachment path but do not
guarantee atomic exclusion of independent non-Artifex programs, Windows
shutdown survival, a drained ComfyUI queue or real-GPU qualification.


## Prepare PC-B with existing ComfyUI automatically (preview first)

On the actual native Windows PC-B, point Artifex at your existing ComfyUI or
portable installation. It scans only that supplied directory, **not all drives**.

```powershell
# Nothing is written and no GPU service is started.
uv run artifex onboard renderer-auto --comfy-root D:\ComfyUI_windows_portable --json

# Create a new PC-B config when the preview selects a single safe checkpoint.
uv run artifex onboard renderer-auto --comfy-root D:\ComfyUI_windows_portable --apply --json

# When multiple checkpoints or extra_model_paths.yaml exist, choose explicitly.
uv run artifex onboard renderer-auto --comfy-root D:\ComfyUI_windows_portable --checkpoint-path D:\models\my-model.safetensors --apply --json

# Change existing config with explicitly authorized merge, preserving unrelated keys.
uv run artifex onboard renderer-auto --comfy-root D:\ComfyUI_windows_portable --apply --update --json
```

The preview discovers the exact `main.py` directory, embedded or venv
Python candidates, local checkpoint candidates and local LoRA roots. If **one**
local checkpoint and **one** Python candidate are found, it offers them as
a proposal (the `--apply` flag accepts this particular
proposal). If several are found or extra model paths could introduce more,
it fails closed and requests the exact `--checkpoint-path`.
Multiple Python installations require `--comfy-exe`. Paths,
IP/host and all three ports are configurable. It will not silently
choose a model, executable or LoRA root from some other drive.

For a **managed** PC-B configuration, the resulting YAML declares:
- ComfyUI bound to **127.0.0.1:8188**, not an unprotected LAN listener
- an **authenticated protected gateway** on port 8191
- a token-required renderer attestation agent on port 8190
- the selected checkpoint and discovered LoRA roots

The command merely prepares YAML: it **does not install Windows scheduled
tasks, start/adopt/restart/stop ComfyUI, perform CUDA operations, or prove
that port 8188 is unoccupied**. An already running unmanaged ComfyUI must
not be adopted or interrupted. To remain unmanaged, use the explicit
`--external-comfy` option; this disables the protected
gateway and **does not meet the protected-ownership production gate**.
Before enabling startup or management, separately verify the actual
renderer process ownership, Scheduler safety and current TCP port state.
Only afterwards use the authenticated PC-A `onboard pair-sync`
preview to derive its connection settings. Docker is not involved.

## Prepare PC-A from a live authenticated PC-B with minimal typing

When the actual Windows PC-B render agent and its protected ComfyUI gateway
are already running, PC-A can derive its node ID, LAN render URL, and exact
configured production checkpoint **without guessing a model from a directory**.
Provide only the PC-B attestation URL (and change the gateway port if not 8191):

```powershell
# On PC-A, configure the *existing* shared token environment variable.
# Never paste the secret into a CLI argument, issue or JSON report.
uv run artifex onboard pair-sync --attestation-url http://192.168.1.20:8190 --json

# Explicit opt-in: create config/local.yaml (never overwrite an existing file).
uv run artifex onboard pair-sync --attestation-url http://192.168.1.20:8190 --apply --json

# Explicit opt-in: merge into existing config/local.yaml, retaining unrelated settings.
uv run artifex onboard pair-sync --attestation-url http://192.168.1.20:8190 --apply --update --json
```

The command uses the existing `ARTIFEX_RENDER_NODE_TOKEN` value from
PC-A's environment, retrieves a fresh authenticated PC-B
`GET /v1/attestation?fresh=1` snapshot, and verifies the
**authenticated** gateway at the *same PC-B LAN host* (by default port 8191)
via read-only `GET /system_stats`. It ignores PC-B's upstream
`comfyui_base_url` (usually loopback on the renderer) for
PC-A transport and **does not fall back to directly exposing ComfyUI**.
Node IDs, checkpoint names, SHA-256 fields, timestamps and remote URL shapes
are bounded and validated; no HTTP redirects are followed. Gateway and
attestation endpoints must already be configured and reachable.
Native Windows PC-B ownership and eight-hour soak **are not proved** by
this setup command; use `qualify readiness` followed by real
qualification.

Preview is the default and performs no filesystem write. Only `--apply`
creates the PC-A YAML; `--update` requires `--apply`
and preserves unrelated controller settings. The command never downloads
models, modifies PC-B, starts or restarts any process, or submits GPU work.
Tokens never enter emitted reports or written config values.
If PC-B is not running a protected gateway, the setup fails closed rather
than silently picking an unprotected port. Because default HTTP on a LAN
does not encrypt credentials, use a trusted private LAN or a secure tunnel;
do not publish these ports on the Internet.

## Recheck the safe actions and update the readiness plan

After generating or saving a readiness snapshot, PC-A can refresh only its
**known read-only diagnostic probes** and reconcile the actual current
results with the earlier snapshot:

```powershell
# Current PC-A/PC-B state; probes run once, no previous-state comparison
uv run artifex qualify recheck --config .\config\local.yaml --json

# Compare the saved diagnostic to a fresh live observation and update the plan
uv run artifex qualify recheck --config .\config\local.yaml --from-report .\readiness-report.json --json

# Optional immutable comparison report
uv run artifex qualify recheck --config .\config\local.yaml --from-report .\readiness-report.json --output .\readiness-recheck.json --json
```

Unlike `qualify plan --from-report` (offline, no probing), this
`qualify recheck` command **always checks the currently configured
PC-A and authenticated PC-B services**. The only active probes are the existing
in-process controller preflight and the authenticated, read-only PC-B owner
audit; no subprocess command from a plan or saved JSON is ever executed. The
output contains a status-change list (improved, regressed, newly observed or
unavailable), an updated **fresh** action plan, and an itemized check-execution
ledger. Previous JSON can supply historical comparison only; it cannot make a
failed *current* check PASS. A missing or unreachable PC-B remains unknown
or blocked, never successful.

Some instructions in a plan point at native commands on PC-B. In particular,
`render-node preflight` on PC-B is **not** run by a controller-only
session: the ledger explicitly marks `requires_local_pc_b` even when
the remote asset attestation succeeds. Unknown future plan steps are likewise
never run automatically. Use the separate PC-B CLI in a safe maintenance
window for the remaining direct checks. The report always says
`arbitrary_plan_commands_executed=false`,
`mutated_services=false`,
`production_qualified=false` and
`gpu_soak_qualified=false`. Real GPU operations, process
restarts, model downloads, Task Scheduler updates and all 14 actual machine
qualification stages remain outside this command.

## From a readiness diagnostic to a safe PC-by-PC action plan

Run on PC-A, using your existing real two-PC configuration:

```powershell
uv run artifex qualify plan --config .\config\local.yaml --renderer-config .\config\render-node.yaml --json

# If a readiness report was saved already, generate the plan offline
# without contacting either machine or loading the YAML:
uv run artifex qualify plan --from-report .\readiness-report.json --json
```

The result classifies each unresolved check into explicit **PC-A** or **PC-B**
work and provides a suggested *structured argv list* (not a command string
to pass to a shell). Read-only checks use existing tools such as
`artifex preflight`, `artifex render-node owner-audit` and
`artifex render-node preflight`. Installation, config edits, token
provisioning and any maintenance that might affect a live GPU process are
categorized `review_required` and **are never executed by this tool**.
The 14-stage actual-machine evidence is shown separately as
`real_machine_evidence` and is never claimed complete by
diagnostics. Every step explicitly reports `automatically_executed=false`.

Saved readiness reports are limited to 1 MiB, reject symlinks and are
considered **stale after five minutes**. A stale report always includes
a fresh-readiness step and cannot be considered a current environment pass.
The optional `--output PATH` writes a new JSON report without
overwriting existing files. All paths in proposed commands are derived only
from operator-selected config paths, and are passed as separate argv values
to avoid shell-injection problems; neither a failed check's detail nor a
remote endpoint's error text becomes executable shell input.

This is a preparation plan only: it does not modify Windows Task Scheduler
or ComfyUI, download files, set credentials, start an LLM, run a GPU job,
or make changes to qualification evidence. An operator can carry out safe
read-only commands first, then explicitly authorize riskier changes during
a maintenance window. Complete real two-PC 14-stage and eight-hour
qualification remains required after the plan is clear.

## One-command PC-A / PC-B qualification readiness diagnosis

Before creating a real qualification session, run this **read-only** command on
PC-A, with the intended actual PC-B online:

```powershell
uv run artifex qualify readiness --config .\config\local.yaml --json

# Optional: retain a new report for audit; an existing file is never overwritten.
uv run artifex qualify readiness --config .\config\local.yaml --output .\readiness-report.json --json

# Optional: see the 14 stages already *recorded* for an existing session.
uv run artifex qualify readiness --config .\config\local.yaml --session-id SESSION_ID --json
```

The report groups actionable observations by `target: pc_a`,
`target: pc_b` and `target: qualification`, with individual
`pass`/`fail`/`unknown` states and specific `next_action` guidance.
It reuses existing PC-A controller preflight (LLM endpoint, ComfyUI API,
authenticated PC-B model inventory, clocks and configured model filenames)
and the existing read-only authenticated PC-B owner audit (the Windows CIM
process/launcher chain, Scheduler policy, receipt and TCP socket ownership).
Controller-side Windows/uv/primary node/credential **presence only** and
selected local GGUF file are also checked. The Bearer token **value** is
never displayed or persisted. Every remote snapshot must be current and
all nine mandatory ownership checks must pass. Unavailable probes remain
unknown, never green.

`environment_ready=true` means **only that the observed PC-A/PC-B
environment prerequisites passed at this point in time**. Both
`actual_machine_qualification_complete` and
`actual_gpu_soak_verified` are *always false*. The optional
`--session-id` reports recorded stage statuses without changing or
independently revalidating them. Run `qualify verify` to revalidate the
full 14-stage evidence; a recorded PASS is not accepted as proof by this
readiness command. No Task Scheduler registration, ComfyUI launch/stop,
GPU image generation, background daemon start or DB mutation occurs.
Non-ready observations exit with a nonzero code while still returning
structured JSON for automation. The optional JSON report output refuses
existing paths and symlinks.

Failures commonly indicate a missing PC-B authentication token, an
unreachable LLM/ComfyUI service, wrong model names, a stopped Scheduler
task, or an unverified ownership receipt. This command reports the
next safe step rather than automatically repairing a possibly
running GPU service.

## Continuous PC-B ownership surveillance during the eight-hour soak

The existing `qualify soak-run` and `qualify soak-observe` commands on PC-A
now sample the **actual PC-B owner audit** over the authenticated
`GET /v1/owner-audit` channel at **every** monitoring interval when a primary
renderer is configured and `qualification.require_renderer_owner_observation`
is enabled (the two-PC default). There is no extra operator-side command
to repeat every few minutes.

Each fsynced JSONL sample records the owner snapshot timestamp, actual
ComfyUI listener PID and creation time, launcher PID, scheduler state,
all nine ownership checks, and whether the live observation was stable.
It deliberately omits process command lines, passwords, tokens and config
contents. The trace header uses schema v2 and explicitly declares
`owner_observation_required=true` for such runs. A missing sample,
unreachable audit endpoint, stale observation, failed ownership check,
unverified node identity, or **changed ComfyUI PID or creation timestamp**
makes the overall eight-hour trace fail review. Monitoring gaps continue
to be checked against the monotonic clock; recovered connectivity cannot
erase a previous outage. A surviving Task Scheduler supervisor alone
does not demonstrate child survival and never overrides a failed owner
audit.

Review an existing trace safely on PC-A:

```powershell
uv run artifex qualify soak-check --config .\config\local.yaml --evidence PATH\TO\soak.jsonl --json
```

The assessment includes `owner_observed_samples` and `owner_incidents`
alongside the original GPU/DB/Pack counters. A production qualification
stage revalidates the trace with the currently enforced owner-audit policy:
an older schema-v1 or artificially downgraded trace **cannot** fulfill the
two-PC ownership requirement even if its other telemetry looks healthy.
Existing synthetic legacy tests retain their explicit test-only policy.
This adds **observation**, not process management: the monitor neither
sends GPU work nor stops, restarts or takes ownership of ComfyUI. A
passing soak trace is still not automatic production readiness: all 14
qualification stages and current PC-B preflight must pass separately.

## Automatically bind PC-B owner evidence to PC-A qualification

With PC-A's `render_nodes.primary` configured and authenticated
`attestation_url`, the current PC-B attestation server offers a **new
read-only** `GET /v1/owner-audit` endpoint. It uses the existing Bearer
token, requires the PC-B renderer's selected YAML config, and always takes a
fresh Task Scheduler/CIM/TCP snapshot. Unauthorized requests are rejected.
No ComfyUI start, GPU render, restart, stop or local task change is performed.

On PC-A, `qualify start --config .\config\local.yaml` now tries to
fetch and store the initial PC-B owner snapshot automatically. Subsequent
`qualify verify SESSION_ID --config .\config\local.yaml` **refreshes** the
snapshot automatically, and fails closed if that retrieval is unavailable
even when an earlier snapshot was stable. During an extended soak, operators
can inspect/save an additional snapshot without any GPU intervention:

```powershell
uv run artifex qualify owner-observe SESSION_ID --config .\config\local.yaml --json
```

Every snapshot is a separate JSON file under
`qualification.evidence_dir/SESSION_ID/owner-observation-*.json`, with
an integrity SHA-256 reference bound into the qualification session. The
last snapshot must belong to the configured primary node and same session,
have all nine ownership checks passing, and have been collected recently
(default freshness window five minutes). The client rejects malformed
response shapes, redirects, mismatched node IDs, and any claim that
`restart_authorized`, `child_survival_qualified`, `production_qualified`
or `mutated_services` is true.

For two-PC sessions,
`qualification.require_renderer_owner_observation` defaults to `true`.
If PC-B is absent, insecure, ambiguous or unreachable, the qualification
cannot report `ready=true`. This **does not** write PASS for any of the
14 existing real-machine stages; those still need measured Pack, recovery,
Discord (when enabled), archive reproduction and the full eight-hour soak
evidence. Observations alone prove only a point-in-time state, not survival
under Windows logout/shutdown/Job Object behavior. HTTP on a trusted LAN is
not cryptographically confidential by itself; secure the LAN or use an
appropriately protected tunnel, and do not expose the attestation port to
the public Internet.

## Single-command read-only PC-B ownership evidence

Run this on **the actual renderer PC-B** from the Artifex checkout. No GPU
generation is submitted, no ComfyUI process is started/stopped, no task is
registered or changed, and no current job is interrupted.

```powershell
uv run artifex render-node owner-audit --config .\config\render-node.yaml --json

# Explicitly retain a new, time-stamped evidence report.
uv run artifex render-node owner-audit --config .\config\render-node.yaml --save --json
```

The command checks the **current** Windows Task Scheduler task against the
selected config and safety policy; ComfyUI's saved ownership receipt; CIM
process PID, executable, argument identity, start timestamp and Windows venv
launcher ancestry (if applicable); two consecutive real upstream TCP listener
inventories; and unchanged process/receipt during the observation. It refuses
unknown or foreign owners, changed/recycled PIDs, unexpected direct clients,
publicly exposed listeners, missing/unsafe scheduled tasks and unverifiable
launcher state. A scheduler state of `Ready` is reported **inconclusive**:
a live orphan can still exist, but scheduled reattachment is not demonstrated.
Missing receipt and conflicting evidence produce **blocked**.

`observed_stable` is **only a point-in-time snapshot**: it is *not*
evidence that any real GPU job survived Task Scheduler stop, logoff or Windows
shutdown. Every result explicitly sets `restart_authorized=false`,
`child_survival_qualified=false`, `production_qualified=false` and
`mutated_services=false`. Failed/inconclusive observations exit nonzero,
but may still be saved using `--save` for diagnosis. `--output PATH` can
select a new destination; existing files, symlinks and overwrites are refused.
The JSON report deliberately omits raw command lines, config file contents,
tokens and environment variables. **Do not** record an Issue #40 PASS merely
because this check succeeds; actual two-PC GPU, 14-stage and 8-hour soak
evidence is still required.

## PC-A automatic authenticated PC-B evidence (no manual JSON copy)

After installing this version on both Windows PCs, the **protected** PC-B
attestation service starts a bounded local background evidence sampler.
Only on native Windows with a configured managed ComfyUI, enabled protected
gateway, an existing non-symlinked PC-B YAML, and a configured Bearer token,
the PC-B service runs PR #116's temporary **no-GPU** launcher fixture
at startup and approximately every four minutes, observing actual
ComfyUI ownership independently. The temporary fixture is never ComfyUI.
The result is cached in RAM for up to ten minutes. If unconfigured,
missing, blocked, or stale, the API fails closed. It changes no GPU job,
Task Scheduler registration, renderer service, or ownership receipt.

On PC-A, use just one command (no PC-B JSON transfer):

```powershell
uv run artifex qualify overview --config .\config\local.yaml --pc-b-owner-live --json
```

PC-A fetches the protected, Bearer-authenticated
`GET /v1/owner-readiness-evidence` from the configured PC-B node, checks
its bounded cached proof and node ID, and *independently* fetches fresh
authenticated PC-B ownership plus fresh node attestation. It cross-checks
host, listener PID, start timestamp, launcher PID, receipt schema and
Task Scheduler state. **GET never launches Python or ComfyUI**; the
temporary mock is run exclusively by the local PC-B worker, not by
a remote request. An unauthorized endpoint request returns 401.

`pc_b_owner_evidence.source_mode=authenticated_remote` indicates this
transport; `remote_bearer_checked=true` indicates a matching authenticated
PC-B endpoint response, not TLS encryption or proof that the remote
snapshot was never tampered with by its originating machine. Saved
provenance is still not independently trusted. Even a correlated snapshot
does not authorize a restart, GPU job, Issue #93/#40 closure, or a stage PASS.

The previous `--pc-b-owner-report` transferred-JSON path remains available
when desired. Both flags together fail rather than choosing one silently.
If PC-B has not yet collected a fixture, is on an older software version,
or is unavailable, the live option returns a blocked/unavailable correlation;
do not restart the running ComfyUI just to change this result.
## PC-A correlation of PC-B local owner evidence

On the actual Windows PC-B, run the no-GPU, read-only combined diagnostic
with a NEW report destination. Never overwrite previously collected evidence:

```powershell
uv run artifex render-node owner-readiness --config .\config\render-node.yaml --output .\data\qualification\owner-readiness-pc-b-unique.json --json
```

Transfer this JSON to PC-A through the operator's trusted file-transfer path.
On PC-A, use its existing authenticated controller configuration:

```powershell
uv run artifex qualify overview --config .\config\local.yaml --pc-b-owner-report .\data\pair\owner-readiness-pc-b-unique.json --json
```

The new `pc_b_owner_evidence` section is **optional**. The previous
`qualify overview` behavior is unchanged without that option. With it,
PC-A strictly parses the bounded, non-symlinked copied JSON and correlates
the configured PC-B host, actual listener PID, process start timestamp,
launcher PID, receipt schema and Task Scheduler state against a fresh
authenticated `/v1/owner-audit` plus fresh renderer attestation.
Never follow a URL specified in the copied JSON or execute a command
on PC-B. A changed PID, stale/unreadable report, unknown ownership,
hostname mismatch, or inaccessible service blocks the correlation.
The resulting states are `correlated_read_only`, `blocked`, `stale`,
`mismatch`, or `unavailable`. If requested correlation cannot be completed,
the PC-A overview's `environment_ready` is false and it suggests gathering
new evidence; it does not schedule a GPU job or register a stage PASS.

**A copied JSON file is not cryptographically authenticated.** A match
only establishes consistency of two observations, never authenticity of
the file, ComfyUI child survival after supervisor loss, actual GPU
performance, or Issue #93/#40 completion. `file_source_authenticated=false`,
`stage_pass_registered=false`, `production_qualified=false`, and
`renderer_restart_authorized=false` are hard-coded in the correlation.
## One-command read-only PC-B owner evidence reconciliation

Run on the actual native Windows PC-B with the original ComfyUI still running:

```powershell
uv run artifex render-node owner-readiness --config .\config\render-node.yaml --json
uv run artifex render-node owner-readiness --config .\config\render-node.yaml --output .\data\qualification\owner-readiness-unique.json --json
```

The command uses exactly the configured managed ComfyUI Python executable
for PR #115's disposable no-GPU launcher fixture (no interpreter fallback),
then independently observes the real listener using the receipt, Windows
CIM identity, Task Scheduler policy, and repeated TCP ownership inventories.
It never submits /prompt, changes a task, adopts or restarts a process, or
modifies the production ownership receipt.

The JSON reports both independent checks, original observations and
`remaining_real_machine_evidence`. Results are `observed_independently`,
`blocked` or `unsupported`. Passing mock + real observations **does not**
prove that the disposable Python's PID belongs to the real ComfyUI.
Even `observed_independently` always retains
`issue_93_closure_authorized=false`, `issue_40_closure_authorized=false`,
`production_qualified=false`, and `renderer_restart_authorized=false`.
Supervisor-loss survival, a real GPU render, all 14 actual stages and the
eight-hour soak still require separate target-PC evidence.

Missing/unsupported configured Python, unsafe Task Scheduler policy, missing
receipt, foreign listener, process drift, or mismatched host fails closed.
Optional --output exclusively creates a new JSON file; use a unique name.
Native CI mock successes never substitute for physical PC-B qualification.
## Native PC-B venv launcher evidence without GPU or live renderer changes

On the intended **Windows PC-B**, run the isolated diagnostic from an Artifex
`uv` environment, optionally choosing the *actual* known ComfyUI Python
executable with `--python`. This is the minimum local mock check for Issue
#93, **not** a production listener probe:

```powershell
uv run artifex render-node launcher-probe --json
uv run artifex render-node launcher-probe --output data/qualification/launcher-evidence.json --json
# Optional explicit Python: --python "D:/ComfyUI/.venv/Scripts/python.exe"
```

The tool launches a **single temporary non-Comfy Python loopback fixture** on
an OS-selected ephemeral port. The fixture emits its actual PID, and the
diagnostic compares native Windows CIM start time, original venv launcher
executable/command line, **direct parent ancestry** when separate, repeated
real TCP ownership, and an in-memory receipt validated by the same checks
used for protected ComfyUI. It returns `observed` only if all match and
otherwise fails closed with `blocked`; a PID or responsive port alone does
not pass. The child reads no Artifex token environment variables.

The fixture voluntarily exits when its private release flag appears (or
its bounded lifetime expires); **nothing kills or restarts** the actual
ComfyUI, and no receipt is saved in the production renderer configuration.
The optional output report is exclusive-create and must never overwrite
existing evidence. Its `test_listener_only=true`,
`actual_comfyui_inspected=false`, `renderer_restart_authorized=false`,
and `production_child_survival_qualified=false` fields mean that a
successful native mock **does not close Issue #93**. Independently observe
the real PC-B listener with `render-node owner-audit` and complete the
Windows process survival and full #40 real-GPU soak checks.

## Windows virtualenv Python launcher and actual renderer PID

Native Windows may run a venv `Scripts\python.exe` as a **launcher** and execute
the Python interpreter as a separate child process. `Popen.pid` can then be
different from the ComfyUI TCP listener PID. In protected gateway mode Artifex
must **not** infer ownership from the listener port alone. It now first
verifies the launcher's full executable/arguments and Windows CIM identity;
a distinct listening Python PID is accepted only when it is its **direct**
child, starts no earlier than the launcher, has the expected arguments and
matches a repeat socket inventory. A schema-2 ownership receipt preserves
both identities. Subsequent supervisor recovery checks the *actual* child
PID, immutable creation time, executable, command line and recorded parent
rather than trusting a possibly exited/reused launcher PID. An unknown,
foreign, reordered or ambiguous chain blocks automatic ownership/restart.

The native Windows CI tests launch only a temporary loopback Python listener,
not ComfyUI or a GPU workload. They prove identity checking on a disposable
venv launcher but **do not** qualify the production PC-B, GPU process,
Task Scheduler logoff/shutdown or Windows Job Object effects. On the target
PC-B, record the launcher PID, actual listener PID, creation time, parent PID
and receipt while a real workload is active. Any mismatch remains
fail-closed and should be investigated before activating protected startup.

## Isolated Windows Task Scheduler process-tree probe

This deliberately uses **only a temporary, random-name Task Scheduler task**,
a no-GPU Python mock supervisor and a short-lived heartbeat child. It never
stops or replaces Artifex-Renderer, Artifex-Controller, production ComfyUI,
or any active GPU render.

Preview (no task registration or process launch):

```powershell
uv run artifex startup lifecycle-probe --json
```

Explicit isolated experiment on PC-B or disposable Windows CI:

```powershell
uv run artifex startup lifecycle-probe --apply --mode protected --json
uv run artifex startup lifecycle-probe --apply --mode baseline --json
```

The protected variant registers the mock task with DisallowHardTerminate;
baseline intentionally omits it, **only for the randomly named mock task**.
It records the actual Task Scheduler setting, original supervisor/child CIM
identity, child heartbeat progress after Stop-ScheduledTask, and whether the
supervisor itself stopped. A surviving child is *not* reported as a successful
supervisor-loss test if the supervisor is still running. Every report includes
production_qualified=false and restart_authorized=false. Reports are saved to
the configurable --report-dir (default data/qualification/scheduler-probe).
CI archives the report JSON for inspection. Cleanup verifies the task marker
and action before unregistering; the mock processes are bounded and receive
a voluntary release signal. A blocked/inconclusive result must **not** be
treated as evidence of child survival.

The test does not prove Task Scheduler logoff/shutdown, power-loss or
Job Object behavior on the operator's actual PC-B, nor does it authorize an
actual ComfyUI stop. Issue #40 still requires verification with a real
GPU render and two-PC configuration.

## PC-B exclusive supervisor process lease (Task Scheduler overlap safety)

With the protected managed gateway enabled, the renderer supervisor now
acquires an **OS-maintained nonblocking interprocess lock** before inspecting
or starting ComfyUI. It keeps the lock through orphan reattachment and the
entire monitoring lifecycle. A second Task Scheduler or manually started
supervisor using the **same configured ownership receipt path** refuses
startup rather than trying to adopt or launch another renderer.

The lease file is created alongside
`render_agent.comfyui_process.ownership_receipt_path` with the suffix
`.supervisor.lock`. The file is **not** deleted on supervisor shutdown;
its existence does not mean it is currently locked. Windows automatically
releases the actual file lock when the owning supervisor process exits or
crashes. The surviving original GPU child is not terminated, and a subsequent
supervisor can acquire the lock and attempt the receipt-verified reattachment
path. These rules do not require Docker, a paid service or an always-running
lock server.

**Configuration constraint:** all launch entries managing the same ComfyUI
must resolve this receipt/lease path to the exact same local filesystem
location. Do not configure separate working directories or receipt paths
for simultaneous Task Scheduler and manual runs. Do **not** delete the lock
file to "unlock" it; investigate the existing supervisor process instead.

The lease protects only against competing Artifex supervisors. It cannot
control other PC-B-local clients reaching `127.0.0.1:8188`, and it does
not guarantee that Windows Task Scheduler itself leaves descendants running.
That latter behavior, including stop/retry and Windows Job Objects, must be
tested on the real two-PC Windows deployment; automatic live-child
termination remains prohibited.

## Windows PC-B socket provenance and non-destructive supervision

The PC-B render-node can inspect the **actual Windows TCP listener owner
PID and live local client connections** using the built-in PowerShell
`Get-NetTCPConnection`. This does not require Docker, psutil, an external
service or manual socket data collection:

```powershell
uv run artifex render-node socket-audit `
  --config .\\config\\render-node.yaml
```

The read-only report shows upstream listener addresses/PIDs, currently
observed direct-loopback client PIDs, and a fail-closed diagnostic state.
A listener exposed on `0.0.0.0` or `::`, missing TCP information, or a
foreign client produces a nonzero CLI exit. The detached CLI cannot prove
which PID was spawned by Artifex; it therefore explicitly reports
`owner_unverified` for an otherwise valid loopback listener and **never**
sets `restart_authorized=true`.

In **managed protected mode** the running supervisor compares
`Get-NetTCPConnection` listener PIDs to its actual
`subprocess.Popen.pid` after startup and on subsequent health checks.
Unknown PID ownership, externally visible listeners and locally observed
direct clients cause protection to fail closed.

**Important safety change:** if the child is still alive but stops answering
HTTP health probes, Artifex will not terminate/kill it to auto-restart. A
live child may be executing a GPU job. Exited children can still be
restarted under the configured retry budget. When the supervisor shuts
down/crashes, it detaches a surviving ComfyUI child rather than killing it;
the child may consequently remain running. With protected gateway mode enabled,
Artifex now writes a Windows-only ownership receipt to the configurable
`render_agent.comfyui_process.ownership_receipt_path` (default:
`data/render/comfy-owner-receipt.json`). On the next supervisor launch it
checks the **actual** Windows process PID, creation timestamp, executable,
full command line and ComfyUI socket owner against that receipt and the
current configured launch. Only a perfect match is safely reattached as an
observed survivor. This reattachment does not kill or respawn the live GPU
child; the authenticated gateway/attestation can be served again.

When that same orphan naturally exits, the supervisor permits a replacement
only if the Windows TCP listener port is also demonstrably unused. PID
reuse, missing or malformed receipt, changed launch config, mismatching
host, foreign socket owners or an unhealthy-but-live child fail closed;
the software never assumes a stale receipt is permission to terminate
a process. Existing deployments with a live pre-receipt ComfyUI must
manually migrate during a safe maintenance window: the new supervisor
cannot infer ownership after the fact.

The receipt is historical identity evidence, **not an authentication
credential or a process-kill permit**. Keep it in a local non-shared
directory with normal restricted Windows file permissions; never copy
between machines or generate one manually. Windows Task Scheduler may
apply its own process-tree termination policy, which still requires
real-host qualification. This feature doesn't certify that policy or
prove that direct loopback clients are excluded.

Even an owner-verified idle TCP snapshot does NOT exclude a future local
process opening a new `/prompt` connection. It is **diagnostic evidence**,
not an atomic admission lock, a Windows firewall, proof of external
client isolation, or permission to restart. No automatic live-child
restart is introduced in this step.

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
session ID, `uv run artifex qualify collect <SESSION> --config .\\config\\local.yaml`.

Automatic collection and the manual `qualify record` command may
run simultaneously: both use one durable per-session Windows process lock
and atomic file replacement, and competing PASS registrations cannot
silently overwrite each other. Authenticated PC-B owner snapshots append
through the same lock. Leave `qualification.lock` in place; never delete it
during live sessions. A lock timeout is a blocked verification, **not** a
production PASS. For changed evidence, start a new qualification session.

When PC-A already has an explicitly started **ready** qualification session,
its native daemon now checks finalized Pack evidence approximately every
15 minutes and automatically applies only stages passing the strict
validators. It keeps the most recently started session ID in PC-A's DB;
an unready/new/foreign/configuration-changed session cannot be collected.
This can be disabled through `qualification.auto_collect_enabled: false`
(or its cadence adjusted with `auto_collect_interval_seconds`).
The collector only reads existing archives and records verified stages,
without sending new ComfyUI/GPU jobs, restarting services, or granting
production qualification. A standalone `qualify collect` is still useful
for immediate preview; `qualify verify` is the final authority.

The unattended three-Pack stage now requires three distinct finalized
post-session Packs carrying events from **one actual continuous daemon run**,
not any three manual archives. This evidence is auto-discovered with stable keyset pagination
(250 Packs/query, 10,000 total by default; adjust with `--scan-limit`),
and authoritatively rechecked by `qualify collect --apply`. Routine
telemetry pruning preserves daemon loop/completion provenance independently
so an eight-hour run does not silently lose its original start event.
If evidence is missing, leave
Artifex running normally and inspect the telemetry rather than forcing
manual PASS. `qualify recheck` also recognizes the PC-A authenticated
remote safety probe as observed (or unavailable) without executing its
advisory commands.
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
