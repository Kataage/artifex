# Target Windows Production Qualification

Issue #40 is the final production-readiness gate. Architecture/CI evidence alone is not sufficient. Run this procedure on the actual native Windows host with the intended models, ComfyUI workflow, LoRAs and services. Docker, Podman and WSL are optional and are not accepted as substitutes.

## PC-B registered Task Scheduler hard-termination audit

On PC-B run `uv run artifex startup audit --role renderer
--config .\\config\\render-node.yaml`. It must prove the
Artifex-owned registration has exactly one action matching the current
executable/config/working directory, `AllowHardTerminate=false`,
`MultipleInstances=IgnoreNew`, and unlimited execution time. An older
task may require an explicitly authorized `startup install --replace`
during a safe **stopped-renderer** maintenance window. A scheduler task
marked `Ready` is not enough: replacement also requires exclusive
supervisor lease access, no living PID identified by a persisted ownership
receipt, and a clear upstream TCP listener inventory (checked again in
PowerShell before registration). Any unknown observation fails closed.
Windows CI verifies the native PowerShell
`New-ScheduledTaskSettingsSet -DisallowHardTerminate` behavior without
registering or stopping any task.

This prevents a documented Task Scheduler hard-stop mechanism but does
not prove job-object/process-tree, sign-out or shutdown survival.
Test those separately on the actual PC-B, recording the original
ComfyUI PID and an active GPU job before and after supervisor loss.

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

## PC-B Task Scheduler supervisor exclusivity qualification

In managed protected gateway mode, Artifex uses a cross-process OS lease
derived from `render_agent.comfyui_process.ownership_receipt_path`.
Real-PC checks must explicitly attempt two simultaneous supervisor launches
(one Task Scheduler launch and one manual launch) and verify the second
fails **without** starting a new ComfyUI child, changing the ownership
receipt or disrupting a render. Then stop the first **supervisor only**,
verify the file lock is released even if the supervisor crashes, and that a
new supervisor reattaches to the original living ComfyUI PID.

CI tests include a real separate Windows Python interpreter trying to
acquire the lock while it is owned, and a child interpreter exiting via
`os._exit` before a replacement acquires it. These tests do not exercise
the actual Windows Task Scheduler's process-tree termination semantics or
prove whether its configured job object kills descendants. Confirm the
actual scheduled-task configuration and an active GPU generation on PC-B
before considering those steps qualified. Do not enable an automatic
ComfyUI terminate/kill path on the basis of an exclusive lock alone.

## PC-B socket ownership / never-kill-live-child qualification

On PC-B run
`uv run artifex render-node socket-audit --config
.\\config\\render-node.yaml`. This uses Windows'
`Get-NetTCPConnection` to report the ComfyUI listener addresses/PIDs
and observed loopback clients. The protected running manager additionally
checks whether the port's actual listener PID equals its own spawned
child PID; a failed or ambiguous probe prevents protected startup or
continued supervision. This report never authorizes a restart because
future local client submissions are not atomically blocked.

Regression tests verify that a live ComfyUI child is **not** killed if its
health endpoint hangs, even when the supervisor errors or stops. An
already exited child can be relaunched within the configured retry budget.
The protected supervisor now records a Windows CIM process identity
receipt for each child it directly launches. Real-PC qualification must
kill/restart only the **supervisor** during a running GPU generation,
then verify the surviving ComfyUI PID is reattached without any generation
cancellation or replacement. Test both a naturally exited original
process (new spawn allowed after vacant TCP port verification) and a
reused/foreign PID (strict refusal). Verify the receipt path configured by
`render_agent.comfyui_process.ownership_receipt_path` persists on PC-B,
and check that process creation time, executable and command line remain
stable across supervisor restarts. These read-only identity checks are
not permission to terminate a live ComfyUI child or bypass PC-B gateway
admission, and neither they nor CI simulations prove Windows Task
Scheduler's actual descendant-process behavior. Unverified provenance
remains an explicit qualification blocker.

## Coordinated PC-A/PC-B acceptance and release

`uv run artifex maintenance pair --config .\\config\\local.yaml`
previews both controller and authenticated renderer gateway states. With
`--apply`, the operation first drains and atomically seals controller
submissions, then attempts a real PC-B authenticated gateway seal. A
lost/competing remote service leaves the already-sealed PC-A untouched
and the resulting `partial` status explicitly visible.

`--apply --release` is permitted only after both seals have been
verified, the controller is still PAUSED, and a fresh live workflow/queue
audit has succeeded. The remote gate opens before the controller gate,
with observed partial failures reported honestly; there is no implicit
daemon resume. A successful `both_sealed` result still never authorizes
a ComfyUI process restart, since third-party PC-B loopback clients remain
outside the gateway. The real two-PC and eight-hour GPU qualification
remains a separate requirement.

## PC-B authenticated gateway: required real-machine checks

The optional PC-B managed-renderer gateway requires an Artifex-owned
ComfyUI child bound exclusively to `127.0.0.1:8188` and exposes the
authenticated, allowlisted Artifex API separately on port 8191.
The PC-A `comfyui.base_url` and `render_nodes.primary` must point to
the gateway, with `comfyui.gateway_token_env` set to the securely shared
`ARTIFEX_RENDER_NODE_TOKEN` variable. Validate PC-A can read `/queue`
and submit one **explicit** qualification smoke through the proxy; verify
unauthenticated requests return 401 and that an inbound LAN request to
ComfyUI's upstream port **fails** from a separate machine. After
`POST /v1/gateway/seal`, confirm future authenticated mutation attempts
return 423; after an authorized release verify they work again.

The gateway does not secure other local PC-B clients accessing loopback
or configure firewall rules, so a successful test is not blanket permission
to restart. `restart_authorized` stays false. The 14-stage two-PC
qualification and GPU soak tests are separate requirements.

## Durable, atomic controller-only submission fencing

On PC-A, `uv run artifex maintenance fence --config
.\\config\\local.yaml --apply --wait-seconds 600` pauses scheduling,
waits for persisted in-flight Packs/attempts and the actual PC-B ComfyUI
queue, and makes a second check under a SQLite transactional lock shared
by all Artifex `/prompt` submissions. If safe within **Artifex's scope**,
the closed fence is durable across PC-A restarts; it refuses future Artifex
POSTs rather than interrupting existing renders. Preview and release are
explicit. The controller's `comfyui.submission_fence_path` must be identical
for all Artifex processes on PC-A. The operation **cannot** stop third-party
clients from sending prompts directly to PC-B, so it reports
`restart_authorized=false`. A successful seal is not a 14-stage production
qualification or permission to restart arbitrary ComfyUI processes.

## Controller maintenance pause and non-destructive in-flight drain

On PC-A, `uv run artifex maintenance drain --config .\\config\\local.yaml
--apply --wait-seconds 600` pauses new controller daemon work and observes
the primary PC-B ComfyUI queue alongside persisted generation attempts and
unfinished Packs. Zero in-flight counts must be observed twice; malformed,
unreachable and external busy queues fail closed. This is not permission to
terminate an external ComfyUI instance: `restart_authorized` remains
**false** until all submitters can be fenced atomically. After verified
maintenance, resume explicitly via `artifex resume` and re-run live renderer
reconciliation. This is separate from the required 14-stage qualification.

## Automatic safe runtime revalidation after renderer changes

After a model or node change on PC-B, run
`uv run artifex onboard reconcile-renderer --config
.\\config\\render-node.yaml --wait-seconds 600`.
This read-only observer waits for the actual ComfyUI `/object_info`
production/repair workflow check to pass while verifying its live
`queue_running` and `queue_pending` state. It never sends a prompt or
interrupt, terminates any child/external renderer, or treats a mere installed
file as runtime proof. Busy, missing, unreachable and malformed states remain
explicitly unqualified; an idle queue alone is not safe permission to restart.
Only `qualify verify` can declare 14-stage production qualification.

## Prepare renderer model dependencies without manual repeated downloads

On PC-B, `artifex onboard prepare-renderer --config
.\\config\\render-node.yaml` produces a read-only live workflow/asset
report and a SHA-256-pinned plan for **previously approved** model source
bindings. To download verified missing weights into an Artifex-owned isolated
ComfyUI, explicitly add `--comfy-root PATH --apply --accept-licenses`.
Unknown model publishers, unverified choices and third-party Python
custom-node suggestions cannot be auto-installed by this command.
See [two-pc-setup.md](two-pc-setup.md) for details and the source approval
registry. A successful installation is not proof of model availability:
restart/refresh ComfyUI and re-run its live audit before beginning the
real-machine production qualification. Re-running `onboard prepare-renderer`
with `--comfy-root` safely reuses previously downloaded models only if their
physical file size and full SHA-256 still match the pinned manifest. Its
`existing_verified`, `download_required` and `next_actions` entries
explain which work is already complete and what is still blocked; an existing
correct model that ComfyUI has not indexed yet triggers a refresh/re-audit
action, **not** a second download or a false qualification PASS.

## Configure pinned production assets

For a single-PC deployment, set qualification.asset_paths in the operator config for the real production checkpoint, refiner, VAE, upscaler, LLM model and semantic model.

For the supported two-PC topology, keep controller-local assets such as the semantic model in qualification.asset_paths, use the LLM bootstrap profile for the local LLM model, and configure the primary render node's render_agent.asset_paths for the production checkpoint/refiner/VAE/upscaler. The controller obtains those renderer hashes through authenticated attestation; it does not need filesystem access to PC-B. See [two-pc-setup.md](two-pc-setup.md).

Keep qualification.require_native_windows, require_uv, require_nvidia_gpu, require_lora_validation_evidence, require_stable_asset_hashes and require_stable_workflow_snapshot enabled.

Directories are recursively hashed with stable relative paths. Final verification re-hashes local assets and re-fetches renderer attestations, so replacing a model on either PC during qualification invalidates the session.

## Activate both native Windows PCs with minimum operator steps

After each PC has its own valid configuration, use the safe
`deployment activate --role renderer --apply --install-missing` command on
PC-B, then `deployment activate --role controller --apply --install-missing`
on PC-A. The activation command installs only a missing Artifex-owned
current-user scheduled task and starts it if not already running. The same
command performs live deployment checks and reports missing/failed services;
use `--wait-seconds 600` to allow slow model loading. Without `--apply`
the command is read-only and does not start anything. Running tasks with
stale configuration and unowned scheduled tasks are refused; paths/IPs/models
remain configurable. See [two-pc-setup.md](two-pc-setup.md).

Activation does not generate an image, run `qualify start`, initiate the
8-hour soak or replace the 14-stage qualification ladder.

## Start

On PC-B, first run `uv run artifex render-node preflight --config .\\config\\render-node.yaml --json` and address each reported issue. Run the authenticated render-node service and ComfyUI on the trusted LAN. On PC-A, run `artifex setup`, select the production checkpoint explicitly, configure controller-local evaluation assets and start the local LLM endpoint. The renderer preflight checks only PC-B-local state; the PC-A doctor and qualification check the end-to-end configuration.

With the real LLM and ComfyUI already running:

    uv run artifex doctor --config .\config\local.yaml
    uv run artifex qualify start --config .\config\local.yaml --json

Save the returned session_id. Start records Windows/Python/uv versions, NVIDIA GPU UUID/driver/VRAM, Git revision, production configuration, workflow template/version/hash data, production asset SHA-256 digests, LoRA lifecycle state and the complete doctor report.

## Required ladder

### Automatically collect already-verified real Pack evidence

Instead of copying Pack IDs into many `qualify record` commands, Artifex
can inspect **only finalized Packs created after the current qualification
session started** and automatically select qualifying evidence:

```powershell
# Preview only: never records PASS, restarts services or generates an image.
uv run artifex qualify collect <SESSION> --config .\\config\\local.yaml

# Register only the stages whose actual persisted evidence passes verification.
uv run artifex qualify collect <SESSION> --apply --config .\\config\\local.yaml
```

The collector uses the exact same strict validators as manual registration,
including validated LoRAs, Scene publication tiers, continued Series IDs,
recorded retries, restart/recovery telemetry and unique finalized Packs.
It also recognizes **real delivered and authorized Discord interactions**
recorded after the session started. It never fabricates missing evidence,
changes an existing stage status, queues GPUs, or changes running services.
A stale/different-host session or changed production config is rejected.
The default bounded scan covers up to 250 post-session finalized Packs;
use `--max-packs` (1–1000) to adjust. A truncated scan is reported
explicitly. Invalid or altered archived manifests are skipped and shown
as issues, never treated as valid Packs. Repeated `--apply` runs are safe:
already recorded stages stay untouched.

The 8-hour overnight soak and archive reproduction **still require their
dedicated real observation and replay** (`qualify soak-run` and
`qualify archive-reproduce --confirm-render`) and cannot be satisfied
by `qualify collect`. An overall production-ready decision is **only**
available through `qualify verify` after all stages actually pass.

Use `uv run artifex qualify stages` to list the canonical names. All of these are required: doctor, single_character, lora_required, duo, group, public_member, series_continuation, forced_retry, restart_generation, backend_recovery, unattended_multi_pack, overnight_soak, discord_controls, archive_reproduction.

Normal Pack stages are recorded with finalized Pack IDs, for example:

    uv run artifex qualify record <SESSION> single_character --status pass --pack-id <PACK>
    uv run artifex qualify record <SESSION> lora_required --status pass --pack-id <PACK>
    uv run artifex qualify record <SESSION> duo --status pass --pack-id <PACK>
    uv run artifex qualify record <SESSION> group --status pass --pack-id <PACK>
    uv run artifex qualify record <SESSION> public_member --status pass --pack-id <PACK>

Series continuation requires at least two finalized Packs from the same persisted Series:

    uv run artifex qualify record <SESSION> series_continuation --status pass --pack-id <EP1> --pack-id <EP2>

Forced retry must contain actual persisted retry history:

    uv run artifex qualify record <SESSION> forced_retry --status pass --pack-id <PACK>

For restart qualification, terminate and restart Artifex while the selected Pack is generating. Recovery must persist recovery.pack_checked telemetry. Repeated Comfy prompt IDs are rejected automatically:

    uv run artifex qualify record <SESSION> restart_generation --status pass --pack-id <PACK>

For backend recovery, disconnect/restart both ComfyUI and the configured LLM during production. The health supervisor must persist unhealthy -> healthy transitions for both after the session started:

    uv run artifex qualify record <SESSION> backend_recovery --status pass --pack-id <PACK>

Unattended production requires at least three finalized Packs:

    uv run artifex qualify record <SESSION> unattended_multi_pack --status pass --pack-id <PACK1> --pack-id <PACK2> --pack-id <PACK3>

For minimal manual interaction, after the production controller and
PC-B services are already running, use **one command** on PC-A:

```powershell
uv run artifex qualify soak-run --config .\\config\\local.yaml
```

This runs Doctor, creates a session, watches a real 8-hour window,
revalidates the persisted JSONL/SHA-256, and automatically records
`overnight_soak` PASS only if every required observation succeeds.
The command prints the session ID and output evidence location, and
reports the 14-stage ladder still unfinished. The full production
qualification is **not** automatically declared complete. The
controller must already be making genuine Packs, and the PC-B renderer
must already be accessible. For an existing partially qualified
session, supply `--session-id <SESSION>`. A failed run retains
evidence and exits nonzero without registering PASS. No Docker
dependency, GPU-generating test loop, or service restarts are added.

**Advanced manual commands** for independent observation and review:

The overnight soak MUST use an actual read-only observer trace from
the same qualification session. Start the normal Artifex daemon first,
then on PC-A run the following in another terminal (saving evidence
under `qualification.evidence_dir`):

```powershell
uv run artifex qualify soak-observe `
  --config .\\config\\local.yaml `
  --session-id <SESSION> `
  --output .\\data\\qualification\\soak-unique.jsonl `
  --hours 8 --sample-seconds 300

uv run artifex qualify soak-check `
  --config .\\config\\local.yaml `
  --evidence .\\data\\qualification\\soak-unique.jsonl

uv run artifex qualify record <SESSION> overnight_soak `
  --status pass `
  --soak-evidence .\\data\\qualification\\soak-unique.jsonl `
  --config .\\config\\local.yaml
```

The last command hashes the evidence automatically and performs the
strict review. It only records PASS if the session identity, controller,
renderer, configuration, model hashes, actual duration, sampling,
service health, at least 3 new Packs and zero serious failures match.
`qualify verify` rechecks the original evidence and digest. Do not
use manually entered elapsed time or resource values; these are no
longer accepted as stage evidence. Without `--session-id`, a standalone
observer trace is diagnostic only and cannot qualify a session.

When Discord is enabled, **manually claiming that six commands worked is not sufficient**. Configure the real `discord.guild_id`, `discord.channel_id` and allowed operator user/roles before starting the qualification session. From that authorized operator account, run `status`, `pause`, `resume`, `approve`, `reject`, and `retry` as **real Discord slash commands or review buttons** in the configured guild/channel. `pause` and `resume` must actually change the scheduler state; no-op "already paused/running" responses are not counted. Review actions must genuinely succeed on review items. Each command must produce a successful Discord response after the session started.

The bot records a durable `discord.interaction_completed` event **only after the Discord API response has been sent**, recording the Discord interaction ID, operator, channel, guild and command, without recording command arguments or review content. To register the stage, run:

```powershell
uv run artifex qualify record <SESSION> discord_controls --status pass --config .\\config\\local.yaml
```

The qualification service independently reads SQLite and verifies six **distinct**, authorized, successfully delivered interaction IDs against the current configuration and session start time. It persists the actual event IDs and rechecks them on `qualify verify`. Hand-entered `--detail verified_commands=...` values, events from a different guild/channel, denied interactions, duplicate IDs and events recorded before the session **cannot** satisfy the gate. If Discord is disabled, this stage alone is skipped automatically.

Archive reproduction now requires an **actual, new ComfyUI execution** of one archived selected attempt, rather than operator-supplied booleans. The selected Pack must have a valid manifest, an archived original image, persisted compiled prompts/LoRA plan, Seed, workflow ID, and original ComfyUI prompt ID. Those fields must agree with SQLite and current backend settings. Old Packs without complete provenance are **not automatically qualified**.

On PC-A, after Artifex and the renderer are healthy, explicitly permit one real GPU job:

```powershell
uv run artifex qualify archive-reproduce <SESSION> <PACK> `
  --confirm-render --config .\\config\\local.yaml
```

The command re-submits the selected attempt with its archived prompt/negative prompt, LoRA weights, Seed, and versioned template. It writes a fresh image to a unique qualification evidence directory (never overwriting the archived original), records the **new** ComfyUI prompt ID, independently hashes both images, and saves a JSON proof plus a SQLite `qualification.archive_replayed` event. If the SHA-256 digests of the **actual generated image bytes** differ, the command exits nonzero, retains the evidence for inspection, and does not register PASS. If they match, it records PASS using the evidence file hash; `qualify verify` recomputes the hashes, checks the original selected attempt, the archive manifest, and the replay event.

Exact byte-for-byte reproducibility can be affected by GPU kernels, ComfyUI nodes and runtime versions. **A successful new render with different pixels is useful diagnostic evidence, but never an archive-reproduction PASS** under this strict rule. This is a distinct safety gate, not an indicator of semantic similarity.

`--detail reproduction_verified=true --detail hash_match=true` is explicitly rejected; those previously documented self-reports are not valid production evidence. CI uses an injected fake backend to test the fail-closed verifier, and does not replace a real PC-A/PC-B replay.

## Final gate

Inspect progress and then perform strict revalidation:

    uv run artifex qualify status <SESSION> --config .\config\local.yaml
    uv run artifex qualify verify <SESSION> --config .\config\local.yaml --json

`ready=true` is the only acceptable result for closing #40 and declaring the operator workflow production-ready. Final verification rechecks Pack/archive evidence, restart/backend telemetry, current asset hashes and current workflow snapshots.
