# Target Windows Production Qualification

Issue #40 is the final production-readiness gate. Architecture/CI evidence alone is not sufficient. Run this procedure on the actual native Windows host with the intended models, ComfyUI workflow, LoRAs and services. Docker, Podman and WSL are optional and are not accepted as substitutes.

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

## One-command PC-A evidence-aware qualification handoff (Issue #93 / #40)

After PC-A and PC-B both run the latest Artifex, the **default** PC-A
handoff now includes the authenticated PC-B owner-readiness cache, current
owner audit and workflow checks automatically. No JSON copy, manual
PC-B probe command, renderer restart, GPU submission or stage mutation:

```powershell
uv run artifex qualify handoff --config .\config\local.yaml --json
```

The `authenticated_owner_evidence_*` fields and embedded
`overview.pc_b_owner_evidence` distinguish whether the protected
PC-B evidence was actually correlated. Even when the usual PC-A checks and
workflow prerequisites are green, a missing, stale, mismatched, wrong-node
or unavailable PC-B owner report blocks the session-start suggestion rather
than silently using the older less-complete handoff criteria. The blockers
include `pc_b_authenticated_owner_evidence_missing_or_unverified`.
Native evidence still missing for supervisor-loss child survival, real
production GPU renders, 8-hour soak and the complete 14-stage ladder remains
visible. A successful handoff **only suggests** explicitly beginning the
qualification process: it is never a production PASS.

For intentionally *legacy advisory-only* diagnostics (e.g. PC-B software
has not yet been upgraded), `--no-pc-b-owner-live` disables just the
new correlation gate. This is **not** a substitute for proving Issue #93 or
#40 and does not authorize launching ComfyUI or production workloads. Do
not use this opt-out to interpret an unverified PC-B as production-ready.

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
## Authenticated PC-B survival report pickup without file copying

PC-B writes an immutable observer trace automatically to the configured
`render_agent.survival_evidence_dir` (default `data/qualification/survival`)
when `render-node survival-observe` is run WITHOUT `--output`. Existing
`--output` remains available, but a custom path outside the spool cannot
be picked up through this API. No supervisor exit is triggered by this command.

```powershell
uv run artifex render-node survival-observe --config .\config\render-node.yaml --duration-seconds 600 --interval-seconds 10 --json
```

When the PC-B attestation service is running, its Bearer-protected
`GET /v1/survival-evidence` reads ONLY the newest correctly named report
in the configured local spool (up to 512 candidates). The URL accepts
NO filename, file path, query parameters, POST body, or remote commands.
An absent, symlinked, malformed or oversize newest file fails closed:
the service never searches backward for an older apparently good trace.
Files survive the attestation service stopping and can be read once it
returns. A temporarily offline service remains unavailable: there is
no separate relay or hidden restart of the existing ComfyUI renderer.

PC-A can request and verify that latest file with NO manual file copy:

```powershell
uv run artifex qualify overview --config .\config\local.yaml --pc-b-survival-live --json
uv run artifex qualify handoff --config .\config\local.yaml --pc-b-survival-live --json
```

This is an explicit optional read-only review. Passing both
`--pc-b-survival-live` and `--pc-b-survival-report` is rejected.
The previous file-based evidence review is still supported.
PC-A requires its configured node ID and Bearer token, refuses HTTP
redirects, caps network bytes, verifies the SHA-256 of the exact
saved UTF-8 payload, replays the complete trace, and cross-checks
against the fresh owner audit/attestation if the service is available.

**Bearer-authenticated transport is not independently signed evidence
of what happened on PC-B.** A valid remote response can still carry
tampered historical observations. Even `replayed_and_live_identity_matched`
does not establish that a natural exit really occurred and never
authorizes a restart, production GPU job, stage PASS or Issue #93/#40 closure.
If there is no naturally observed exit, the newest report remains
`inconclusive` and the reviewer must not silently use an older success.
## PC-A review of saved PC-B natural-exit evidence (Issue #93)

PR #120 can record a passive survival trace on the real PC-B, even when its
supervisor/attestation service naturally goes offline. A copied (or mounted
read-only shared) JSON can now be examined by PC-A in its ordinary reports:

```powershell
uv run artifex qualify overview --config .\config\local.yaml --pc-b-owner-live --pc-b-survival-report .\data\pair\pc-b-survival-unique.json --json
uv run artifex qualify handoff --config .\config\local.yaml --pc-b-survival-report .\data\pair\pc-b-survival-unique.json --json
```

An explicit `--pc-b-survival-report` is optional; without it the existing
commands behave as before. PC-A enforces a size limit and non-symlink file,
strictly parses the saved trace, recomputes the status from all observation
samples and verifies order, wall-clock/monotonic agreement, host, configured
node, process identity and the original Task Scheduler supervisor-PID absence
claims. It **never trusts the saved status string alone**.

PC-A then tries to independently query the *currently* authenticated PC-B
owner audit and node attestation. `replayed_and_live_identity_matched` means
the copied trace is internally consistent and the same ComfyUI process is
seen now; `replayed_historical_identity` means today's live PID/creation time
differs. `replayed_live_unavailable` means a coherent historical trace but
no reachable fresh PC-B. `blocked` and `mismatch` reject invalid details.

These statuses are **review only, not real-machine attestation**. JSON can
be forged, including all its timestamps and samples. The trace's source
and historical event remain unauthenticated, even when the live process
matches. No status closes Issue #93 or #40, adds a qualification stage PASS,
authorizes a renderer restart, or counts as a real 8-hour GPU soak.
By design, when an explicitly requested trace cannot be correlated with
current PC-B, handoff cannot recommend a qualification session start.

While this release removes manual reconstruction of the report, securely
transporting a trace from PC-B to PC-A is still required. Never stop a
live ComfyUI task solely to create a survival trace.
## PC-A + PC-B read-only installation audit (Issue #93 / #40)

Use the ONE read-only PC-A command below to check whether the two real
Windows installations have the prerequisites for passive monitoring:

```powershell
uv run artifex qualify pair-install-audit --config .\config\local.yaml --json
```

The local PC-A inspection verifies the **Artifex-Controller** Task Scheduler
registration, current Python action/config ownership and state. PC-A also
fetches the fresh PC-B `GET /v1/installation-audit` over the already configured
Bearer-authenticated attestation URL. PC-B separately inspects the protected
`Artifex-Renderer` and `Artifex-Survival-Observer` Task Scheduler actions,
configured managed ComfyUI mode and passive evidence-spool metadata.

Results include `ready_for_passive_monitoring`, `pc_a_needs_setup`,
`pc_b_needs_setup`, `pc_b_unreachable`, `pc_b_unconfigured`, and
`unsupported_platform`. Each check distinguishes missing registration,
registered-but-not-running, unsafe/foreign/task-policy drift, or unavailable
native inspection. A not-yet-created survival evidence directory is
normal before any event; the report lists the missing evidence separately.

**Security:** the PC-B endpoint requires Bearer authentication, accepts
no query parameters or file paths, and returns only small typed status
codes (not original Python arguments, task execution paths, receipt files,
token variables or service credentials). PC-A rejects redirects,
over-sized responses, wrong node IDs and stale remote snapshots. Offline
PC-B is reported as unreachable; there is no attempt to start it.

All output includes `production_qualified=false` and does not certify real
GPU output, child survival, 8-hour soak or any of the 14 actual stages.
No commands, GPU jobs, Task Scheduler changes, process starts/stops or
automatic repairs occur. For a missing PC-B observer, run the **dry-run**
`uv run artifex startup observer-enable --config config/render-node.yaml --json`
directly on PC-B; only operator-requested `--apply` changes its own task.
## Independent unattended Windows survival watcher (PC-B)

The passive watcher has its own native Windows Task Scheduler registration
named `Artifex-Survival-Observer`, **separate from** the protected
`Artifex-Renderer` task and the ComfyUI process. It starts automatically
at subsequent interactive logons once explicitly installed, and does not
depend on the renderer supervisor being alive. A read-only command checks
the registration without changing Windows tasks:

```powershell
uv run artifex startup observer-enable --config .\config\render-node.yaml --json
```

An operator can explicitly install/start ONLY the observer, once:

```powershell
uv run artifex startup observer-enable --config .\config\render-node.yaml --apply --json
```

The dry-run is the default. An existing foreign/unowned task is refused,
and a drifted, active observer task cannot be replaced. The observer
registration uses the existing current user's limited Task Scheduler
principal, protected no-hard-terminate/IgnoreNew settings and unlimited
execution time. It will NEVER modify/start/stop/re-register the renderer
task or ComfyUI. No renderer task actions are issued on GET or sampling.

For local supervised diagnostics (without installing a task):

```powershell
uv run artifex render-node survival-watch --config .\config\render-node.yaml --poll-seconds 15 --window-seconds 3600 --max-seconds 120 --json
```

With no `--max-seconds`, the watcher continues until the independent
process exits; no restart is attempted on the actual ComfyUI. It only
arms after an actual verified Running Task Scheduler supervisor, valid
receipt, process identities and exclusive TCP. It periodically checks
the original supervisor PIDs, retains samples across a natural exit,
and saves strictly checked terminal/partial evidence into the PC-B
`survival_evidence_dir` (read by authenticated `GET /v1/survival-evidence`).
A healthy unchanged renderer yields no repeating JSON files. Natural
exit is never artificially caused; observations remain inconclusive or
blocked if identity changes, samples are missed or the real supervisor
does not exit. Long monitoring windows rotate without ever claiming a
success from a sample gap or a mere Task Scheduler Ready state.

**Critical:** a logon-triggered task only starts after the task has
been explicitly installed; this change does not remotely activate
anything on the operator's physical PC-B. Windows CI tests use isolated
simulations. Even a naturally observed trace does NOT authenticate
historical events, qualify GPU production or close Issues #93/#40.
## Passive native PC-B supervisor-loss survival observation (Issue #93)

Run on the **actual Windows PC-B** from an independent terminal while the
managed ComfyUI and Artifex renderer task are already running. The observer
only reads Task Scheduler, CIM process identity, receipts and TCP; it does
not create/stop/restart or replace any supervisor, service or GPU process.
Never stop the live renderer to force this diagnostic to succeed.

```powershell
uv run artifex render-node survival-observe --config .\config\render-node.yaml --output .\data\qualification\pc-b-survival-unique.json --duration-seconds 600 --interval-seconds 10 --json
```

The baseline must establish a Running managed task with exact supervisor
PID plus creation time and a schema-2-owned ComfyUI listener. Only an
unforced natural transition to Ready can support the outcome. Every
original supervisor PID must independently be absent from CIM; the same
ComfyUI PID, creation time, launcher PID, receipt, identity and exclusive
TCP listener must remain verified for two separated after-exit samples.
Unexpected supervisor replacement/reappearance, process/PID reuse, gaps
in monitoring, unsupported task/legacy receipt, foreign listener or a
missing/unsafe task policy fail closed.

Possible results: `observed_after_supervisor_absence`, `inconclusive`
(transition never witnessed), or `blocked` (missing/contradictory evidence).
All attempts write a NEW, exclusive JSON evidence file; a non-observed
result exits nonzero. Output is never overwritten or symlink-followed.
An independent PC-B observer remains able to sample after the managed
supervisor/attestation service goes offline.

This is **only natural-process-continuity evidence**; it does not prove
reattachment, forced-failure safety, GPU recovery, eight-hour soak, or all
14 qualification stages. Even if naturally witnessed, the report keeps
`child_survival_qualified=false`, `issue_93_closure_authorized=false`,
`renderer_restart_authorized=false` and `production_qualified=false`.
CI simulations cannot stand in for actual physical-PC observations.
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

### Parallel safety: one authoritative session, no lost PASS evidence

PC-A's periodic `qualify collect --apply` and operator `qualify record`
may execute at the same time. Their shared qualification session now has a
**persistent cross-process lock** (Windows byte-range / POSIX advisory,
standard library only). The entire fresh-load, actual evidence validation,
stage mutation and atomic JSON replacement is serialized. Concurrent PC-B
owner snapshots use the same lock and cannot erase recorded stages.
A lock contention timeout leaves the session unchanged instead of recording
a speculative PASS. Lock files persist and **must not be deleted while
Artifex or a qualifying command is running**.

Once a stage has been recorded, the exact same result may be retried without
a second write, but competing stage values or different Pack IDs cannot
silently replace an earlier PASS/FAIL/SKIPPED record. A concurrently running
collector reports the already recorded value instead of inventing a new PASS.
For intentional requalification after corrected evidence, start a **new
explicit qualification session** rather than overwriting a previous record.
This is not actual native two-PC acceptance or eight-hour soak evidence.

### PC-A automatic evidence collection during native daemon operation

After starting a qualification session with
`uv run artifex qualify start --config .\\config\\local.yaml --json`,
Artifex stores the latest explicit session binding in the **PC-A SQLite DB**.
Only a session with a ready doctor/native baseline and the same hostname and
configuration is eligible. The `qualify start` output exposes
`auto_collection.active`, `enabled`, and the configured cadence.

While the normal **PC-A `artifex daemon`** is running, it scans **already
finalized** Packs and authentic Discord telemetry about once every 15 minutes
(default) and registers only stages that pass the existing authoritative
qualification validators. No separate repeated `qualify collect --apply`
command, Pack ID entry, or additional service is required. The collector
runs in a worker thread to avoid occupying the normal production scheduler;
failures are isolated as `qualification.auto_collect_failed` (error type only).
Newly verified stages generate `qualification.auto_collected` telemetry.
Repeated scans cannot re-register already passed stages. A newly started,
unready session replaces the old active binding and prevents accidental reuse.

Optional settings in the PC-A YAML:

```yaml
qualification:
  auto_collect_enabled: true
  auto_collect_interval_seconds: 900
  auto_collect_scan_limit: 10000
```

Set `auto_collect_enabled: false` to disable all automatic stage recording;
the manual preview and apply commands remain available. Merely updating
Artifex or launching a daemon does not activate old sessions: start a new
session explicitly. Auto-collection does **not** generate images, start or
restart ComfyUI, grant ownership permission, run an overnight soak, replay
archived GPU outputs, or claim full production qualification. The separate
soak and archive-reproduction stages still require actual evidence.
`qualify verify` remains the sole overall production acceptance decision.

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

**Unattended multi-Pack proof is stricter than a total of three finished
Packs.** Native Artifex `daemon` records a unique run ID when it enters
its sustained `run_forever` loop, and associates only *persisted, finalized
archive checkpoints* reached by that scheduler with its run ID.
`unattended_multi_pack` requires at least three **distinct post-session
Packs** completed in one continuous daemon run and matching
`daemon.loop_started` / `daemon.pack_finalized` telemetry. Running
`run_once`, creating manual finalized Packs, or combining different daemon
runs does not satisfy this stage. `qualify collect` automatically selects
the matching Pack IDs; no manual list is needed. Absent/pruned telemetry
leaves the stage pending, not a false PASS.

`daemon.loop_started`, `daemon.pack_finalized`, and
`daemon.loop_stopped` are retained outside the bounded routine health-event
queue. Otherwise routine telemetry pruning during a long soak could silently
erase a legitimate original daemon-run marker. These small provenance events
remain in SQLite until a separate evidence retention/archival policy is
implemented; do not manually delete them before production qualification.

The recorded event sequence is a proof of sustained scheduler execution
for the selected Packs, **not** proof of eight uninterrupted hours or
permission to restart a live ComfyUI renderer. The separate soak and
recovery stages remain mandatory.
It also recognizes **real delivered and authorized Discord interactions**
recorded after the session started. It never fabricates missing evidence,
changes an existing stage status, queues GPUs, or changes running services.
A stale/different-host session or changed production config is rejected.
The scan now walks all post-session finalized Packs using deterministic
keyset pagination, **250 Packs per query** by default (`--max-packs` sets
the page size, 1–1000). The default total safety ceiling is **10,000 Packs**;
use `--scan-limit` (1–100,000) only if a very long session exceeds that.
A partial scan sets `scan_truncated=true` / `scan_incomplete=true`, and
must never be interpreted as proof that no further eligible Packs exist.
The same `qualify collect` command can be run again safely, without manually
tracking a cursor or entering Pack IDs. Invalid or altered archived manifests are skipped and shown
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
