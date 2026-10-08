# Target Windows Production Qualification

Issue #40 is the final production-readiness gate. Architecture/CI evidence alone is not sufficient. Run this procedure on the actual native Windows host with the intended models, ComfyUI workflow, LoRAs and services. Docker, Podman and WSL are optional and are not accepted as substitutes.

## Configure pinned production assets

For a single-PC deployment, set qualification.asset_paths in the operator config for the real production checkpoint, refiner, VAE, upscaler, LLM model and semantic model.

For the supported two-PC topology, keep controller-local assets such as the semantic model in qualification.asset_paths, use the LLM bootstrap profile for the local LLM model, and configure the primary render node's render_agent.asset_paths for the production checkpoint/refiner/VAE/upscaler. The controller obtains those renderer hashes through authenticated attestation; it does not need filesystem access to PC-B. See [two-pc-setup.md](two-pc-setup.md).

Keep qualification.require_native_windows, require_uv, require_nvidia_gpu, require_lora_validation_evidence, require_stable_asset_hashes and require_stable_workflow_snapshot enabled.

Directories are recursively hashed with stable relative paths. Final verification re-hashes local assets and re-fetches renderer attestations, so replacing a model on either PC during qualification invalidates the session.

## Start

On PC-B, first run `uv run artifex render-node preflight --config .\\config\\render-node.yaml --json` and address each reported issue. Run the authenticated render-node service and ComfyUI on the trusted LAN. On PC-A, run `artifex setup`, select the production checkpoint explicitly, configure controller-local evaluation assets and start the local LLM endpoint. The renderer preflight checks only PC-B-local state; the PC-A doctor and qualification check the end-to-end configuration.

With the real LLM and ComfyUI already running:

    uv run artifex doctor --config .\config\local.yaml
    uv run artifex qualify start --config .\config\local.yaml --json

Save the returned session_id. Start records Windows/Python/uv versions, NVIDIA GPU UUID/driver/VRAM, Git revision, production configuration, workflow template/version/hash data, production asset SHA-256 digests, LoRA lifecycle state and the complete doctor report.

## Required ladder

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

When Discord is enabled, verify status, pause, resume, approve, reject and retry against the real guild/channel. If Discord is disabled, only this stage may be skipped automatically.

Archive reproduction requires a finalized Pack whose manifest SHA still matches both SQLite payload and checkpoint provenance. Re-run the archived provenance against the pinned assets/workflow, compare the selected reproduction, then record:

    uv run artifex qualify record <SESSION> archive_reproduction --status pass --pack-id <PACK> --detail reproduction_verified=true --detail hash_match=true

## Final gate

Inspect progress and then perform strict revalidation:

    uv run artifex qualify status <SESSION> --config .\config\local.yaml
    uv run artifex qualify verify <SESSION> --config .\config\local.yaml --json

`ready=true` is the only acceptable result for closing #40 and declaring the operator workflow production-ready. Final verification rechecks Pack/archive evidence, restart/backend telemetry, current asset hashes and current workflow snapshots.
