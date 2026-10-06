# Target Windows Production Qualification

Issue #40 is the final production-readiness gate. Architecture/CI evidence alone is not sufficient. Run this procedure on the actual native Windows host with the intended models, ComfyUI workflow, LoRAs and services. Docker, Podman and WSL are optional and are not accepted as substitutes.

## Configure pinned production assets

For a single-PC deployment, set qualification.asset_paths in the operator config for the real production checkpoint, refiner, VAE, upscaler, LLM model and semantic model.

For the supported two-PC topology, keep controller-local assets such as the semantic model in qualification.asset_paths, use the LLM bootstrap profile for the local LLM model, and configure the primary render node's render_agent.asset_paths for the production checkpoint/refiner/VAE/upscaler. The controller obtains those renderer hashes through authenticated attestation; it does not need filesystem access to PC-B. See [two-pc-setup.md](two-pc-setup.md).

Keep qualification.require_native_windows, require_uv, require_nvidia_gpu, require_lora_validation_evidence, require_stable_asset_hashes and require_stable_workflow_snapshot enabled.

Directories are recursively hashed with stable relative paths. Final verification re-hashes local assets and re-fetches renderer attestations, so replacing a model on either PC during qualification invalidates the session.

## Start

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

The overnight soak must meet the configured minimum duration and record fatal error count plus peak VRAM/RAM, minimum free disk and telemetry row count:

    uv run artifex qualify record <SESSION> overnight_soak --status pass --detail duration_hours=8.5 --detail fatal_errors=0 --detail peak_vram_mib=11000 --detail peak_ram_mib=24000 --detail minimum_free_disk_gib=100.0 --detail telemetry_rows=5000

When Discord is enabled, verify status, pause, resume, approve, reject and retry against the real guild/channel. If Discord is disabled, only this stage may be skipped automatically.

Archive reproduction requires a finalized Pack whose manifest SHA still matches both SQLite payload and checkpoint provenance. Re-run the archived provenance against the pinned assets/workflow, compare the selected reproduction, then record:

    uv run artifex qualify record <SESSION> archive_reproduction --status pass --pack-id <PACK> --detail reproduction_verified=true --detail hash_match=true

## Final gate

Inspect progress and then perform strict revalidation:

    uv run artifex qualify status <SESSION> --config .\config\local.yaml
    uv run artifex qualify verify <SESSION> --config .\config\local.yaml --json

`ready=true` is the only acceptable result for closing #40 and declaring the operator workflow production-ready. Final verification rechecks Pack/archive evidence, restart/backend telemetry, current asset hashes and current workflow snapshots.
