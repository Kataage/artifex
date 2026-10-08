# Two-PC Artifex topology

Artifex supports a split native-Windows deployment where the controller and renderer are separate machines.

- **PC-A / controller**: Artifex, local LLM, research/planning, SQLite, archive, evaluation, Discord/Patreon integration.
- **PC-B / render node**: ComfyUI, production checkpoint, refiner, VAE, upscaler, LoRAs and the lightweight Artifex render-node attestation server.

No Docker, WSL, shared SMB output folder, or duplicate LoRA/model copy on PC-A is required.

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

Set the same strong token on PC-A and PC-B through the `ARTIFEX_RENDER_NODE_TOKEN` environment variable. Do not put the token in YAML.

Before starting the agent, run the **read-only native-Windows preflight** on PC-B.
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
the local llama.cpp backend has `auto_download: true`. Bootstrap installs the
GGUF but does not start llama-server: configure the compatible local
llama.cpp server on PC-A separately, listening on `127.0.0.1:8899`.

A different model can be selected by adding another bootstrap profile or by using a local profile; Artifex is not tied permanently to Spark.

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

## Qualification

Complete the PC-B preflight first; then on PC-A run `artifex setup`,
configure the explicit production checkpoint and controller-local semantic
assets, start the local LLM server and run `artifex doctor`. PC-B preflight
does not prove network reachability from PC-A; the controller must check LAN
connectivity and the remote token separately.

For the split topology, qualification records controller-local hashes and primary-renderer attestations in one session. Final verification re-fetches the remote attestation and rejects model drift.

The controller no longer needs a local filesystem path to the renderer checkpoint/VAE/upscaler. The primary renderer must expose authenticated attestation evidence for those assets.
