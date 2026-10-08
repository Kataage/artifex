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

## Qualification

Complete the PC-B preflight first; then on PC-A run `artifex setup`,
configure the explicit production checkpoint and controller-local semantic
assets, start the local LLM server and run `artifex doctor`. PC-B preflight
does not prove network reachability from PC-A; the controller must check LAN
connectivity and the remote token separately.

For the split topology, qualification records controller-local hashes and primary-renderer attestations in one session. Final verification re-fetches the remote attestation and rejects model drift.

The controller no longer needs a local filesystem path to the renderer checkpoint/VAE/upscaler. The primary renderer must expose authenticated attestation evidence for those assets.
