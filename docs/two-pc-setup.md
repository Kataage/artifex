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
