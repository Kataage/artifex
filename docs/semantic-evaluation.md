# Production semantic evaluation

Artifex uses a real local SigLIP2 embedding model for production similarity and
reference-grounded identity evaluation. The legacy hash/8x8-thumbnail embedder
remains available only as an explicitly enabled degraded diagnostic fallback.

The production path is Windows-native and does not require Docker or an external
vector database. Embeddings and their source/model metadata are persisted in the
Artifex SQLite database.

## Install

Install the semantic runtime alongside the normal environment:

```powershell
uv sync --extra semantic
```

The default model is `google/siglip2-base-patch16-224`. The default semantic
device is CPU so ComfyUI retains GPU VRAM. Set `evaluation.semantic_device` to
`cuda` or `auto` only when that tradeoff is intentional.

## Character identity references

Each character profile can declare one or more `reference_image_dirs`.
Production identity evaluation scans those directories for PNG/JPEG/WebP files
and supplies a bounded set to both the semantic identity gate and the vision
evaluator.

Identity references are never mixed with novelty references:

- **Identity**: only the requested character's curated local reference set.
- **Duplicate hard gate**: finalized historical outputs with the exact same
  character set, prioritizing the same scene role/composition/camera family.
- **Novelty**: same-character history plus a bounded global-diversity sample.
- **Continuity**: adjacent Pack images, handled independently.

For multi-character scenes, every requested character must have identity
reference coverage when `identity_reference_required=true`.

## Persistent semantic index

Artifex stores semantic vectors with:

- subject type and stable subject ID;
- text/image modality;
- source SHA-256;
- provider, model, resolved model revision and dimensions;
- vector values and bounded metadata.

A changed text/image hash invalidates the cached vector. A changed model or
revision creates a separate model identity, so vectors from incompatible models
are not silently compared.

Historical reference retrieval scans up to
`evaluation.historical_reference_scan_limit` finalized outputs (default 2000);
there is no last-20 cutoff.

## Required calibration

Production readiness requires a validated calibration profile generated from a
curated corpus of actual ILXL/Hololive outputs. This avoids treating generic
CLIP/SigLIP cosine values as universal thresholds.

A calibration manifest contains six labeled pair sets:

```json
{
  "corpus_id": "hololive-ilxl-2026-10",
  "profile_id": "siglip2-hololive-ilxl-v1",
  "minimum_pairs_per_class": 8,
  "minimum_balanced_accuracy": 0.9,
  "identity_positive_pairs": [
    {"left": "images/kanata-output-01.png", "right": "refs/kanata-official-01.png"}
  ],
  "identity_negative_pairs": [
    {"left": "images/kanata-output-01.png", "right": "refs/similar-palette-other-character.png"}
  ],
  "duplicate_positive_pairs": [
    {"left": "images/scene-a-original.png", "right": "images/scene-a-near-duplicate.png"}
  ],
  "duplicate_negative_pairs": [
    {"left": "images/scene-a-original.png", "right": "images/scene-a-distinct-composition.png"}
  ],
  "text_paraphrase_pairs": [
    {
      "left": "angel sitting half outside an open window",
      "right": "winged girl perched in the open window with her body partly outside"
    }
  ],
  "text_distinct_pairs": [
    {
      "left": "angel sitting half outside an open window",
      "right": "underwater action battle in a ruined city"
    }
  ]
}
```

Use enough examples for every class; the default example above is illustrative
only and is intentionally too small for production qualification.

Run calibration:

```powershell
uv run artifex semantic calibrate path\to\manifest.json --json
```

The calibration run also backfills every selected historical Concept and every
existing image selected by a finalized Pack into the persistent SQLite semantic
index. Because production requires calibration before normal operation, this
establishes a complete historical baseline; subsequent generated images are
indexed during reference-grounded evaluation.

By default Artifex writes the measured profile to
`data/calibration/siglip2-hololive-ilxl-v1.json`. The profile records the
resolved model revision and measured operating points for:

- reference-grounded identity hard/accept thresholds;
- image duplicate hard threshold;
- paraphrased-concept text similarity threshold.

The profile is marked validated only when identity, duplicate and paraphrase
classifiers each meet the manifest's minimum balanced accuracy. A non-validated
run is still written for diagnosis but exits non-zero and is not production-ready.

Inspect the active result:

```powershell
uv run artifex semantic status
uv run artifex doctor
```

At production startup Artifex loads the validated profile, pins the semantic
model to its calibrated revision when no explicit revision was configured, and
applies the measured thresholds. A missing, invalid, mismatched, or unvalidated
profile blocks production readiness.

## Corpus guidance

Identity positives should pair actual generated images with correct character
references across outfits, angles, expressions and lighting. Identity negatives
should deliberately include other characters with similar hair/clothing colors
and similar compositions.

Duplicate positives should contain outputs a human would judge "the same
picture/idea" despite seed, crop, minor pose, or prompt wording changes.
Duplicate negatives should include the same character and same general theme
but materially different pose/composition.

Text paraphrase positives should describe the same visual concept with
substantially different wording. Text negatives should include plausible but
visually different concepts. This directly checks that concept deduplication is
semantic rather than token matching.
