# Hololive character catalog

Artifex ships a versioned roster manifest for the operator's requested Hololive scope.
The current catalog is `hololive@2026.10.06.1`, checked on 2026-10-06.

## Scope

The packaged catalog contains 88 stable character IDs:

- hololive Japan, including historical graduates and contract-ended talents
- hololive Indonesia
- hololive English
- hololive DEV_IS
- ASOBI★MAWARI-TAI!
- historical hololive China
- current `Affiliate` and `Alum` statuses represented explicitly

HOLOSTARS, HOLOSTARS English, holoAN and office staff are deliberately excluded.
The exclusion is part of the catalog contract and is checked by the roster audit.

The primary source of truth is the official hololive production TALENTS roster.
Historical entries no longer shown on the current roster are supplemented with
COVER official notices. Hitomi Chris retains a clearly marked contemporary
secondary archive provenance because the original 2018 operations announcement
is no longer available on the current official site.

Every profile carries provenance records, lifecycle status, branch/group/generation,
canonical identity tags, aliases, model-family compatibility, policy profile,
readiness, an explicit outfit model and a local reference-image slot. Copyrighted
reference images are never bundled.

## Bootstrap

On a fresh Windows checkout:

```powershell
uv sync
uv run artifex characters bootstrap-hololive --json
uv run artifex characters audit --json
```

By default bootstrap writes `hololive.yaml` under the first configured
`characters.profile_dirs` directory. It never overwrites an existing generated
catalog unless `--force` is supplied.

A generated file is ordinary CharacterRegistry YAML. Operators can therefore add
local overrides or extra metadata without teaching the runtime about the packaged
manifest format. Duplicate IDs are reported by `characters audit`.

## Reference images

Each profile receives a local slot:

```text
references/characters/<stable-character-id>
```

Place operator-owned/authorized reference material there. Artifex records only the
path in the catalog; media is not committed to the repository. Issue #35 consumes
these character-specific references for production identity evaluation.

## Outfits and LoRAs

Outfits are first-class structured records rather than prompt prose. Every character
starts with a `default`/primary-outfit record, and additional official/common
wardrobe variants can add canonical tags, aliases, generation notes and optional
`preferred_lora_ids` without changing the CharacterProfile schema.

The character catalog itself does not claim that a discovered LoRA is production
safe. Local LoRA discovery/validation and identity qualification remain independent
gates.

## Updating the roster

Do not edit the generated runtime YAML as the canonical roster source. For a roster
refresh:

1. Re-check the official TALENTS page and relevant COVER notices.
2. Copy the packaged manifest to a new dated/versioned filename.
3. Add or update stable IDs, lifecycle status, branch/group/generation and source refs.
4. Preserve old entries when a talent graduates or a contract ends; change status
   rather than deleting history.
5. Add explicit provenance for any historical record not present on the live roster.
6. Keep HOLOSTARS/holoAN/office-staff exclusions intact unless the project scope is
   deliberately changed.
7. Update `HololiveCatalog` to point at the new manifest and bump the tests/counts.
8. Run `uv run pytest` and `uv run artifex characters audit --json`.

Canonical character tags are stored for ILXL/Danbooru-family prompting. Vocabulary
validation, aliases, implications and checkpoint-specific prompt correctness are
enforced by the Prompt Compiler work tracked separately in issue #36.
