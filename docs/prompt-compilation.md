# Validated ILXL / Danbooru prompt compilation

Artifex does not pass planner prose directly to the image model. The planner
produces a structured `ScenePlan`; the ILXL prompt adapter then compiles each
visual field through a versioned local tag lexicon.

## Safety rule

Unknown prose is never converted by replacing spaces with underscores.

For example, an unsupported phrase such as:

```text
floating upside down on impossible moonbeam
```

does **not** become `floating_upside_down_on_impossible_moonbeam`.

Instead, the compiler either:

- omits the unsupported concept and records it in
  `CompiledPrompt.unresolved_concepts` when
  `prompts.unresolved_policy: report`; or
- raises a prerequisite error when `prompts.unresolved_policy: error`.

This makes unresolved semantics visible in persisted Scene provenance.

## Vocabulary and aliases

The packaged profile is
`src/artifex/prompts/profiles/ilxl_danbooru_2026_10.json`.

It contains:

- canonical validated prompt tags;
- human-readable phrase aliases;
- multi-tag phrase mappings;
- semantic categories;
- implication rules;
- mutually exclusive conflict groups;
- versioned prompt profiles and category ordering.

Scene-derived tags must exist in this vocabulary. Character canonical tags and
LoRA trigger tokens are separately trusted registry data because model-specific
trigger strings are not necessarily Danbooru vocabulary.

## Ordering

The default `ilxl-danbooru-v2` profile orders prompt concepts by purpose:

1. metadata / character identity / LoRA triggers;
2. composition and camera;
3. pose and gaze;
4. expression;
5. clothing;
6. setting and time;
7. lighting, atmosphere and effects.

Conflicts are deterministic and first-wins according to structured field order.
Implications are expanded before final category ordering.

## Character and outfit rules

`CharacterProfile` and `CharacterOutfit` support:

- `required_tags`;
- `forbidden_tags`.

An outfit is selected when the Scene clothing description matches the outfit ID,
display name, alias, or canonical tag. Required tags are injected; forbidden
ones are removed from the positive prompt and added to the negative prompt.

## Checkpoint-specific profiles

The default profile can be overridden by checkpoint glob:

```yaml
prompts:
  ilxl_profile: ilxl-danbooru-v2
  unresolved_policy: report
  checkpoint_profile_overrides:
    "*my-special-ilxl*": ilxl-danbooru-character-first-v2
  extra_lexicon_paths: []
```

The selected profile, lexicon ID/version, and checkpoint are recorded in prompt
provenance. Extra local lexicon JSON files can extend or replace tags, aliases,
conflicts, and profiles without allowing the LLM to author the final prompt.

## Regression qualification

The automated regression corpus covers at least 30 representative scenes,
including single-character, duo and group compositions, common camera angles,
poses, expressions, outfits, settings, lighting and atmosphere.

The suite checks that:

- compilation is deterministic;
- generated Scene-derived tags are vocabulary-valid;
- implications and conflicts resolve consistently;
- unknown prose is reported instead of invented;
- character/outfit required and forbidden rules are enforced;
- checkpoint-specific profile selection is reproducible.
