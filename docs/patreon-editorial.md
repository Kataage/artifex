# Patreon editorial and publication classification

Artifex treats Patreon access tier and generated-content rating as independent axes.

- Publication tier: `public`, `member`, `private_review`, `blocked`
- Content rating: `general`, `suggestive`, `adult`, `explicit`

A member-only image is not automatically adult content, and adult/explicit output is
never considered public-safe merely because the creator page is age-gated.

## Editorial archetypes

The Pack planner receives one explicit archetype and must satisfy it structurally:

- `public_preview_member_continuation`: one coherent public preview followed by
  member-only continuation scenes.
- `sfw_complete_member_alternate`: a complete public-safe sequence followed by
  a member alternate/variation.
- `mini_story`: opening/development/resolution roles, with public scenes preceding
  any member continuation.
- `variation_pack`: structured visual variations rather than disconnected images.
- `seasonal_trend_pack`: a public hero/preview followed by optional member scenes.
- `public_only` and `member_only`: explicit single-tier modes.

Every `PackFormat` also has a scene-role contract. Validation happens after LLM
structured generation, so schema-valid but editorially incoherent plans are rejected
and repaired rather than reaching ComfyUI.

## Post-generation classification

Preflight policy evaluates the planned publication tier and planned content rating.
After an image passes visual-quality evaluation, the vision evaluator separately
returns:

- `content_rating`
- bounded policy-oriented `content_labels`

The evaluator only describes the visible output. It does not decide whether the image
may be published.

Artifex then runs `PolicyApplicationService` again with
`phase=post_generation`, always against the Scene's original planned publication
tier. This prevents a temporary `private_review` state from bypassing the public
rules.

The resulting state is:

- allow: the selected attempt may proceed.
- review: the selected attempt moves to operator review.
- block: publication is hard-blocked. The operator may regenerate/retry or reject;
  a hard block cannot be approved away.

## Versioned Patreon policy

The packaged profile `patreon_2026_10@2026-10-06` is a conservative automation
guardrail derived from Patreon official documentation checked on 2026-10-06:

- https://www.patreon.com/policy/guidelines
- https://www.patreon.com/policy/benefits
- https://support.patreon.com/hc/en-gb/articles/36188028163725-Understanding-Public-Previews-and-Commerce-guidelines-for-adult-18-creators
- https://support.patreon.com/hc/en-us/articles/360004061452-Flexible-previews-Draggable-paywall

The profile is not a legal determination. It is deliberately versioned so official
rule changes can be reviewed and tested without silently changing historical
production decisions.

Current automation policy includes:

- public general: allow
- public suggestive: manual review
- public adult/explicit: block
- member general/suggestive/adult: allow, subject to content-label rules
- member explicit: manual review
- prohibited/high-risk labels: block or review according to the profile
- commercial/shop use: review, because Patreon Commerce rules are an additional
  policy surface

Rights/character policy profiles and the Patreon platform profile are composed; the
most restrictive applicable outcome wins.

## Patreon post package

When Patreon packaging is enabled, final archive creates:

`metadata/patreon-post-package.json`

It contains:

- post title and caption/logline
- public preview copy
- tags
- format and editorial archetype
- public/member/review Scene IDs
- per-Scene role, tier, planned rating, observed rating and labels
- selected archived media paths
- a `publication_ready` flag

Actual external posting remains intentionally separate. The package is a stable
handoff artifact for a later publishing connector or operator workflow.
