from __future__ import annotations

from collections.abc import Sequence

from artifex.characters import CharacterRegistry
from artifex.config.models import RightsConfig
from artifex.domain import PublicationTier, ResultState
from artifex.planner.models import PackFormat
from artifex.policy.models import (
    OperatorPolicyOutcome,
    PolicyDecision,
    PolicyReference,
    PolicyStage,
)
from artifex.policy.registry import RightsPolicyRegistry

_TIER_RANK = {
    PublicationTier.PUBLIC: 0,
    PublicationTier.MEMBER: 1,
    PublicationTier.PRIVATE_REVIEW: 2,
    PublicationTier.BLOCKED: 3,
}


def _more_restrictive(
    left: PublicationTier,
    right: PublicationTier,
) -> PublicationTier:
    return left if _TIER_RANK[left] >= _TIER_RANK[right] else right


class PolicyEngine:
    def __init__(
        self,
        characters: CharacterRegistry,
        policies: RightsPolicyRegistry,
        config: RightsConfig,
    ) -> None:
        self._characters = characters
        self._policies = policies
        self._config = config

    def pre_generation(
        self,
        *,
        character_ids: Sequence[str],
        pack_format: PackFormat,
        requested_tier: PublicationTier,
    ) -> PolicyDecision:
        if not self._config.enforce:
            return PolicyDecision(
                stage=PolicyStage.PRE_GENERATION,
                allowed_generation=True,
                requested_tier=requested_tier,
                policy_floor=PublicationTier.PUBLIC,
                publication_tier=requested_tier,
                requires_operator_review=False,
                reasons=("rights_enforcement_disabled",),
            )

        reasons: list[str] = []
        references: list[PolicyReference] = []
        allowed_generation = True
        policy_floor = PublicationTier.PUBLIC
        requires_review = False

        for character_id in dict.fromkeys(character_ids):
            character = self._characters.get(character_id)
            if character is None:
                allowed_generation = False
                reasons.append(f"unknown_character:{character_id}")
                continue

            policy_id = character.policy_profile or self._config.default_policy_id
            policy = self._policies.get(policy_id)
            if policy is None and not self._config.fail_closed_unknown_policy:
                policy_id = self._config.default_policy_id
                policy = self._policies.get(policy_id)

            if policy is None:
                allowed_generation = False
                reasons.append(f"unknown_rights_policy:{character_id}:{policy_id}")
                continue

            reference = PolicyReference(policy_id=policy.id, version=policy.version)
            if reference not in references:
                references.append(reference)

            policy_floor = _more_restrictive(
                policy_floor,
                policy.minimum_publication_tier,
            )
            if not policy.allow_generation:
                allowed_generation = False
                reasons.append(f"generation_disallowed:{character_id}:{policy.id}")
            if policy.allowed_formats and pack_format not in policy.allowed_formats:
                allowed_generation = False
                reasons.append(
                    f"format_disallowed:{character_id}:{policy.id}:{pack_format.value}"
                )
            if policy.requires_operator_review:
                requires_review = True
                reasons.append(f"operator_review_required:{character_id}:{policy.id}")

        tier = _more_restrictive(requested_tier, policy_floor)
        if not allowed_generation:
            tier = PublicationTier.BLOCKED
        elif requires_review:
            tier = _more_restrictive(tier, PublicationTier.PRIVATE_REVIEW)

        return PolicyDecision(
            stage=PolicyStage.PRE_GENERATION,
            allowed_generation=allowed_generation,
            requested_tier=requested_tier,
            policy_floor=policy_floor,
            publication_tier=tier,
            requires_operator_review=requires_review,
            reasons=tuple(reasons),
            policy_refs=tuple(references),
        )

    @staticmethod
    def post_generation(
        previous: PolicyDecision,
        *,
        evaluation_state: ResultState,
    ) -> PolicyDecision:
        if previous.publication_tier is PublicationTier.BLOCKED:
            return previous.model_copy(update={"stage": PolicyStage.POST_GENERATION})

        tier = previous.publication_tier
        requires_review = previous.requires_operator_review
        reasons = list(previous.reasons)

        if evaluation_state is ResultState.REJECTED:
            tier = PublicationTier.BLOCKED
            reasons.append("post_generation_rejected")
        elif evaluation_state is ResultState.REVIEW:
            tier = _more_restrictive(tier, PublicationTier.PRIVATE_REVIEW)
            requires_review = True
            reasons.append("post_generation_review_required")
        else:
            reasons.append("post_generation_accepted")

        return PolicyDecision(
            stage=PolicyStage.POST_GENERATION,
            allowed_generation=previous.allowed_generation,
            requested_tier=previous.requested_tier,
            policy_floor=previous.policy_floor,
            publication_tier=tier,
            requires_operator_review=requires_review,
            reasons=tuple(dict.fromkeys(reasons)),
            policy_refs=previous.policy_refs,
        )

    @staticmethod
    def apply_operator_outcome(
        previous: PolicyDecision,
        outcome: OperatorPolicyOutcome,
    ) -> PolicyDecision:
        reasons = list(previous.reasons)
        reasons.append(f"operator_outcome:{outcome.value}")

        if (
            not previous.allowed_generation
            or previous.publication_tier is PublicationTier.BLOCKED
        ):
            tier = PublicationTier.BLOCKED
            requires_review = False
        elif outcome is OperatorPolicyOutcome.BLOCK:
            tier = PublicationTier.BLOCKED
            requires_review = False
        elif outcome is OperatorPolicyOutcome.PRIVATE_REVIEW:
            tier = _more_restrictive(
                previous.policy_floor,
                PublicationTier.PRIVATE_REVIEW,
            )
            requires_review = False
        elif outcome is OperatorPolicyOutcome.MEMBER_ONLY:
            base = _more_restrictive(
                previous.requested_tier,
                previous.policy_floor,
            )
            tier = _more_restrictive(base, PublicationTier.MEMBER)
            requires_review = False
        else:
            tier = _more_restrictive(
                previous.requested_tier,
                previous.policy_floor,
            )
            requires_review = False

        return PolicyDecision(
            stage=PolicyStage.OPERATOR_REVIEW,
            allowed_generation=previous.allowed_generation,
            requested_tier=previous.requested_tier,
            policy_floor=previous.policy_floor,
            publication_tier=tier,
            requires_operator_review=requires_review,
            reasons=tuple(dict.fromkeys(reasons)),
            policy_refs=previous.policy_refs,
        )
