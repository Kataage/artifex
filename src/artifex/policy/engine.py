from __future__ import annotations

from datetime import UTC, datetime

from artifex.characters import CharacterRegistry
from artifex.config.models import RightsConfig
from artifex.domain import PublicationTier
from artifex.policy.models import (
    PolicyDecision,
    PolicyOutcome,
    PolicyProfile,
    PolicyRequest,
)
from artifex.policy.registry import PolicyRegistry
from artifex.policy.repository import PolicyDecisionRepository

_OUTCOME_RANK = {
    PolicyOutcome.ALLOW: 0,
    PolicyOutcome.REVIEW: 1,
    PolicyOutcome.BLOCK: 2,
}


class PolicyEngine:
    def __init__(
        self,
        registry: PolicyRegistry,
        characters: CharacterRegistry,
        repository: PolicyDecisionRepository,
        config: RightsConfig,
    ) -> None:
        self._registry = registry
        self._characters = characters
        self._repository = repository
        self._config = config

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        now = datetime.now(UTC)
        if not self._config.enforce:
            return self._repository.persist(
                PolicyDecision(
                    decision_id=self._repository.new_id(),
                    subject_type=request.subject_type,
                    subject_id=request.subject_id,
                    outcome=PolicyOutcome.ALLOW,
                    requested_tier=request.requested_tier,
                    effective_tier=request.requested_tier,
                    profile_versions=("enforcement-disabled",),
                    reasons=("rights policy enforcement disabled by configuration",),
                    content_labels=request.content_labels,
                    created_at=now,
                    content_rating=request.content_rating,
                )
            )

        profiles, load_reasons = self._profiles_for(request.character_ids)
        if not profiles:
            return self._repository.persist(
                PolicyDecision(
                    decision_id=self._repository.new_id(),
                    subject_type=request.subject_type,
                    subject_id=request.subject_id,
                    outcome=PolicyOutcome.BLOCK,
                    requested_tier=request.requested_tier,
                    effective_tier=PublicationTier.BLOCKED,
                    profile_versions=(),
                    reasons=tuple(load_reasons) or ("no policy profile resolved",),
                    content_labels=request.content_labels,
                    created_at=now,
                    content_rating=request.content_rating,
                )
            )

        outcomes: list[PolicyOutcome] = []
        if load_reasons:
            outcomes.append(PolicyOutcome.BLOCK)
        reasons = list(load_reasons)
        forced_private_review = False
        profile_versions: list[str] = []

        normalized_labels = tuple(
            dict.fromkeys(
                label.strip().casefold()
                for label in request.content_labels
                if label.strip()
            )
        )

        for profile in profiles:
            profile_versions.append(f"{profile.id}@{profile.version}")

            use_outcome = profile.use_rules[request.use_class]
            outcomes.append(use_outcome)
            if use_outcome is not PolicyOutcome.ALLOW:
                reasons.append(
                    f"{profile.id}@{profile.version}: use "
                    f"{request.use_class.value} -> {use_outcome.value}"
                )

            tier_outcome = profile.tier_rules[request.requested_tier]
            outcomes.append(tier_outcome)
            if tier_outcome is not PolicyOutcome.ALLOW:
                reasons.append(
                    f"{profile.id}@{profile.version}: tier "
                    f"{request.requested_tier.value} -> {tier_outcome.value}"
                )

            rating_rule = profile.rating_rules.get(
                request.requested_tier,
                {},
            ).get(request.content_rating)
            if rating_rule is not None:
                outcomes.append(rating_rule.outcome)
                if rating_rule.reason:
                    reasons.append(
                        f"{profile.id}@{profile.version}: rating "
                        f"{request.content_rating.value}: {rating_rule.reason}"
                    )
                else:
                    reasons.append(
                        f"{profile.id}@{profile.version}: rating "
                        f"{request.content_rating.value} -> "
                        f"{rating_rule.outcome.value}"
                    )
                if rating_rule.force_tier is PublicationTier.PRIVATE_REVIEW:
                    forced_private_review = True
                if rating_rule.force_tier is PublicationTier.BLOCKED:
                    outcomes.append(PolicyOutcome.BLOCK)

            label_rules = {
                key.strip().casefold(): value
                for key, value in profile.content_label_rules.items()
            }
            for label in normalized_labels:
                rule = label_rules.get(label)
                if rule is None:
                    continue
                outcomes.append(rule.outcome)
                if rule.reason:
                    reasons.append(
                        f"{profile.id}@{profile.version}: {label}: {rule.reason}"
                    )
                else:
                    reasons.append(
                        f"{profile.id}@{profile.version}: label {label} "
                        f"-> {rule.outcome.value}"
                    )
                if rule.force_tier is PublicationTier.PRIVATE_REVIEW:
                    forced_private_review = True
                if rule.force_tier is PublicationTier.BLOCKED:
                    outcomes.append(PolicyOutcome.BLOCK)

        outcome = max(outcomes, key=_OUTCOME_RANK.__getitem__, default=PolicyOutcome.ALLOW)
        if outcome is PolicyOutcome.BLOCK:
            effective_tier = PublicationTier.BLOCKED
        elif outcome is PolicyOutcome.REVIEW or forced_private_review:
            effective_tier = PublicationTier.PRIVATE_REVIEW
            if outcome is PolicyOutcome.ALLOW:
                outcome = PolicyOutcome.REVIEW
        else:
            effective_tier = request.requested_tier

        return self._repository.persist(
            PolicyDecision(
                decision_id=self._repository.new_id(),
                subject_type=request.subject_type,
                subject_id=request.subject_id,
                outcome=outcome,
                requested_tier=request.requested_tier,
                effective_tier=effective_tier,
                profile_versions=tuple(sorted(set(profile_versions))),
                reasons=tuple(reasons),
                content_labels=normalized_labels,
                created_at=now,
                content_rating=request.content_rating,
            )
        )

    def _profiles_for(
        self,
        character_ids: tuple[str, ...],
    ) -> tuple[tuple[PolicyProfile, ...], tuple[str, ...]]:
        profiles: list[PolicyProfile] = []
        reasons: list[str] = []

        for character_id in character_ids:
            character = self._characters.get(character_id)
            if character is None:
                reasons.append(f"unknown character: {character_id}")
                continue

            profile_id = character.policy_profile or self._config.default_profile
            profile = self._registry.get(profile_id)
            if profile is None:
                reasons.append(
                    f"policy profile unavailable for {character_id}: {profile_id}"
                )
                continue
            profiles.append(profile)

        if self._config.platform_profile is not None:
            platform = self._registry.get(self._config.platform_profile)
            if platform is None:
                reasons.append(
                    "platform policy profile unavailable: "
                    + self._config.platform_profile
                )
            else:
                profiles.append(platform)

        return tuple(profiles), tuple(reasons)
