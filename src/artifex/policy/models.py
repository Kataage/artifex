from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artifex.domain import PublicationTier


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicyOutcome(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class UseClass(StrEnum):
    PRIVATE = "private"
    PUBLIC_FREE = "public_free"
    PAID_MEMBERSHIP = "paid_membership"
    COMMERCIAL = "commercial"


class PolicyRule(PolicyModel):
    outcome: PolicyOutcome
    force_tier: PublicationTier | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_force_tier(self) -> PolicyRule:
        if self.force_tier not in {
            None,
            PublicationTier.PRIVATE_REVIEW,
            PublicationTier.BLOCKED,
        }:
            raise ValueError(
                "policy rules may only force private_review or blocked tiers"
            )
        if (
            self.outcome is PolicyOutcome.BLOCK
            and self.force_tier is PublicationTier.PRIVATE_REVIEW
        ):
            raise ValueError("blocked policy rule cannot force private_review")
        return self


class PolicyProfile(PolicyModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    use_rules: dict[UseClass, PolicyOutcome]
    tier_rules: dict[PublicationTier, PolicyOutcome]
    content_label_rules: dict[str, PolicyRule] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_required_rules(self) -> PolicyProfile:
        missing_use = set(UseClass) - set(self.use_rules)
        if missing_use:
            raise ValueError(
                "policy profile missing use rules: "
                + ", ".join(sorted(value.value for value in missing_use))
            )
        missing_tiers = set(PublicationTier) - set(self.tier_rules)
        if missing_tiers:
            raise ValueError(
                "policy profile missing tier rules: "
                + ", ".join(sorted(value.value for value in missing_tiers))
            )
        normalized = [label.strip().casefold() for label in self.content_label_rules]
        if len(normalized) != len(set(normalized)):
            raise ValueError("content label rules must be unique case-insensitively")
        return self


class PolicyRequest(PolicyModel):
    subject_type: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    character_ids: tuple[str, ...] = Field(min_length=1)
    use_class: UseClass
    requested_tier: PublicationTier
    content_labels: tuple[str, ...] = ()
    phase: str = Field(default="preflight", pattern="^(preflight|post_generation)$")


class PolicyDecision(PolicyModel):
    decision_id: str
    subject_type: str
    subject_id: str
    outcome: PolicyOutcome
    requested_tier: PublicationTier
    effective_tier: PublicationTier
    profile_versions: tuple[str, ...]
    reasons: tuple[str, ...]
    content_labels: tuple[str, ...]
    created_at: datetime
    review_of: str | None = None
    operator_review: OperatorReviewOutcome | None = None


class OperatorReviewOutcome(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
