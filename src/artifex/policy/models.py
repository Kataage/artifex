from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from artifex.domain import PublicationTier
from artifex.planner.models import PackFormat


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicyStage(StrEnum):
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"
    OPERATOR_REVIEW = "operator_review"


class OperatorPolicyOutcome(StrEnum):
    APPROVE = "approve"
    MEMBER_ONLY = "member_only"
    PRIVATE_REVIEW = "private_review"
    BLOCK = "block"


class PolicyReference(PolicyModel):
    policy_id: str
    version: str


class RightsPolicyProfile(PolicyModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    allow_generation: bool = True
    minimum_publication_tier: PublicationTier = PublicationTier.PUBLIC
    requires_operator_review: bool = False
    allowed_formats: tuple[PackFormat, ...] = ()
    notes: tuple[str, ...] = ()


class PolicyDecision(PolicyModel):
    stage: PolicyStage
    allowed_generation: bool
    requested_tier: PublicationTier
    policy_floor: PublicationTier
    publication_tier: PublicationTier
    requires_operator_review: bool
    reasons: tuple[str, ...] = ()
    policy_refs: tuple[PolicyReference, ...] = ()
