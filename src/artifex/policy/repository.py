from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from artifex.db import Database
from artifex.db.models import PolicyDecisionRow
from artifex.domain import PublicationTier
from artifex.policy.models import (
    OperatorReviewOutcome,
    PolicyDecision,
    PolicyOutcome,
)


def _new_id() -> str:
    return uuid4().hex


class PolicyDecisionRepository:
    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def new_id(self) -> str:
        return self._id_factory()

    def persist(self, decision: PolicyDecision) -> PolicyDecision:
        with self._database.session() as session:
            if session.get(PolicyDecisionRow, decision.decision_id) is not None:
                raise ValueError(
                    f"policy decision already exists: {decision.decision_id}"
                )
            profile_names = decision.profile_versions or ("none",)
            session.add(
                PolicyDecisionRow(
                    id=decision.decision_id,
                    subject_type=decision.subject_type,
                    subject_id=decision.subject_id,
                    policy_name=";".join(profile_names),
                    policy_version="composed",
                    decision=decision.outcome.value,
                    reason="; ".join(decision.reasons) or None,
                    payload_json=decision.model_dump(mode="json"),
                    created_at=decision.created_at,
                )
            )
        return decision

    def get(self, decision_id: str) -> PolicyDecision | None:
        with self._database.session() as session:
            row = session.get(PolicyDecisionRow, decision_id)
            if row is None:
                return None
            return PolicyDecision.model_validate(row.payload_json)

    def require(self, decision_id: str) -> PolicyDecision:
        decision = self.get(decision_id)
        if decision is None:
            raise KeyError(f"unknown policy decision: {decision_id}")
        return decision

    def resolve_review(
        self,
        decision_id: str,
        *,
        approved: bool,
    ) -> PolicyDecision:
        original = self.require(decision_id)
        if original.outcome is not PolicyOutcome.REVIEW:
            raise ValueError("only review decisions can receive operator resolution")

        outcome = PolicyOutcome.ALLOW if approved else PolicyOutcome.BLOCK
        effective_tier = (
            original.requested_tier
            if approved
            else PublicationTier.BLOCKED
        )
        review = (
            OperatorReviewOutcome.APPROVED
            if approved
            else OperatorReviewOutcome.REJECTED
        )
        now = datetime.now(UTC)
        resolved = PolicyDecision(
            decision_id=self.new_id(),
            subject_type=original.subject_type,
            subject_id=original.subject_id,
            outcome=outcome,
            requested_tier=original.requested_tier,
            effective_tier=effective_tier,
            profile_versions=original.profile_versions,
            reasons=(
                *original.reasons,
                f"operator review: {review.value}",
            ),
            content_labels=original.content_labels,
            created_at=now,
            review_of=original.decision_id,
            operator_review=review,
        )
        return self.persist(resolved)
