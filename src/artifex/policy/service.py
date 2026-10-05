from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from artifex.domain import PublicationTier, ResultState
from artifex.planner.models import PackFormat
from artifex.policy.engine import PolicyEngine
from artifex.policy.models import OperatorPolicyOutcome, PolicyDecision
from artifex.policy.repository import PolicyDecisionRepository


class PersistedPolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    decision: PolicyDecision


class PolicyGateService:
    def __init__(
        self,
        engine: PolicyEngine,
        repository: PolicyDecisionRepository,
    ) -> None:
        self._engine = engine
        self._repository = repository

    def pre_generation(
        self,
        *,
        subject_type: str,
        subject_id: str,
        character_ids: Sequence[str],
        pack_format: PackFormat,
        requested_tier: PublicationTier,
    ) -> PersistedPolicyDecision:
        decision = self._engine.pre_generation(
            character_ids=character_ids,
            pack_format=pack_format,
            requested_tier=requested_tier,
        )
        row = self._repository.record(
            subject_type=subject_type,
            subject_id=subject_id,
            decision=decision,
        )
        return PersistedPolicyDecision(id=row.id, decision=decision)

    def post_generation(
        self,
        previous: PersistedPolicyDecision,
        *,
        subject_type: str,
        subject_id: str,
        evaluation_state: ResultState,
    ) -> PersistedPolicyDecision:
        decision = self._engine.post_generation(
            previous.decision,
            evaluation_state=evaluation_state,
        )
        row = self._repository.record(
            subject_type=subject_type,
            subject_id=subject_id,
            decision=decision,
            parent_decision_id=previous.id,
        )
        return PersistedPolicyDecision(id=row.id, decision=decision)

    def operator_outcome(
        self,
        previous: PersistedPolicyDecision,
        *,
        subject_type: str,
        subject_id: str,
        outcome: OperatorPolicyOutcome,
    ) -> PersistedPolicyDecision:
        decision = self._engine.apply_operator_outcome(previous.decision, outcome)
        row = self._repository.record(
            subject_type=subject_type,
            subject_id=subject_id,
            decision=decision,
            parent_decision_id=previous.id,
            operator_outcome=outcome.value,
        )
        return PersistedPolicyDecision(id=row.id, decision=decision)
