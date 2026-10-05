from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from artifex.db import Database
from artifex.db.models import PolicyDecisionRow
from artifex.policy.models import PolicyDecision


def _new_id() -> str:
    return uuid4().hex


class PolicyDecisionRepository:
    engine_name = "artifex_rights_gate"
    engine_version = "1"

    def __init__(
        self,
        database: Database,
        *,
        id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._id_factory = id_factory

    def record(
        self,
        *,
        subject_type: str,
        subject_id: str,
        decision: PolicyDecision,
        parent_decision_id: str | None = None,
        operator_outcome: str | None = None,
    ) -> PolicyDecisionRow:
        now = datetime.now(UTC)
        row_id = self._id_factory()
        payload = {
            "decision": decision.model_dump(mode="json"),
            "parent_decision_id": parent_decision_id,
            "operator_outcome": operator_outcome,
        }
        with self._database.session() as session:
            row = PolicyDecisionRow(
                id=row_id,
                subject_type=subject_type,
                subject_id=subject_id,
                policy_name=self.engine_name,
                policy_version=self.engine_version,
                decision=decision.publication_tier.value,
                reason="; ".join(decision.reasons) if decision.reasons else None,
                payload_json=payload,
                created_at=now,
            )
            session.add(row)
        return self.require(row_id)

    def require(self, decision_id: str) -> PolicyDecisionRow:
        with self._database.session() as session:
            row = session.get(PolicyDecisionRow, decision_id)
            if row is None:
                raise KeyError(f"unknown policy decision: {decision_id}")
            session.expunge(row)
            return row
