from __future__ import annotations

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import SceneRow
from artifex.domain import PublicationTier
from artifex.packs import ScenePlan
from artifex.policy.engine import PolicyEngine
from artifex.policy.models import PolicyDecision, PolicyRequest, UseClass


class PolicyApplicationService:
    def __init__(
        self,
        database: Database,
        engine: PolicyEngine,
    ) -> None:
        self._database = database
        self._engine = engine

    def evaluate_scene(
        self,
        scene_id: str,
        *,
        use_class: UseClass,
        content_labels: tuple[str, ...] = (),
        phase: str = "preflight",
    ) -> PolicyDecision:
        with self._database.session() as session:
            row = session.get(SceneRow, scene_id)
            if row is None:
                raise KeyError(f"unknown scene: {scene_id}")
            raw_plan = row.payload_json.get("plan")
            if not isinstance(raw_plan, dict):
                raise TypeError(f"scene {scene_id} is missing persisted plan")
            plan = ScenePlan.model_validate(raw_plan)
            requested_tier = PublicationTier(row.publication_tier)

        decision = self._engine.evaluate(
            PolicyRequest(
                subject_type="scene",
                subject_id=scene_id,
                character_ids=plan.character_ids,
                use_class=use_class,
                requested_tier=requested_tier,
                content_labels=content_labels,
                phase=phase,
            )
        )

        with self._database.session() as session:
            row = session.get(SceneRow, scene_id)
            if row is None:
                raise KeyError(f"unknown scene: {scene_id}")
            row.publication_tier = decision.effective_tier.value
            payload = dict(row.payload_json)
            payload["latest_policy_decision_id"] = decision.decision_id
            payload["policy_phase"] = phase
            row.payload_json = payload

        return decision

    def scenes_for_tier(
        self,
        tier: PublicationTier,
    ) -> tuple[str, ...]:
        with self._database.session() as session:
            ids = session.scalars(
                select(SceneRow.id)
                .where(SceneRow.publication_tier == tier.value)
                .order_by(SceneRow.pack_id.asc(), SceneRow.ordinal.asc())
            ).all()
        return tuple(ids)
