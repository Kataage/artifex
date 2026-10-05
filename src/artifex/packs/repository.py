from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import PackRow, SceneRow
from artifex.domain import PackState, SceneState
from artifex.packs.models import ContentPackPlan, PackRecord, ScenePlan


def _new_id() -> str:
    return uuid4().hex


class PackRepository:
    def __init__(
        self,
        database: Database,
        *,
        pack_id_factory: Callable[[], str] = _new_id,
        scene_id_factory: Callable[[], str] = _new_id,
    ) -> None:
        self._database = database
        self._pack_id_factory = pack_id_factory
        self._scene_id_factory = scene_id_factory

    def create_planned_pack(
        self,
        plan: ContentPackPlan,
        *,
        concept_id: str | None,
        planning_provenance: dict[str, object] | None = None,
    ) -> PackRecord:
        now = datetime.now(UTC)
        pack_id = self._pack_id_factory()
        scene_ids = tuple(self._scene_id_factory() for _ in plan.scenes)
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene id factory produced duplicate ids")

        with self._database.session() as session:
            if session.get(PackRow, pack_id) is not None:
                raise ValueError(f"pack already exists: {pack_id}")

            payload = {
                "plan": plan.model_dump(mode="json"),
                "planning_provenance": dict(planning_provenance or {}),
            }
            session.add(
                PackRow(
                    id=pack_id,
                    concept_id=concept_id,
                    series_id=plan.series_id,
                    state=PackState.PLANNED.value,
                    format_type=plan.format.value,
                    payload_json=payload,
                    checkpoint_json={
                        "planning_complete": True,
                        "scene_count": len(plan.scenes),
                    },
                    created_at=now,
                    updated_at=now,
                )
            )
            for scene_id, scene in zip(scene_ids, plan.scenes, strict=True):
                session.add(
                    SceneRow(
                        id=scene_id,
                        pack_id=pack_id,
                        ordinal=scene.ordinal,
                        state=SceneState.PLANNED.value,
                        publication_tier=scene.publication_tier.value,
                        payload_json={"plan": scene.model_dump(mode="json")},
                    )
                )

        return self.require(pack_id)

    def get(self, pack_id: str) -> PackRecord | None:
        with self._database.session() as session:
            pack = session.get(PackRow, pack_id)
            if pack is None:
                return None
            scenes = session.scalars(
                select(SceneRow)
                .where(SceneRow.pack_id == pack_id)
                .order_by(SceneRow.ordinal.asc(), SceneRow.id.asc())
            ).all()
            return self._record(pack, scenes)

    def require(self, pack_id: str) -> PackRecord:
        record = self.get(pack_id)
        if record is None:
            raise KeyError(f"unknown pack: {pack_id}")
        return record

    def scene_plans(self, pack_id: str) -> tuple[ScenePlan, ...]:
        record = self.require(pack_id)
        return record.plan.scenes

    @staticmethod
    def _record(pack: PackRow, scenes: list[SceneRow]) -> PackRecord:
        raw_plan = pack.payload_json.get("plan")
        if not isinstance(raw_plan, dict):
            raise TypeError(f"pack {pack.id} is missing persisted full plan")
        plan = ContentPackPlan.model_validate(raw_plan)
        if len(scenes) != len(plan.scenes):
            raise ValueError(
                f"pack {pack.id} scene persistence mismatch: "
                f"{len(scenes)} rows != {len(plan.scenes)} planned"
            )
        persisted_ordinals = [scene.ordinal for scene in scenes]
        planned_ordinals = [scene.ordinal for scene in plan.scenes]
        if persisted_ordinals != planned_ordinals:
            raise ValueError(f"pack {pack.id} scene ordinal persistence mismatch")

        return PackRecord(
            pack_id=pack.id,
            concept_id=pack.concept_id,
            series_id=pack.series_id,
            state=PackState(pack.state),
            plan=plan,
            scene_states=tuple(SceneState(scene.state) for scene in scenes),
            created_at=pack.created_at,
            updated_at=pack.updated_at,
        )
