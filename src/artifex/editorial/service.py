from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import select

from artifex.config.models import EditorialConfig
from artifex.db import Database
from artifex.db.models import CharacterRow, ConceptRow, PackRow
from artifex.domain import PackState
from artifex.editorial.models import (
    EditorialLane,
    EditorialPlanDecision,
    InventoryCounts,
    SeriesPlanKind,
)
from artifex.editorial.repository import EditorialRepository, PackInventoryRepository
from artifex.planner.models import CharacterOption, PlanningContext
from artifex.series import SeriesProfile, SeriesRepository, SeriesStatus


class EditorialService:
    """Persisted policy layer above the generic runtime work queue."""

    def __init__(
        self,
        database: Database,
        config: EditorialConfig,
        series: SeriesRepository,
        inventory: PackInventoryRepository,
        decisions: EditorialRepository,
    ) -> None:
        self._database = database
        self._config = config
        self._series = series
        self._inventory = inventory
        self._decisions = decisions

    def inventory_counts(self) -> InventoryCounts:
        self._inventory.sync_finalized()
        self._inventory.expire_due()
        return self._inventory.counts()

    def production_allowed(self) -> tuple[bool, str]:
        counts = self.inventory_counts()
        if self._config.mode == "continuous":
            return True, "continuous editorial mode"

        active = self._decisions.refill_active()
        if active is None:
            active = counts.available <= self._config.inventory_low_watermark
            self._decisions.set_refill_active(active)

        if active and counts.available >= self._config.inventory_high_watermark:
            active = False
            self._decisions.set_refill_active(False)
        elif (
            not active
            and counts.available <= self._config.inventory_low_watermark
        ):
            active = True
            self._decisions.set_refill_active(True)

        if active:
            return (
                True,
                (
                    "completed inventory is in refill window "
                    f"({counts.available}/{self._config.inventory_high_watermark})"
                ),
            )
        return (
            False,
            (
                "completed inventory above low-water mark "
                f"({counts.available}; resume <= "
                f"{self._config.inventory_low_watermark})"
            ),
        )

    def decide_plan(self, *, has_ideas: bool) -> EditorialPlanDecision:
        pending = self._valid_pending()
        if pending is not None:
            return pending

        recent = self._recent_packs(
            max(
                self._config.cadence_window_packs,
                self._config.diversity_window_packs,
                1,
            )
        )
        series = self._eligible_series(recent)
        lane, lane_reason = self._choose_lane(recent, bool(series))

        if lane is EditorialLane.SERIES and series:
            selected = series[0]
            kind = self._series_kind(selected)
            reason = (
                f"{lane_reason}; selected series={selected.id} "
                f"priority={selected.editorial_policy.priority} "
                f"episode={selected.current_episode + 1}"
            )
            return self._decisions.create(
                lane=EditorialLane.SERIES,
                reason=reason,
                series_id=selected.id,
                concept_id=None,
                series_plan_kind=kind,
                payload={
                    "recent_pack_ids": [row.id for row in recent],
                    "series_current_episode": selected.current_episode,
                    "series_target_share": self._config.series_target_share,
                },
            )

        if not has_ideas:
            return EditorialPlanDecision(
                decision_id=None,
                lane=EditorialLane.STANDALONE,
                reason=f"{lane_reason}; standalone idea inventory is empty",
            )

        concept_id = self._select_standalone_concept(recent)
        if concept_id is None:
            return EditorialPlanDecision(
                decision_id=None,
                lane=EditorialLane.STANDALONE,
                reason=f"{lane_reason}; no valid standalone idea available",
            )
        return self._decisions.create(
            lane=EditorialLane.STANDALONE,
            reason=f"{lane_reason}; selected diversity-ranked idea={concept_id}",
            series_id=None,
            concept_id=concept_id,
            series_plan_kind=None,
            payload={
                "recent_pack_ids": [row.id for row in recent],
                "series_target_share": self._config.series_target_share,
            },
        )

    def apply_diversity(self, context: PlanningContext) -> PlanningContext:
        recent = self._recent_packs(self._config.diversity_window_packs)
        if not recent:
            return context

        character_counts: Counter[str] = Counter()
        branch_counts: Counter[str] = Counter()
        group_counts: Counter[str] = Counter()
        option_by_id = {item.id: item for item in context.characters}
        cooldown_ids: set[str] = set()

        for index, row in enumerate(recent):
            ids = self._pack_character_ids(row)
            character_counts.update(ids)
            if index < self._config.character_cooldown_packs:
                cooldown_ids.update(ids)
            for character_id in ids:
                option = option_by_id.get(character_id)
                if option is None:
                    continue
                branch_counts[self._branch_key(option)] += 1
                group_counts[self._group_key(option)] += 1

        appearances = max(1, sum(character_counts.values()))
        pack_count = max(1, len(recent))
        weight_total = (
            self._config.character_diversity_weight
            + self._config.branch_diversity_weight
            + self._config.group_diversity_weight
        )

        adjusted: list[CharacterOption] = []
        for option in context.characters:
            character_component = character_counts[option.id] / pack_count
            branch_component = branch_counts[self._branch_key(option)] / appearances
            group_component = group_counts[self._group_key(option)] / appearances
            editorial_penalty = min(
                1.0,
                (
                    character_component
                    * self._config.character_diversity_weight
                    + branch_component
                    * self._config.branch_diversity_weight
                    + group_component
                    * self._config.group_diversity_weight
                )
                / weight_total,
            )
            if option.id in cooldown_ids:
                editorial_penalty = max(editorial_penalty, 0.95)
            adjusted.append(
                option.model_copy(
                    update={
                        "recent_use_penalty": max(
                            option.recent_use_penalty,
                            editorial_penalty,
                        )
                    }
                )
            )

        return context.model_copy(update={"characters": tuple(adjusted)})

    def mark_finalized_pack(
        self,
        pack_id: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._inventory.mark_available(pack_id, metadata=metadata)

    def reserve_pack(self, pack_id: str) -> None:
        self._inventory.reserve(pack_id)

    def release_pack(self, pack_id: str) -> None:
        self._inventory.release(pack_id)

    def consume_pack(self, pack_id: str) -> None:
        self._inventory.consume(pack_id)

    def expire_pack(self, pack_id: str) -> None:
        self._inventory.expire(pack_id)

    def complete_decision(
        self,
        decision_id: str | None,
        *,
        pack_id: str,
        concept_id: str | None,
    ) -> None:
        if decision_id is None:
            return
        self._decisions.resolve(
            decision_id,
            status="completed",
            pack_id=pack_id,
            concept_id=concept_id,
        )

    def fail_decision(self, decision_id: str | None, *, error: str) -> None:
        if decision_id is None:
            return
        self._decisions.resolve(
            decision_id,
            status="failed",
            payload_patch={"error": error},
        )

    def _valid_pending(self) -> EditorialPlanDecision | None:
        pending = self._decisions.pending()
        if pending is None:
            return None
        if pending.lane is EditorialLane.SERIES:
            if pending.series_id is None:
                self._decisions.resolve(
                    pending.decision_id or "",
                    status="cancelled",
                    payload_patch={"cancel_reason": "missing series id"},
                )
                return None
            series = self._series.get(pending.series_id)
            if (
                series is None
                or series.status is not SeriesStatus.ACTIVE
                or not series.editorial_policy.auto_continue
            ):
                self._decisions.resolve(
                    pending.decision_id or "",
                    status="cancelled",
                    payload_patch={"cancel_reason": "series no longer eligible"},
                )
                return None
            return pending

        if pending.concept_id is None:
            self._decisions.resolve(
                pending.decision_id or "",
                status="cancelled",
                payload_patch={"cancel_reason": "missing concept id"},
            )
            return None
        with self._database.session() as session:
            concept = session.get(ConceptRow, pending.concept_id)
            valid = concept is not None and concept.status == "idea"
        if not valid:
            self._decisions.resolve(
                pending.decision_id or "",
                status="cancelled",
                payload_patch={"cancel_reason": "concept no longer available"},
            )
            return None
        return pending

    def _eligible_series(self, recent: tuple[PackRow, ...]) -> tuple[SeriesProfile, ...]:
        eligible: list[SeriesProfile] = []
        for item in self._series.list(status=SeriesStatus.ACTIVE):
            policy = item.editorial_policy
            if not policy.auto_continue:
                continue
            if self._series_has_open_pack(item.id):
                continue
            if (
                policy.max_episodes is not None
                and item.current_episode >= policy.max_episodes
            ):
                self._series.set_status(item.id, SeriesStatus.COMPLETED)
                continue
            gap = self._series_gap(item.id, recent)
            if gap < policy.min_gap_packs:
                continue
            eligible.append(item)

        eligible.sort(
            key=lambda item: (
                -item.editorial_policy.priority,
                _utc(item.updated_at),
                item.current_episode,
                item.id,
            )
        )
        return tuple(eligible)

    def _series_has_open_pack(self, series_id: str) -> bool:
        terminal = {
            PackState.FINALIZED.value,
            PackState.FAILED.value,
            PackState.BLOCKED.value,
        }
        with self._database.session() as session:
            pack_id = session.scalar(
                select(PackRow.id)
                .where(
                    PackRow.series_id == series_id,
                    PackRow.state.not_in(terminal),
                )
                .limit(1)
            )
        return pack_id is not None

    def _choose_lane(
        self,
        recent: tuple[PackRow, ...],
        has_series: bool,
    ) -> tuple[EditorialLane, str]:
        if not has_series:
            return EditorialLane.STANDALONE, "no eligible active Series"
        if not recent:
            return EditorialLane.SERIES, "active Series exists and no cadence history exists"

        lanes = [
            EditorialLane.SERIES if row.series_id else EditorialLane.STANDALONE
            for row in recent[: self._config.cadence_window_packs]
        ]
        newest = lanes[0]
        consecutive = 0
        for lane in lanes:
            if lane is not newest:
                break
            consecutive += 1

        if (
            newest is EditorialLane.SERIES
            and consecutive >= self._config.max_consecutive_series
        ):
            return EditorialLane.STANDALONE, "maximum consecutive Series Packs reached"
        if (
            newest is EditorialLane.STANDALONE
            and consecutive >= self._config.max_consecutive_standalone
        ):
            return EditorialLane.SERIES, "maximum consecutive standalone Packs reached"

        series_ratio = (
            sum(1 for lane in lanes if lane is EditorialLane.SERIES)
            / len(lanes)
        )
        if series_ratio < self._config.series_target_share:
            return (
                EditorialLane.SERIES,
                (
                    f"Series share {series_ratio:.2f} below target "
                    f"{self._config.series_target_share:.2f}"
                ),
            )
        return (
            EditorialLane.STANDALONE,
            (
                f"Series share {series_ratio:.2f} meets target "
                f"{self._config.series_target_share:.2f}"
            ),
        )

    def _select_standalone_concept(
        self,
        recent: tuple[PackRow, ...],
    ) -> str | None:
        taxonomy = self._character_taxonomy()
        recent_character_counts: Counter[str] = Counter()
        recent_branch_counts: Counter[str] = Counter()
        recent_group_counts: Counter[str] = Counter()
        recent_format_counts: Counter[str] = Counter()
        recent_theme_counts: Counter[str] = Counter()
        cooldown_ids: set[str] = set()

        for index, row in enumerate(recent[: self._config.diversity_window_packs]):
            ids = self._pack_character_ids(row)
            recent_character_counts.update(ids)
            if index < self._config.character_cooldown_packs:
                cooldown_ids.update(ids)
            for character_id in ids:
                branch, group = taxonomy.get(
                    character_id,
                    ("__unknown_branch__", "__unknown_group__"),
                )
                recent_branch_counts[branch] += 1
                recent_group_counts[group] += 1
            plan = row.payload_json.get("plan")
            if isinstance(plan, dict):
                raw_format = plan.get("format")
                if isinstance(raw_format, str):
                    recent_format_counts[raw_format] += 1
            concept = row.payload_json.get("planning_provenance")
            if isinstance(concept, dict):
                raw_theme = concept.get("theme")
                if isinstance(raw_theme, str) and raw_theme.strip():
                    recent_theme_counts[raw_theme.strip().casefold()] += 1

        with self._database.session() as session:
            concept_rows = session.scalars(
                select(ConceptRow)
                .where(ConceptRow.status == "idea")
                .order_by(ConceptRow.created_at.asc(), ConceptRow.id.asc())
            ).all()

        ranked: list[tuple[int, float, float, datetime, str]] = []
        for concept_row in concept_rows:
            candidate = concept_row.payload_json.get("candidate")
            if not isinstance(candidate, dict):
                continue
            raw_ids = candidate.get("character_ids", ())
            if not isinstance(raw_ids, list | tuple):
                continue
            ids = tuple(str(value) for value in raw_ids)
            if not ids:
                continue

            char_penalty = max(
                (recent_character_counts[character_id] for character_id in ids),
                default=0,
            )
            branches = [
                taxonomy.get(
                    character_id,
                    ("__unknown_branch__", "__unknown_group__"),
                )[0]
                for character_id in ids
            ]
            groups = [
                taxonomy.get(
                    character_id,
                    ("__unknown_branch__", "__unknown_group__"),
                )[1]
                for character_id in ids
            ]
            branch_penalty = max(
                (recent_branch_counts[value] for value in branches),
                default=0,
            )
            group_penalty = max(
                (recent_group_counts[value] for value in groups),
                default=0,
            )
            raw_format = candidate.get("format")
            format_penalty = (
                recent_format_counts[str(raw_format)]
                if isinstance(raw_format, str)
                else 0
            )
            raw_theme = candidate.get("theme")
            theme_penalty = (
                recent_theme_counts[raw_theme.strip().casefold()]
                if isinstance(raw_theme, str)
                else 0
            )
            diversity_penalty = (
                char_penalty * self._config.character_diversity_weight
                + branch_penalty * self._config.branch_diversity_weight
                + group_penalty * self._config.group_diversity_weight
                + format_penalty * self._config.format_diversity_weight
                + theme_penalty * self._config.theme_diversity_weight
            )
            cooldown = int(any(character_id in cooldown_ids for character_id in ids))
            ranked.append(
                (
                    cooldown,
                    float(diversity_penalty),
                    -float(concept_row.score or 0.0),
                    _utc(concept_row.created_at),
                    concept_row.id,
                )
            )
        return min(ranked)[4] if ranked else None

    def _character_taxonomy(self) -> dict[str, tuple[str, str]]:
        with self._database.session() as session:
            rows = session.scalars(select(CharacterRow)).all()
        result: dict[str, tuple[str, str]] = {}
        for row in rows:
            profile = dict(row.profile_json)
            branch = profile.get("branch")
            group = profile.get("group")
            result[row.id] = (
                str(branch) if branch else row.namespace or "__unknown_branch__",
                str(group) if group else row.namespace or "__unknown_group__",
            )
        return result

    def _recent_packs(self, limit: int) -> tuple[PackRow, ...]:
        if limit <= 0:
            return ()
        with self._database.session() as session:
            rows = session.scalars(
                select(PackRow)
                .order_by(PackRow.created_at.desc(), PackRow.id.desc())
                .limit(limit * 2)
            ).all()
            usable = [
                row
                for row in rows
                if row.state
                not in {
                    PackState.FAILED.value,
                    PackState.BLOCKED.value,
                }
            ][:limit]
            for row in usable:
                session.expunge(row)
            return tuple(usable)

    @staticmethod
    def _pack_character_ids(row: PackRow) -> tuple[str, ...]:
        plan = row.payload_json.get("plan")
        if not isinstance(plan, dict):
            return ()
        raw = plan.get("character_ids", ())
        if not isinstance(raw, list | tuple):
            return ()
        return tuple(str(value) for value in raw)

    @staticmethod
    def _series_gap(series_id: str, recent: tuple[PackRow, ...]) -> int:
        for index, row in enumerate(recent):
            if row.series_id == series_id:
                return index
        return len(recent) + 1

    @staticmethod
    def _series_kind(series: SeriesProfile) -> SeriesPlanKind:
        if series.current_episode == 0:
            return SeriesPlanKind.START
        next_episode = series.current_episode + 1
        every = series.editorial_policy.bonus_every
        if every is not None and next_episode % every == 0:
            return SeriesPlanKind.BONUS
        return SeriesPlanKind.CONTINUE

    @staticmethod
    def _branch_key(option: CharacterOption) -> str:
        return option.branch or option.namespace or "__unknown_branch__"

    @staticmethod
    def _group_key(option: CharacterOption) -> str:
        return option.group or option.namespace or "__unknown_group__"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
