from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import PackRow
from artifex.domain import PackState
from artifex.qualification.models import (
    QualificationStage,
    QualificationStatus,
)
from artifex.qualification.service import QualificationService

# Stages with their own live proof (8-hour observation / new GPU replay) must
# never be manufactured from existing Packs. Doctor belongs to session start.
_COLLECTIBLE = (
    QualificationStage.SINGLE_CHARACTER,
    QualificationStage.LORA_REQUIRED,
    QualificationStage.DUO,
    QualificationStage.GROUP,
    QualificationStage.PUBLIC_MEMBER,
    QualificationStage.SERIES_CONTINUATION,
    QualificationStage.FORCED_RETRY,
    QualificationStage.RESTART_GENERATION,
    QualificationStage.BACKEND_RECOVERY,
    QualificationStage.UNATTENDED_MULTI_PACK,
    QualificationStage.DISCORD_CONTROLS,
)


class QualificationEvidenceCollector:
    """Safely register real, already-persisted evidence without GPU side effects."""

    def __init__(self, service: QualificationService, database: Database) -> None:
        self._service = service
        self._database = database

    def collect(
        self,
        session_id: str,
        *,
        apply: bool = False,
        max_packs: int = 250,
    ) -> dict[str, object]:
        if not 1 <= max_packs <= 1000:
            raise ValueError("max_packs must be between 1 and 1000")
        qualification = self._service.load(session_id)
        self._service.require_collection_candidate(qualification)

        # Only Packs CREATED after this qualification began can be selected.
        # Completed before session creation, a Pack cannot prove this run.
        with self._database.session() as db_session:
            rows = db_session.scalars(
                select(PackRow)
                .where(
                    PackRow.state == PackState.FINALIZED.value,
                    PackRow.created_at >= qualification.created_at,
                )
                .order_by(PackRow.created_at.asc(), PackRow.id.asc())
                .limit(max_packs + 1)
            ).all()
            ids = [row.id for row in rows]
        truncated = len(ids) > max_packs
        ids = ids[:max_packs]

        evidence: dict[str, dict[str, object]] = {}
        invalid: list[dict[str, str]] = []
        for pack_id in ids:
            try:
                evidence[pack_id] = self._service.inspect_pack(pack_id)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                invalid.append({"pack_id": pack_id, "reason": str(exc)})

        statuses: list[dict[str, object]] = []
        for stage in _COLLECTIBLE:
            current = qualification.stage(stage)
            if current.status is not QualificationStatus.PENDING:
                statuses.append(
                    {
                        "stage": stage.value,
                        "state": "already_recorded",
                        "status": current.status.value,
                        "pack_ids": list(current.pack_ids),
                    }
                )
                continue

            chosen: tuple[str, ...] = ()
            failure_reason = "no matching verified evidence in this qualification session"
            for candidate in self._candidates(stage, evidence):
                try:
                    self._service.validate_candidate(qualification, stage, candidate)
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    failure_reason = str(exc)
                    continue
                chosen = candidate
                break

            if stage is QualificationStage.DISCORD_CONTROLS and not chosen:
                # A verified Discord stage has no Pack IDs. A successful
                # empty-tuple validation above is distinguished separately.
                try:
                    self._service.validate_candidate(qualification, stage, ())
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    failure_reason = str(exc)
                else:
                    if apply:
                        qualification = self._service.record(
                            session_id, stage, status=QualificationStatus.PASS
                        )
                    statuses.append(
                        {
                            "stage": stage.value,
                            "state": "recorded" if apply else "ready",
                            "pack_ids": [],
                        }
                    )
                    continue

            if not chosen:
                statuses.append(
                    {
                        "stage": stage.value,
                        "state": "missing",
                        "reason": failure_reason,
                        "pack_ids": [],
                    }
                )
                continue
            if apply:
                # The authoritative service revalidates everything before
                # writing; this is not a shortcut around record().
                qualification = self._service.record(
                    session_id,
                    stage,
                    status=QualificationStatus.PASS,
                    pack_ids=chosen,
                )
            statuses.append(
                {
                    "stage": stage.value,
                    "state": "recorded" if apply else "ready",
                    "pack_ids": list(chosen),
                }
            )

        return {
            "session_id": session_id,
            "mode": "apply" if apply else "preview",
            "finalized_packs_scanned": len(ids),
            "valid_packs": len(evidence),
            "invalid_packs": invalid,
            "scan_truncated": truncated,
            "stages": statuses,
            "remaining_stages": [
                evidence.stage.value for evidence in qualification.stages.values()
                if stage.status is QualificationStatus.PENDING
            ],
            # Only qualify verify can assess the complete 14-stage ladder.
            "production_qualified": False,
        }

    @staticmethod
    def _candidates(
        stage: QualificationStage,
        evidence: dict[str, dict[str, object]],
    ) -> Iterator[tuple[str, ...]]:
        if stage is QualificationStage.DISCORD_CONTROLS:
            return
        if stage is QualificationStage.SERIES_CONTINUATION:
            by_series: dict[str, list[str]] = defaultdict(list)
            for pack_id, data in evidence.items():
                series_id = data.get("series_id")
                if isinstance(series_id, str) and series_id:
                    by_series[series_id].append(pack_id)
            for series_id in sorted(by_series):
                related = by_series[series_id]
                if len(related) >= 2:
                    yield (related[0], related[1])
            return
        if stage is QualificationStage.UNATTENDED_MULTI_PACK:
            if len(evidence) >= 3:
                yield tuple(list(evidence)[:3])
            return
        for pack_id, data in evidence.items():
            characters = data.get("character_ids", [])
            tiers = data.get("publication_tiers", [])
            if not isinstance(characters, list):
                continue
            if not isinstance(tiers, list):
                continue
            if stage is QualificationStage.SINGLE_CHARACTER and len(characters) != 1:
                continue
            if stage is QualificationStage.LORA_REQUIRED and not data.get("lora_ids"):
                continue
            if stage is QualificationStage.DUO and len(characters) != 2:
                continue
            if stage is QualificationStage.GROUP and len(characters) < 3:
                continue
            if stage is QualificationStage.PUBLIC_MEMBER and not {"public", "member"} <= set(tiers):
                continue
            if stage is QualificationStage.FORCED_RETRY and not data.get("retry_count"):
                continue
            yield (pack_id,)
