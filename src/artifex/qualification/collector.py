from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime

from sqlalchemy import and_, or_, select

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
        scan_limit: int = 10000,
    ) -> dict[str, object]:
        if not 1 <= max_packs <= 1000:
            raise ValueError("max_packs must be between 1 and 1000")
        if not 1 <= scan_limit <= 100000:
            raise ValueError("scan_limit must be between 1 and 100000")
        qualification = self._service.load(session_id)
        self._service.require_collection_candidate(qualification)

        # Only Packs CREATED after this qualification began can be selected.
        # Keyset pagination avoids silently starving newly finalized Packs
        # after the first 250 rows. The scan is bounded and read-only.
        evidence: dict[str, dict[str, object]] = {}
        invalid: list[dict[str, str]] = []
        cursor: tuple[datetime, str] | None = None
        scanned = 0
        pages = 0
        truncated = False
        while scanned < scan_limit:
            remaining = scan_limit - scanned
            with self._database.session() as db_session:
                query = select(PackRow.id, PackRow.created_at).where(
                    PackRow.state == PackState.FINALIZED.value,
                    PackRow.created_at >= qualification.created_at,
                )
                if cursor is not None:
                    query = query.where(or_(
                        PackRow.created_at > cursor[0],
                        and_(PackRow.created_at == cursor[0], PackRow.id > cursor[1]),
                    ))
                batch = db_session.execute(
                    query.order_by(PackRow.created_at.asc(), PackRow.id.asc())
                    .limit(min(max_packs, remaining) + 1)
                ).all()
            count = min(max_packs, remaining, len(batch))
            if not count:
                break
            for pack_id, created_at in batch[:count]:
                try:
                    evidence[pack_id] = self._service.inspect_pack(pack_id)
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    invalid.append({"pack_id": pack_id, "reason": str(exc)})
                cursor = (created_at, pack_id)
            scanned += count
            pages += 1
            if count < len(batch) and scanned >= scan_limit:
                truncated = True
                break
            if len(batch) <= count:
                break
        else:
            # At the exact configured boundary, only a real extra row
            # justifies the truncated flag. Never guess there are no more.
            assert cursor is not None
            with self._database.session() as db_session:
                truncated = db_session.scalar(
                    select(PackRow.id).where(
                        PackRow.state == PackState.FINALIZED.value,
                        PackRow.created_at >= qualification.created_at,
                        or_(
                            PackRow.created_at > cursor[0],
                            and_(
                                PackRow.created_at == cursor[0],
                                PackRow.id > cursor[1],
                            ),
                        ),
                    ).order_by(
                        PackRow.created_at.asc(), PackRow.id.asc(),
                    ).limit(1)
                ) is not None

        # Select unattended candidates only from real continuous-daemon
        # completion events. Arbitrary finalized Pack triples cannot qualify.
        daemon_runs = self._service.unattended_run_packs(
            since=qualification.created_at,
        )
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
            for candidate in self._candidates(
                stage, evidence, unattended_runs=daemon_runs,
            ):
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
            "finalized_packs_scanned": scanned,
            "scan_pages": pages,
            "scan_limit": scan_limit,
            "valid_packs": len(evidence),
            "invalid_packs": invalid,
            "scan_truncated": truncated,
            "scan_incomplete": truncated,
            "stages": statuses,
            "remaining_stages": [
                evidence.stage.value for evidence in qualification.stages.values()
                if evidence.status is QualificationStatus.PENDING
            ],
            # Only qualify verify can assess the complete 14-stage ladder.
            "production_qualified": False,
        }

    @staticmethod
    def _candidates(
        stage: QualificationStage,
        evidence: dict[str, dict[str, object]],
        *,
        unattended_runs: dict[str, tuple[str, ...]] | None = None,
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
            for run_id, pack_ids in sorted((unattended_runs or {}).items()):
                verified = tuple(
                    pack_id for pack_id in pack_ids if pack_id in evidence
                )
                if len(verified) >= 3:
                    yield verified[:3]
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
