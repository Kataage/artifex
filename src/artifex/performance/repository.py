from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select

from artifex.db import Database
from artifex.db.models import (
    PackRow,
    PerformanceSnapshotRow,
    PublicationLinkRow,
    SceneRow,
)
from artifex.performance.models import (
    PerformanceMetrics,
    PerformanceSnapshot,
    PublicationLink,
)


class PerformanceRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def upsert_publication(
        self,
        *,
        platform: str,
        external_post_id: str,
        pack_id: str,
        scene_id: str | None = None,
        publication_tier: str | None = None,
        url: str | None = None,
        published_at: datetime | None = None,
        source: str,
        metadata: dict[str, object] | None = None,
    ) -> PublicationLink:
        now = datetime.now(UTC)
        with self._database.session() as session:
            pack = session.get(PackRow, pack_id)
            if pack is None:
                raise KeyError(f"unknown pack: {pack_id}")
            if scene_id is not None:
                scene = session.get(SceneRow, scene_id)
                if scene is None:
                    raise KeyError(f"unknown scene: {scene_id}")
                if scene.pack_id != pack_id:
                    raise ValueError(
                        f"scene {scene_id} does not belong to pack {pack_id}"
                    )

            row = session.scalar(
                select(PublicationLinkRow).where(
                    PublicationLinkRow.platform == platform,
                    PublicationLinkRow.external_post_id == external_post_id,
                )
            )
            if row is None:
                row = PublicationLinkRow(
                    id=uuid4().hex,
                    platform=platform,
                    external_post_id=external_post_id,
                    pack_id=pack_id,
                    scene_id=scene_id,
                    publication_tier=publication_tier,
                    url=url,
                    published_at=published_at,
                    source=source,
                    metadata_json=dict(metadata or {}),
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                session.flush()
            else:
                if row.pack_id != pack_id:
                    raise ValueError(
                        "publication is already linked to a different pack: "
                        f"{platform}:{external_post_id} -> {row.pack_id}"
                    )
                if (
                    row.scene_id is not None
                    and scene_id is not None
                    and row.scene_id != scene_id
                ):
                    raise ValueError(
                        "publication is already linked to a different scene: "
                        f"{platform}:{external_post_id} -> {row.scene_id}"
                    )
                row.scene_id = scene_id or row.scene_id
                row.publication_tier = publication_tier or row.publication_tier
                row.url = url or row.url
                row.published_at = published_at or row.published_at
                row.source = source
                row.metadata_json = {
                    **dict(row.metadata_json or {}),
                    **dict(metadata or {}),
                }
                row.updated_at = now
                session.flush()
            return self._publication_from_row(row)

    def get_publication(
        self,
        platform: str,
        external_post_id: str,
    ) -> PublicationLink | None:
        with self._database.session() as session:
            row = session.scalar(
                select(PublicationLinkRow).where(
                    PublicationLinkRow.platform == platform,
                    PublicationLinkRow.external_post_id == external_post_id,
                )
            )
            return None if row is None else self._publication_from_row(row)

    def add_snapshot(
        self,
        publication_link_id: str,
        *,
        observed_at: datetime,
        metrics: PerformanceMetrics,
        source: str,
        provenance: dict[str, object] | None = None,
    ) -> PerformanceSnapshot:
        observed = self._utc(observed_at)
        now = datetime.now(UTC)
        with self._database.session() as session:
            link = session.get(PublicationLinkRow, publication_link_id)
            if link is None:
                raise KeyError(
                    f"unknown publication link: {publication_link_id}"
                )
            row = session.scalar(
                select(PerformanceSnapshotRow).where(
                    PerformanceSnapshotRow.publication_link_id
                    == publication_link_id,
                    PerformanceSnapshotRow.observed_at == observed,
                )
            )
            payload = metrics.model_dump(mode="json", exclude_none=True)
            if row is None:
                row = PerformanceSnapshotRow(
                    id=uuid4().hex,
                    publication_link_id=publication_link_id,
                    observed_at=observed,
                    metrics_json=payload,
                    source=source,
                    provenance_json=dict(provenance or {}),
                    ingested_at=now,
                )
                session.add(row)
                session.flush()
            else:
                row.metrics_json = payload
                row.source = source
                row.provenance_json = dict(provenance or {})
                row.ingested_at = now
                session.flush()
            return self._snapshot_from_row(row)

    def latest_snapshots(
        self,
        *,
        platform: str | None = None,
    ) -> tuple[tuple[PublicationLink, PerformanceSnapshot], ...]:
        with self._database.session() as session:
            links_query = select(PublicationLinkRow).order_by(
                PublicationLinkRow.id.asc()
            )
            if platform is not None:
                links_query = links_query.where(
                    PublicationLinkRow.platform == platform
                )
            links = session.scalars(links_query).all()
            result: list[tuple[PublicationLink, PerformanceSnapshot]] = []
            for link in links:
                snapshot = session.scalar(
                    select(PerformanceSnapshotRow)
                    .where(
                        PerformanceSnapshotRow.publication_link_id == link.id
                    )
                    .order_by(
                        PerformanceSnapshotRow.observed_at.desc(),
                        PerformanceSnapshotRow.ingested_at.desc(),
                        PerformanceSnapshotRow.id.desc(),
                    )
                    .limit(1)
                )
                if snapshot is None:
                    continue
                result.append(
                    (
                        self._publication_from_row(link),
                        self._snapshot_from_row(snapshot),
                    )
                )
            return tuple(result)

    def snapshots_for_publication(
        self,
        publication_link_id: str,
    ) -> tuple[PerformanceSnapshot, ...]:
        with self._database.session() as session:
            rows = session.scalars(
                select(PerformanceSnapshotRow)
                .where(
                    PerformanceSnapshotRow.publication_link_id
                    == publication_link_id
                )
                .order_by(
                    PerformanceSnapshotRow.observed_at.asc(),
                    PerformanceSnapshotRow.id.asc(),
                )
            ).all()
            return tuple(self._snapshot_from_row(row) for row in rows)

    @staticmethod
    def _publication_from_row(row: PublicationLinkRow) -> PublicationLink:
        return PublicationLink(
            id=row.id,
            platform=row.platform,
            external_post_id=row.external_post_id,
            pack_id=row.pack_id,
            scene_id=row.scene_id,
            publication_tier=row.publication_tier,
            url=row.url,
            published_at=row.published_at,
            source=row.source,
            metadata=dict(row.metadata_json or {}),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _snapshot_from_row(row: PerformanceSnapshotRow) -> PerformanceSnapshot:
        return PerformanceSnapshot(
            id=row.id,
            publication_link_id=row.publication_link_id,
            observed_at=row.observed_at,
            metrics=PerformanceMetrics.model_validate(row.metrics_json),
            source=row.source,
            provenance=dict(row.provenance_json or {}),
            ingested_at=row.ingested_at,
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
