from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from artifex.performance.models import (
    ManualPerformanceImport,
    ManualPerformanceRecord,
    PerformanceMetrics,
)
from artifex.performance.repository import PerformanceRepository


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _record_from_mapping(raw: dict[str, Any]) -> ManualPerformanceRecord:
    metrics_raw = raw.get("metrics")
    if isinstance(metrics_raw, dict):
        metrics = PerformanceMetrics.model_validate(metrics_raw)
    else:
        metrics = PerformanceMetrics(
            views=_optional_int(raw.get("views")),
            likes=_optional_int(raw.get("likes")),
            comments=_optional_int(raw.get("comments")),
            engagement_count=_optional_int(raw.get("engagement_count")),
            free_signups=_optional_int(raw.get("free_signups")),
            paid_conversions=_optional_int(raw.get("paid_conversions")),
            subscriber_delta=_optional_int(raw.get("subscriber_delta")),
            revenue_cents=_optional_int(raw.get("revenue_cents")),
            retention_rate=_optional_float(raw.get("retention_rate")),
        )
    metadata = raw.get("metadata")
    return ManualPerformanceRecord(
        external_post_id=str(raw["external_post_id"]),
        pack_id=str(raw["pack_id"]),
        scene_id=(
            str(raw["scene_id"])
            if raw.get("scene_id") not in {None, ""}
            else None
        ),
        publication_tier=(
            str(raw["publication_tier"])
            if raw.get("publication_tier") not in {None, ""}
            else None
        ),
        url=str(raw["url"]) if raw.get("url") not in {None, ""} else None,
        published_at=_optional_datetime(raw.get("published_at")),
        observed_at=(
            _optional_datetime(raw.get("observed_at"))
            or datetime.now().astimezone()
        ),
        metrics=metrics,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def load_manual_performance(path: Path) -> ManualPerformanceImport:
    source_path = path.expanduser().resolve()
    suffix = source_path.suffix.casefold()
    if suffix == ".csv":
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = tuple(
                _record_from_mapping(dict(row))
                for row in csv.DictReader(handle)
            )
        return ManualPerformanceImport(records=records)

    if suffix in {".jsonl", ".ndjson"}:
        records: list[ManualPerformanceRecord] = []
        for index, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL line {index}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    f"JSONL line {index} must contain an object"
                )
            records.append(_record_from_mapping(raw))
        return ManualPerformanceImport(records=tuple(records))

    if suffix != ".json":
        raise ValueError(
            "manual performance import must be .json, .jsonl/.ndjson, or .csv"
        )

    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid performance JSON: {exc}") from exc

    try:
        if isinstance(raw, list):
            return ManualPerformanceImport(
                records=tuple(
                    _record_from_mapping(item)
                    for item in raw
                    if isinstance(item, dict)
                )
            )
        if not isinstance(raw, dict):
            raise ValueError("performance JSON root must be an object or list")
        records_raw = raw.get("records")
        if not isinstance(records_raw, list):
            raise ValueError("performance JSON object requires records[]")
        return ManualPerformanceImport(
            platform=str(raw.get("platform") or "patreon"),
            source=str(raw.get("source") or "manual"),
            records=tuple(
                _record_from_mapping(item)
                for item in records_raw
                if isinstance(item, dict)
            ),
        )
    except (KeyError, TypeError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid performance import: {exc}") from exc


def ingest_manual_performance(
    repository: PerformanceRepository,
    payload: ManualPerformanceImport,
    *,
    import_path: Path | None = None,
) -> tuple[int, int]:
    publications = 0
    snapshots = 0
    seen_links: set[str] = set()
    provenance_base: dict[str, object] = {}
    if import_path is not None:
        provenance_base["import_path"] = str(
            import_path.expanduser().resolve()
        )

    for record in payload.records:
        link = repository.upsert_publication(
            platform=payload.platform,
            external_post_id=record.external_post_id,
            pack_id=record.pack_id,
            scene_id=record.scene_id,
            publication_tier=record.publication_tier,
            url=record.url,
            published_at=record.published_at,
            source=payload.source,
            metadata=record.metadata,
        )
        if link.id not in seen_links:
            seen_links.add(link.id)
            publications += 1
        repository.add_snapshot(
            link.id,
            observed_at=record.observed_at,
            metrics=record.metrics,
            source=payload.source,
            provenance={
                **provenance_base,
                "platform": payload.platform,
                "external_post_id": record.external_post_id,
            },
        )
        snapshots += 1
    return publications, snapshots
