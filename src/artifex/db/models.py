from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from artifex.db.base import Base


class CharacterRow(Base):
    __tablename__ = "characters"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(240))
    namespace: Mapped[str] = mapped_column(String(160), index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    lora_policy: Mapped[str] = mapped_column(String(32), default="optional")
    readiness: Mapped[float] = mapped_column(Float, default=0.0)
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_generated: Mapped[int] = mapped_column(Integer, default=0)


class LoRARow(Base):
    __tablename__ = "loras"

    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    path: Mapped[str] = mapped_column(Text, unique=True)
    checksum: Mapped[str | None] = mapped_column(String(128), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    lora_type: Mapped[str] = mapped_column(String(64), default="character")
    readiness: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class LoRAValidationRunRow(Base):
    __tablename__ = "lora_validation_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    lora_id: Mapped[str] = mapped_column(ForeignKey("loras.id"), index=True)
    checksum: Mapped[str | None] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    report_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConceptRow(Base):
    __tablename__ = "concepts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    score: Mapped[float | None] = mapped_column(Float)
    similarity_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SeriesRow(Base):
    __tablename__ = "series"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    current_episode: Mapped[int] = mapped_column(Integer, default=0)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class PackRow(Base):
    __tablename__ = "packs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    concept_id: Mapped[str | None] = mapped_column(ForeignKey("concepts.id"))
    series_id: Mapped[str | None] = mapped_column(ForeignKey("series.id"))
    state: Mapped[str] = mapped_column(String(32), index=True)
    format_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    checkpoint_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SceneRow(Base):
    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pack_id: Mapped[str] = mapped_column(ForeignKey("packs.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(32), index=True)
    publication_tier: Mapped[str] = mapped_column(String(32))
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    selected_attempt_id: Mapped[str | None] = mapped_column(String(64))


class GenerationAttemptRow(Base):
    __tablename__ = "generation_attempts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.id"), index=True)
    parent_attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("generation_attempts.id")
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    backend_status: Mapped[str] = mapped_column(String(32))
    seed: Mapped[int | None] = mapped_column(Integer)
    prompt: Mapped[str] = mapped_column(Text)
    negative_prompt: Mapped[str] = mapped_column(Text)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EvaluationRow(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("generation_attempts.id"), index=True)
    result_state: Mapped[str] = mapped_column(String(32), index=True)
    scores_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    reasons_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    classification_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SemanticEmbeddingRow(Base):
    __tablename__ = "semantic_embeddings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_type: Mapped[str] = mapped_column(String(64), index=True)
    subject_id: Mapped[str] = mapped_column(String(512), index=True)
    modality: Mapped[str] = mapped_column(String(16), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(100), index=True)
    model: Mapped[str] = mapped_column(String(240), index=True)
    revision: Mapped[str] = mapped_column(String(160), index=True)
    dimensions: Mapped[int] = mapped_column(Integer)
    vector_json: Mapped[list[float]] = mapped_column(JSON)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TrendSignalRow(Base):
    __tablename__ = "trend_signals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(100), index=True)
    topic: Mapped[str] = mapped_column(String(300), index=True)
    strength: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class SeasonalSignalRow(Base):
    __tablename__ = "seasonal_signals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(100), index=True)
    external_id: Mapped[str] = mapped_column(String(200), index=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    relevance: Mapped[float] = mapped_column(Float)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ReviewQueueRow(Base):
    __tablename__ = "review_queue"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_type: Mapped[str] = mapped_column(String(64), index=True)
    subject_id: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    reason: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PolicyDecisionRow(Base):
    __tablename__ = "policy_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_type: Mapped[str] = mapped_column(String(64), index=True)
    subject_id: Mapped[str] = mapped_column(String(64), index=True)
    policy_name: Mapped[str] = mapped_column(String(160))
    policy_version: Mapped[str] = mapped_column(String(80))
    decision: Mapped[str] = mapped_column(String(32), index=True)
    reason: Mapped[str | None] = mapped_column(Text)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AgentEventRow(Base):
    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(120), index=True)
    severity: Mapped[str] = mapped_column(String(32), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SettingRow(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    value_json: Mapped[Any] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ResearchRunRow(Base):
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    normalized_query: Mapped[str] = mapped_column(String(500), index=True)
    intent: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    provider: Mapped[str | None] = mapped_column(String(100), index=True)
    safesearch: Mapped[str] = mapped_column(String(16))
    adult: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ResearchEvidenceRow(Base):
    __tablename__ = "research_evidence"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.id"), index=True)
    provider: Mapped[str] = mapped_column(String(100), index=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(Text)
    canonical_url: Mapped[str] = mapped_column(Text, index=True)
    title: Mapped[str] = mapped_column(Text)
    snippet: Mapped[str] = mapped_column(Text)
    image_url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    adult: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ResearchBriefRow(Base):
    __tablename__ = "research_briefs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    topic: Mapped[str] = mapped_column(Text)
    cache_key: Mapped[str] = mapped_column(String(128), index=True)
    adult: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LlmCallRow(Base):
    __tablename__ = "llm_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    backend: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(240), index=True)
    runtime_version: Mapped[str | None] = mapped_column(String(240))
    schema_name: Mapped[str] = mapped_column(String(160), index=True)
    context_digest: Mapped[str] = mapped_column(String(64), index=True)
    schema_digest: Mapped[str] = mapped_column(String(64), index=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    repair_index: Mapped[int] = mapped_column(Integer, default=0)
    temperature: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), index=True)
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    response_text: Mapped[str | None] = mapped_column(Text)
    response_digest: Mapped[str | None] = mapped_column(String(64), index=True)
    error_text: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))



class PackInventoryRow(Base):
    __tablename__ = "pack_inventory"

    pack_id: Mapped[str] = mapped_column(ForeignKey("packs.id"), primary_key=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    reserved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class EditorialDecisionRow(Base):
    __tablename__ = "editorial_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    series_id: Mapped[str | None] = mapped_column(ForeignKey("series.id"), index=True)
    concept_id: Mapped[str | None] = mapped_column(ForeignKey("concepts.id"), index=True)
    pack_id: Mapped[str | None] = mapped_column(ForeignKey("packs.id"), index=True)
    reason: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
