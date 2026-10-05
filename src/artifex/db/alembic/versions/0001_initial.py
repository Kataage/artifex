"""Initial Artifex schema.

Revision ID: 0001_initial
Revises:
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.JSON:
    return sa.JSON()


def upgrade() -> None:
    op.create_table(
        "characters",
        sa.Column("id", sa.String(length=160), primary_key=True),
        sa.Column("display_name", sa.String(length=240), nullable=False),
        sa.Column("namespace", sa.String(length=160), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("lora_policy", sa.String(length=32), nullable=False),
        sa.Column("readiness", sa.Float(), nullable=False),
        sa.Column("profile_json", _json(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_generated", sa.Integer(), nullable=False),
    )
    op.create_index("ix_characters_namespace", "characters", ["namespace"])

    op.create_table(
        "loras",
        sa.Column("id", sa.String(length=200), primary_key=True),
        sa.Column("path", sa.Text(), nullable=False, unique=True),
        sa.Column("checksum", sa.String(length=128), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("lora_type", sa.String(length=64), nullable=False),
        sa.Column("readiness", sa.Float(), nullable=False),
        sa.Column("metadata_json", _json(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_loras_checksum", "loras", ["checksum"])
    op.create_index("ix_loras_state", "loras", ["state"])

    op.create_table(
        "concepts",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("payload_json", _json(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("similarity_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_concepts_status", "concepts", ["status"])

    op.create_table(
        "series",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_episode", sa.Integer(), nullable=False),
        sa.Column("state_json", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_series_status", "series", ["status"])

    op.create_table(
        "packs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("concept_id", sa.String(length=64), sa.ForeignKey("concepts.id")),
        sa.Column("series_id", sa.String(length=64), sa.ForeignKey("series.id")),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("format_type", sa.String(length=64), nullable=False),
        sa.Column("payload_json", _json(), nullable=False),
        sa.Column("checkpoint_json", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_packs_state", "packs", ["state"])

    op.create_table(
        "scenes",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("pack_id", sa.String(length=64), sa.ForeignKey("packs.id"), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("publication_tier", sa.String(length=32), nullable=False),
        sa.Column("payload_json", _json(), nullable=False),
        sa.Column("selected_attempt_id", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_scenes_pack_id", "scenes", ["pack_id"])
    op.create_index("ix_scenes_state", "scenes", ["state"])

    op.create_table(
        "generation_attempts",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("scene_id", sa.String(length=64), sa.ForeignKey("scenes.id"), nullable=False),
        sa.Column(
            "parent_attempt_id",
            sa.String(length=64),
            sa.ForeignKey("generation_attempts.id"),
            nullable=True,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("backend_status", sa.String(length=32), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("negative_prompt", sa.Text(), nullable=False),
        sa.Column("provenance_json", _json(), nullable=False),
        sa.Column("error_json", _json(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_generation_attempts_scene_id", "generation_attempts", ["scene_id"])

    op.create_table(
        "evaluations",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "attempt_id",
            sa.String(length=64),
            sa.ForeignKey("generation_attempts.id"),
            nullable=False,
        ),
        sa.Column("result_state", sa.String(length=32), nullable=False),
        sa.Column("scores_json", _json(), nullable=False),
        sa.Column("reasons_json", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evaluations_attempt_id", "evaluations", ["attempt_id"])
    op.create_index("ix_evaluations_result_state", "evaluations", ["result_state"])

    op.create_table(
        "trend_signals",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("topic", sa.String(length=300), nullable=False),
        sa.Column("strength", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload_json", _json(), nullable=False),
    )
    op.create_index("ix_trend_signals_provider", "trend_signals", ["provider"])
    op.create_index("ix_trend_signals_topic", "trend_signals", ["topic"])
    op.create_index("ix_trend_signals_observed_at", "trend_signals", ["observed_at"])

    op.create_table(
        "review_queue",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("subject_type", sa.String(length=64), nullable=False),
        sa.Column("subject_id", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("payload_json", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_review_queue_subject_type", "review_queue", ["subject_type"])
    op.create_index("ix_review_queue_subject_id", "review_queue", ["subject_id"])
    op.create_index("ix_review_queue_state", "review_queue", ["state"])

    op.create_table(
        "policy_decisions",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("subject_type", sa.String(length=64), nullable=False),
        sa.Column("subject_id", sa.String(length=64), nullable=False),
        sa.Column("policy_name", sa.String(length=160), nullable=False),
        sa.Column("policy_version", sa.String(length=80), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("payload_json", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_policy_decisions_subject_type", "policy_decisions", ["subject_type"])
    op.create_index("ix_policy_decisions_subject_id", "policy_decisions", ["subject_id"])
    op.create_index("ix_policy_decisions_decision", "policy_decisions", ["decision"])

    op.create_table(
        "agent_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(length=120), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("payload_json", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_events_event_type", "agent_events", ["event_type"])
    op.create_index("ix_agent_events_severity", "agent_events", ["severity"])
    op.create_index("ix_agent_events_created_at", "agent_events", ["created_at"])

    op.create_table(
        "settings",
        sa.Column("key", sa.String(length=200), primary_key=True),
        sa.Column("value_json", _json(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "settings",
        "agent_events",
        "policy_decisions",
        "review_queue",
        "trend_signals",
        "evaluations",
        "generation_attempts",
        "scenes",
        "packs",
        "series",
        "concepts",
        "loras",
        "characters",
    ):
        op.drop_table(table)
