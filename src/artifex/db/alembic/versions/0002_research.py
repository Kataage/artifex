"""Add research evidence schema.

Revision ID: 0002_research
Revises: 0001_initial
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_research"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.JSON:
    return sa.JSON()


def upgrade() -> None:
    op.create_table(
        "research_runs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("normalized_query", sa.String(length=500), nullable=False),
        sa.Column("intent", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=True),
        sa.Column("safesearch", sa.String(length=16), nullable=False),
        sa.Column("adult", sa.Boolean(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload_json", _json(), nullable=False),
    )
    op.create_index("ix_research_runs_normalized_query", "research_runs", ["normalized_query"])
    op.create_index("ix_research_runs_intent", "research_runs", ["intent"])
    op.create_index("ix_research_runs_source", "research_runs", ["source"])
    op.create_index("ix_research_runs_provider", "research_runs", ["provider"])
    op.create_index("ix_research_runs_adult", "research_runs", ["adult"])
    op.create_index("ix_research_runs_state", "research_runs", ["state"])
    op.create_index("ix_research_runs_requested_at", "research_runs", ["requested_at"])
    op.create_index("ix_research_runs_expires_at", "research_runs", ["expires_at"])

    op.create_table(
        "research_evidence",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("research_runs.id"), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("snippet", sa.Text(), nullable=False),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("adult", sa.Boolean(), nullable=False),
        sa.Column("metadata_json", _json(), nullable=False),
    )
    op.create_index("ix_research_evidence_run_id", "research_evidence", ["run_id"])
    op.create_index("ix_research_evidence_provider", "research_evidence", ["provider"])
    op.create_index("ix_research_evidence_source", "research_evidence", ["source"])
    op.create_index("ix_research_evidence_canonical_url", "research_evidence", ["canonical_url"])
    op.create_index("ix_research_evidence_published_at", "research_evidence", ["published_at"])
    op.create_index("ix_research_evidence_observed_at", "research_evidence", ["observed_at"])
    op.create_index("ix_research_evidence_content_hash", "research_evidence", ["content_hash"])
    op.create_index("ix_research_evidence_adult", "research_evidence", ["adult"])

    op.create_table(
        "research_briefs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("cache_key", sa.String(length=128), nullable=False),
        sa.Column("adult", sa.Boolean(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("payload_json", _json(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_research_briefs_cache_key", "research_briefs", ["cache_key"])
    op.create_index("ix_research_briefs_adult", "research_briefs", ["adult"])
    op.create_index("ix_research_briefs_state", "research_briefs", ["state"])
    op.create_index("ix_research_briefs_created_at", "research_briefs", ["created_at"])
    op.create_index("ix_research_briefs_expires_at", "research_briefs", ["expires_at"])


def downgrade() -> None:
    op.drop_table("research_briefs")
    op.drop_table("research_evidence")
    op.drop_table("research_runs")
