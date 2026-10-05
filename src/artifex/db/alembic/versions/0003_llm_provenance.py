"""Add LLM call provenance.

Revision ID: 0003_llm_provenance
Revises: 0002_research
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_llm_provenance"
down_revision: str | None = "0002_research"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_calls",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("backend", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=240), nullable=False),
        sa.Column("runtime_version", sa.String(length=240), nullable=True),
        sa.Column("schema_name", sa.String(length=160), nullable=False),
        sa.Column("context_digest", sa.String(length=64), nullable=False),
        sa.Column("schema_digest", sa.String(length=64), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("repair_index", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("response_digest", sa.String(length=64), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_llm_calls_backend", "llm_calls", ["backend"])
    op.create_index("ix_llm_calls_model", "llm_calls", ["model"])
    op.create_index("ix_llm_calls_schema_name", "llm_calls", ["schema_name"])
    op.create_index("ix_llm_calls_context_digest", "llm_calls", ["context_digest"])
    op.create_index("ix_llm_calls_schema_digest", "llm_calls", ["schema_digest"])
    op.create_index("ix_llm_calls_status", "llm_calls", ["status"])
    op.create_index("ix_llm_calls_response_digest", "llm_calls", ["response_digest"])
    op.create_index("ix_llm_calls_started_at", "llm_calls", ["started_at"])


def downgrade() -> None:
    op.drop_table("llm_calls")
