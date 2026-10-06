"""Persist automated LoRA validation runs.

Revision ID: 0008_lora_validation
Revises: 0007_semantic_embeddings
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_lora_validation"
down_revision: str | None = "0007_semantic_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "lora_validation_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("lora_id", sa.String(length=200), nullable=False),
        sa.Column("checksum", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=True),
        sa.Column("error_json", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["lora_id"], ["loras.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_lora_validation_runs_lora_id"),
        "lora_validation_runs",
        ["lora_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_lora_validation_runs_checksum"),
        "lora_validation_runs",
        ["checksum"],
        unique=False,
    )
    op.create_index(
        op.f("ix_lora_validation_runs_status"),
        "lora_validation_runs",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_lora_validation_runs_started_at"),
        "lora_validation_runs",
        ["started_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_lora_validation_runs_started_at"),
        table_name="lora_validation_runs",
    )
    op.drop_index(
        op.f("ix_lora_validation_runs_status"),
        table_name="lora_validation_runs",
    )
    op.drop_index(
        op.f("ix_lora_validation_runs_checksum"),
        table_name="lora_validation_runs",
    )
    op.drop_index(
        op.f("ix_lora_validation_runs_lora_id"),
        table_name="lora_validation_runs",
    )
    op.drop_table("lora_validation_runs")
