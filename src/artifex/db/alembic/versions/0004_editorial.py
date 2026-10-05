"""Add editorial scheduling and inventory lifecycle.

Revision ID: 0004_editorial
Revises: 0003_llm_provenance
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_editorial"
down_revision: str | None = "0003_llm_provenance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pack_inventory",
        sa.Column("pack_id", sa.String(length=64), sa.ForeignKey("packs.id"), primary_key=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("reserved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_pack_inventory_state", "pack_inventory", ["state"])
    op.create_index("ix_pack_inventory_expires_at", "pack_inventory", ["expires_at"])
    op.create_index("ix_pack_inventory_updated_at", "pack_inventory", ["updated_at"])

    op.create_table(
        "editorial_decisions",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("series_id", sa.String(length=64), sa.ForeignKey("series.id"), nullable=True),
        sa.Column("concept_id", sa.String(length=64), sa.ForeignKey("concepts.id"), nullable=True),
        sa.Column("pack_id", sa.String(length=64), sa.ForeignKey("packs.id"), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_editorial_decisions_action", "editorial_decisions", ["action"])
    op.create_index("ix_editorial_decisions_status", "editorial_decisions", ["status"])
    op.create_index("ix_editorial_decisions_series_id", "editorial_decisions", ["series_id"])
    op.create_index("ix_editorial_decisions_concept_id", "editorial_decisions", ["concept_id"])
    op.create_index("ix_editorial_decisions_pack_id", "editorial_decisions", ["pack_id"])
    op.create_index("ix_editorial_decisions_created_at", "editorial_decisions", ["created_at"])


def downgrade() -> None:
    op.drop_table("editorial_decisions")
    op.drop_table("pack_inventory")
