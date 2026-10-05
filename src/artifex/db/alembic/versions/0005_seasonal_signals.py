"""Add seasonal signal persistence.

Revision ID: 0005_seasonal_signals
Revises: 0004_editorial
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_seasonal_signals"
down_revision: str | None = "0004_editorial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "seasonal_signals",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("external_id", sa.String(length=200), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("relevance", sa.Float(), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
    )
    op.create_index("ix_seasonal_signals_provider", "seasonal_signals", ["provider"])
    op.create_index("ix_seasonal_signals_external_id", "seasonal_signals", ["external_id"])
    op.create_index("ix_seasonal_signals_title", "seasonal_signals", ["title"])
    op.create_index("ix_seasonal_signals_starts_at", "seasonal_signals", ["starts_at"])
    op.create_index("ix_seasonal_signals_ends_at", "seasonal_signals", ["ends_at"])


def downgrade() -> None:
    op.drop_table("seasonal_signals")
