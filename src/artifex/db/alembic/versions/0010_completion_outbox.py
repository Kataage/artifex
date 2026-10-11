"""Create durable, per-Pack Discord completion notification outbox.

Revision ID: 0010_completion_outbox
Revises: 0009_patreon_performance
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_completion_outbox"
down_revision: str | None = "0009_patreon_performance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "completion_notifications",
        sa.Column("pack_id", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["pack_id"], ["packs.id"]),
        sa.PrimaryKeyConstraint("pack_id"),
    )
    op.create_index(
        "ix_completion_notifications_state",
        "completion_notifications",
        ["state"],
    )
    op.create_index(
        "ix_completion_notifications_created_at",
        "completion_notifications",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_completion_notifications_created_at",
        table_name="completion_notifications",
    )
    op.drop_index(
        "ix_completion_notifications_state",
        table_name="completion_notifications",
    )
    op.drop_table("completion_notifications")
