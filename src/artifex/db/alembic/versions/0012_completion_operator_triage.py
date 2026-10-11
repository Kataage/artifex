"""Persist non-delivery dismissal reasons and operator identity.

Revision ID: 0012_completion_operator_triage
Revises: 0011_completion_reconciliation
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012_completion_operator_triage"
down_revision: str | None = "0011_completion_reconciliation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "completion_notifications",
        sa.Column("dismissal_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "completion_notifications",
        sa.Column("dismissed_by_user_id", sa.String(length=40), nullable=True),
    )
    op.add_column(
        "completion_notifications",
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("completion_notifications", "dismissed_at")
    op.drop_column("completion_notifications", "dismissed_by_user_id")
    op.drop_column("completion_notifications", "dismissal_reason")
