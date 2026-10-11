"""Persist independent read-only Discord message identity verification.

Revision ID: 0013_completion_receipt_verification
Revises: 0012_completion_operator_triage
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_completion_receipt_verification"
down_revision: str | None = "0012_completion_operator_triage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "completion_notifications",
        sa.Column("discord_verified_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("completion_notifications", "discord_verified_at")
