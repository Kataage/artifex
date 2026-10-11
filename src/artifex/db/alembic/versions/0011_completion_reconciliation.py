"""Store verified remote Discord message IDs for reconciled Pack notices.

Revision ID: 0011_completion_reconciliation
Revises: 0010_completion_outbox
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_completion_reconciliation"
down_revision: str | None = "0010_completion_outbox"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "completion_notifications",
        sa.Column("discord_message_id", sa.String(length=40), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("completion_notifications", "discord_message_id")
