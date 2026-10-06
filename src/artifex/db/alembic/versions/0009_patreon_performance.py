"""Persist Patreon publication links and performance snapshots.

Revision ID: 0009_patreon_performance
Revises: 0008_lora_validation
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_patreon_performance"
down_revision: str | None = "0008_lora_validation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "publication_links",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("platform", sa.String(length=64), nullable=False),
        sa.Column("external_post_id", sa.String(length=200), nullable=False),
        sa.Column("pack_id", sa.String(length=64), nullable=False),
        sa.Column("scene_id", sa.String(length=64), nullable=True),
        sa.Column("publication_tier", sa.String(length=32), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["pack_id"], ["packs.id"]),
        sa.ForeignKeyConstraint(["scene_id"], ["scenes.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "platform",
            "external_post_id",
            name="uq_publication_link_platform_post",
        ),
    )
    for column in (
        "platform",
        "external_post_id",
        "pack_id",
        "scene_id",
        "publication_tier",
        "source",
    ):
        op.create_index(
            op.f(f"ix_publication_links_{column}"),
            "publication_links",
            [column],
            unique=False,
        )

    op.create_table(
        "performance_snapshots",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("publication_link_id", sa.String(length=64), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("provenance_json", sa.JSON(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["publication_link_id"],
            ["publication_links.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "publication_link_id",
            "observed_at",
            name="uq_performance_snapshot_link_observed",
        ),
    )
    for column in (
        "publication_link_id",
        "observed_at",
        "source",
    ):
        op.create_index(
            op.f(f"ix_performance_snapshots_{column}"),
            "performance_snapshots",
            [column],
            unique=False,
        )


def downgrade() -> None:
    for column in ("source", "observed_at", "publication_link_id"):
        op.drop_index(
            op.f(f"ix_performance_snapshots_{column}"),
            table_name="performance_snapshots",
        )
    op.drop_table("performance_snapshots")
    for column in (
        "source",
        "publication_tier",
        "scene_id",
        "pack_id",
        "external_post_id",
        "platform",
    ):
        op.drop_index(
            op.f(f"ix_publication_links_{column}"),
            table_name="publication_links",
        )
    op.drop_table("publication_links")
