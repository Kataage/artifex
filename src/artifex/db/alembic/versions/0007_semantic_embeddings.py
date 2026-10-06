"""Persist semantic embedding index metadata and vectors.

Revision ID: 0007_semantic_embeddings
Revises: 0006_content_classification
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_semantic_embeddings"
down_revision: str | None = "0006_content_classification"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "semantic_embeddings",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("subject_type", sa.String(length=64), nullable=False),
        sa.Column("subject_id", sa.String(length=512), nullable=False),
        sa.Column("modality", sa.String(length=16), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=240), nullable=False),
        sa.Column("revision", sa.String(length=160), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("vector_json", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "subject_type",
            "subject_id",
            "modality",
            "provider",
            "model",
            "revision",
            name="uq_semantic_embedding_subject_model",
        ),
    )
    op.create_index(
        op.f("ix_semantic_embeddings_subject_type"),
        "semantic_embeddings",
        ["subject_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_semantic_embeddings_subject_id"),
        "semantic_embeddings",
        ["subject_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_semantic_embeddings_modality"),
        "semantic_embeddings",
        ["modality"],
        unique=False,
    )
    op.create_index(
        op.f("ix_semantic_embeddings_content_hash"),
        "semantic_embeddings",
        ["content_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_semantic_embeddings_provider"),
        "semantic_embeddings",
        ["provider"],
        unique=False,
    )
    op.create_index(
        op.f("ix_semantic_embeddings_model"),
        "semantic_embeddings",
        ["model"],
        unique=False,
    )
    op.create_index(
        op.f("ix_semantic_embeddings_revision"),
        "semantic_embeddings",
        ["revision"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_semantic_embeddings_revision"), table_name="semantic_embeddings")
    op.drop_index(op.f("ix_semantic_embeddings_model"), table_name="semantic_embeddings")
    op.drop_index(op.f("ix_semantic_embeddings_provider"), table_name="semantic_embeddings")
    op.drop_index(op.f("ix_semantic_embeddings_content_hash"), table_name="semantic_embeddings")
    op.drop_index(op.f("ix_semantic_embeddings_modality"), table_name="semantic_embeddings")
    op.drop_index(op.f("ix_semantic_embeddings_subject_id"), table_name="semantic_embeddings")
    op.drop_index(op.f("ix_semantic_embeddings_subject_type"), table_name="semantic_embeddings")
    op.drop_table("semantic_embeddings")
