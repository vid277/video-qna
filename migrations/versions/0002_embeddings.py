"""Embeddings: one vector per key frame, event clip, change crop and frame crop.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import HALFVEC

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("create extension if not exists vector")

    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("owner", sa.Text, nullable=False),
        sa.Column("owner_id", sa.Integer, nullable=False),
        sa.Column("part", sa.Integer, nullable=False),
        sa.Column("camera_id", sa.Text, nullable=False),
        sa.Column("event_id", sa.Integer, sa.ForeignKey("events.id", ondelete="CASCADE"), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("vector", HALFVEC(EMBEDDING_DIM), nullable=False),
    )
    op.create_index("embeddings_by_owner", "embeddings", ["owner", "owner_id"])
    op.create_index("embeddings_by_event", "embeddings", ["event_id", "owner"])
    op.create_index("embeddings_by_time", "embeddings", ["camera_id", "captured_at"])
    op.create_index(
        "embeddings_search",
        "embeddings",
        ["vector"],
        postgresql_using="hnsw",
        postgresql_ops={"vector": "halfvec_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_table("embeddings")
