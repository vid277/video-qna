"""Jobs: embeddings record their model, failing jobs back off, hour and day summaries.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FIRST_EMBEDDING_MODEL = "mongodb/voyage-multimodal-3.5"


def upgrade() -> None:
    op.add_column("embeddings", sa.Column("model", sa.Text, nullable=False, server_default=FIRST_EMBEDDING_MODEL))
    op.alter_column("embeddings", "model", server_default=None)
    op.execute(
        "delete from embeddings a using embeddings b"
        " where a.owner = b.owner and a.owner_id = b.owner_id and a.part = b.part and a.model = b.model and a.id < b.id"
    )
    op.create_index("embeddings_unique", "embeddings", ["owner", "owner_id", "part", "model"], unique=True)

    op.create_table(
        "job_failures",
        sa.Column("kind", sa.Text, primary_key=True),
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("attempts", sa.Integer, nullable=False),
        sa.Column("last_error", sa.Text, nullable=False),
        sa.Column("retry_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "summaries",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("camera_id", sa.Text, nullable=False),
        sa.Column("period", sa.Text, nullable=False),
        sa.Column("start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("events", sa.Integer, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("summaries_unique", "summaries", ["camera_id", "period", "start"], unique=True)


def downgrade() -> None:
    op.drop_table("summaries")
    op.drop_table("job_failures")
    op.drop_index("embeddings_unique", "embeddings")
    op.drop_column("embeddings", "model")
