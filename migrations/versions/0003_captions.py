"""Captions: an event summary, the objects seen, and short facts with full-text search.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("events", sa.Column("summary", sa.Text))
    op.add_column("events", sa.Column("objects", JSONB))

    op.create_table(
        "facts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("event_id", sa.Integer, sa.ForeignKey("events.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("search", TSVECTOR, sa.Computed("to_tsvector('english', text)", persisted=True), nullable=False),
    )
    op.create_index("facts_by_event", "facts", ["event_id"])
    op.create_index("facts_search", "facts", ["search"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_table("facts")
    op.drop_column("events", "objects")
    op.drop_column("events", "summary")
