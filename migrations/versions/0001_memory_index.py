"""Memory index: events, frames and scene changes.

Revision ID: 0001
Revises:
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("camera_id", sa.Text, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
    )
    op.create_index("events_by_time", "events", ["camera_id", "started_at"])

    op.create_table(
        "frames",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("camera_id", sa.Text, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("path", sa.Text),
        sa.Column("video_path", sa.Text),
        sa.Column("video_index", sa.Integer),
        sa.Column("event_id", sa.Integer, sa.ForeignKey("events.id", ondelete="SET NULL")),
        sa.Column("is_key", sa.Boolean, nullable=False),
        sa.Column("foreground", JSONB, nullable=False),
    )
    op.create_index("frames_by_time", "frames", ["camera_id", "captured_at"])
    op.create_index("frames_by_event", "frames", ["event_id"])
    op.create_index("frames_by_video", "frames", ["video_path"])

    op.create_table(
        "changes",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("event_id", sa.Integer, sa.ForeignKey("events.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("box", JSONB, nullable=False),
        sa.Column("source", JSONB),
        sa.Column("before_path", sa.Text, nullable=False),
        sa.Column("after_path", sa.Text, nullable=False),
    )
    op.create_index("changes_by_event", "changes", ["event_id"])


def downgrade() -> None:
    for table in ("changes", "frames", "events"):
        op.drop_table(table)
