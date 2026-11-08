"""Changes get their own detection time instead of their event's end.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("changes", sa.Column("detected_at", sa.DateTime(timezone=True)))
    op.execute(
        "update changes set detected_at = coalesce(events.ended_at, events.started_at)"
        " from events where events.id = changes.event_id"
    )
    op.alter_column("changes", "detected_at", nullable=False)
    op.create_index("changes_by_time", "changes", ["detected_at"])


def downgrade() -> None:
    op.drop_index("changes_by_time", "changes")
    op.drop_column("changes", "detected_at")
