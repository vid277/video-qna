"""Caption deltas: each event lists what is new since the previous event's summary.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("events", sa.Column("new", JSONB))


def downgrade() -> None:
    op.drop_column("events", "new")
