"""Changes record a window: the last time the spot looked as before, and when the change was first seen.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("changes", sa.Column("changed_after", sa.DateTime(timezone=True)))
    op.execute("update changes set changed_after = detected_at")
    op.alter_column("changes", "changed_after", nullable=False)


def downgrade() -> None:
    op.drop_column("changes", "changed_after")
