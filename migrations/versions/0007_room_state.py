"""Room state: each change gets a label, where it is, and the item it belongs to.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("changes", sa.Column("label", sa.Text))
    op.add_column("changes", sa.Column("place", sa.Text))
    op.add_column("changes", sa.Column("item_id", sa.Integer))
    op.create_index("changes_by_item", "changes", ["item_id"])


def downgrade() -> None:
    op.drop_index("changes_by_item", "changes")
    for column in ("item_id", "place", "label"):
        op.drop_column("changes", column)
