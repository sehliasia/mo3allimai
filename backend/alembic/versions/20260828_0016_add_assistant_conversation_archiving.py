"""add reversible assistant conversation archiving"""

from alembic import op
import sqlalchemy as sa


revision = "20260828_0016"
down_revision = "20260825_0015"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("assistant_conversations", sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_assistant_conversations_archived_at", "assistant_conversations", ["archived_at"])
    op.create_index(
        "ix_assistant_conversations_user_archived_updated",
        "assistant_conversations",
        ["user_id", "archived_at", "updated_at"],
    )


def downgrade():
    op.drop_index("ix_assistant_conversations_user_archived_updated", table_name="assistant_conversations")
    op.drop_index("ix_assistant_conversations_archived_at", table_name="assistant_conversations")
    op.drop_column("assistant_conversations", "archived_at")
