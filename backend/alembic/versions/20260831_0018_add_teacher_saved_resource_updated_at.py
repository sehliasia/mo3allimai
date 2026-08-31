"""add teacher saved resource updated timestamp"""
from alembic import op
import sqlalchemy as sa

revision = "20260831_0018"
down_revision = "20260830_0017"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("teacher_saved_resources", sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))

def downgrade():
    op.drop_column("teacher_saved_resources", "updated_at")
