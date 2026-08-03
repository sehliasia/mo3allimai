"""add teacher soft deletion fields"""
from alembic import op
import sqlalchemy as sa

revision = "20260803_0002"
down_revision = "20260731_0001"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("users", sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("users", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("deleted_by", sa.Integer(), nullable=True))
    op.create_foreign_key("fk_users_deleted_by", "users", "users", ["deleted_by"], ["id"])

def downgrade():
    op.drop_constraint("fk_users_deleted_by", "users", type_="foreignkey")
    op.drop_column("users", "deleted_by")
    op.drop_column("users", "deleted_at")
    op.drop_column("users", "is_deleted")
