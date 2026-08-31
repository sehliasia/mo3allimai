"""create private teacher library and activity tables"""
from alembic import op
import sqlalchemy as sa
revision = "20260820_0010"
down_revision = "20260819_0009"
branch_labels = None
depends_on = None
def upgrade():
    op.create_table("teacher_library_documents", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False), sa.Column("title", sa.String(255), nullable=False), sa.Column("original_filename", sa.String(255), nullable=False), sa.Column("mime_type", sa.String(120), nullable=False), sa.Column("file_size", sa.Integer(), nullable=False), sa.Column("storage_key", sa.String(500), nullable=False, unique=True), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_teacher_library_documents_owner_id", "teacher_library_documents", ["owner_id"])
    op.create_table("teacher_saved_resources", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False), sa.Column("resource_type", sa.String(50), nullable=False), sa.Column("title", sa.String(255), nullable=False), sa.Column("cefr_level", sa.String(10)), sa.Column("theme", sa.String(255)), sa.Column("content", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_teacher_saved_resources_owner_id", "teacher_saved_resources", ["owner_id"])
    op.create_table("teacher_activities", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False), sa.Column("activity_type", sa.String(60), nullable=False), sa.Column("resource_type", sa.String(50)), sa.Column("resource_id", sa.Integer()), sa.Column("title", sa.String(255), nullable=False), sa.Column("metadata", sa.JSON()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_teacher_activities_owner_id", "teacher_activities", ["owner_id"])
def downgrade():
    op.drop_table("teacher_activities"); op.drop_table("teacher_saved_resources"); op.drop_table("teacher_library_documents")
