"""create knowledge documents"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260812_0003"
down_revision = "20260803_0002"
branch_labels = None
depends_on = None


def upgrade():
    status = postgresql.ENUM("pending", name="knowledge_document_status", create_type=False)
    status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("document_type", sa.String(length=50), nullable=False),
        sa.Column("language", sa.String(length=20), nullable=False),
        sa.Column("cefr_level", sa.String(length=10), nullable=True),
        sa.Column("skill", sa.String(length=50), nullable=True),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("stored_filename", sa.String(length=255), nullable=False, unique=True),
        sa.Column("storage_path", sa.String(length=500), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("status", status, nullable=False, server_default=sa.text("'pending'::knowledge_document_status")),
        sa.Column("uploaded_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade():
    op.drop_table("knowledge_documents")
    postgresql.ENUM("pending", name="knowledge_document_status", create_type=False).drop(op.get_bind(), checkfirst=True)
