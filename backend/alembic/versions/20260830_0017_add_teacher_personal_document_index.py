"""add private teacher document chunks and indexing state"""
from alembic import op
import sqlalchemy as sa

revision = "20260830_0017"
down_revision = "20260828_0016"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("teacher_library_documents", sa.Column("status", sa.String(24), nullable=False, server_default="pending"))
    op.create_index("ix_teacher_library_documents_status", "teacher_library_documents", ["status"])
    op.create_table("teacher_library_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("teacher_library_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False), sa.Column("content", sa.Text(), nullable=False), sa.Column("content_for_embedding", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False), sa.Column("page_number", sa.Integer()), sa.Column("metadata", sa.JSON(), nullable=False), sa.Column("chunk_hash", sa.String(64), nullable=False),
        sa.Column("vector_point_id", sa.String(64), unique=True), sa.Column("embedding_model", sa.String(255)), sa.Column("embedding_status", sa.String(24), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_teacher_library_chunks_owner_document", "teacher_library_chunks", ["owner_id", "document_id"])

def downgrade():
    op.drop_table("teacher_library_chunks")
    op.drop_index("ix_teacher_library_documents_status", table_name="teacher_library_documents")
    op.drop_column("teacher_library_documents", "status")
