"""persist validated knowledge chunks"""

from alembic import op
import sqlalchemy as sa

revision = "20260821_0011"
down_revision = "20260820_0010"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_for_embedding", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(length=32), nullable=False),
        sa.Column("source_page_start", sa.Integer(), nullable=True),
        sa.Column("source_page_end", sa.Integer(), nullable=True),
        sa.Column("extraction_mode", sa.String(length=32), nullable=True),
        sa.Column("quality_status", sa.String(length=20), nullable=False),
        sa.Column("heading_context", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("chunk_hash", sa.String(length=64), nullable=False),
        sa.Column("ingestion_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_knowledge_chunks_document_index"),
    )
    op.create_index("ix_knowledge_chunks_document_id", "knowledge_chunks", ["document_id"])
    op.create_index("ix_knowledge_chunks_document_page", "knowledge_chunks", ["document_id", "source_page_start"])
    op.create_index("ix_knowledge_chunks_document_hash", "knowledge_chunks", ["document_id", "chunk_hash"])


def downgrade():
    op.drop_table("knowledge_chunks")
