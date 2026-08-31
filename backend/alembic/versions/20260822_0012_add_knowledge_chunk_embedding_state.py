"""add KnowledgeChunk embedding synchronization state"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260822_0012"
down_revision = "20260821_0011"
branch_labels = None
depends_on = None


embedding_status_enum = postgresql.ENUM(
    "pending", "processing", "indexed", "failed",
    name="knowledge_chunk_embedding_status",
    create_type=False,
)
embedding_status_enum_create = postgresql.ENUM(
    "pending", "processing", "indexed", "failed",
    name="knowledge_chunk_embedding_status",
)


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        embedding_status_enum_create.create(bind, checkfirst=True)
    status_type = embedding_status_enum if bind.dialect.name == "postgresql" else sa.String(length=20)
    op.add_column("knowledge_chunks", sa.Column("embedding_input_hash", sa.String(length=64), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("embedding_model", sa.String(length=255), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("embedding_status", status_type, nullable=False, server_default="pending"))
    op.add_column("knowledge_chunks", sa.Column("vector_point_id", sa.String(length=64), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("embedding_error", sa.Text(), nullable=True))
    op.create_index("ix_knowledge_chunks_embedding_status", "knowledge_chunks", ["embedding_status"])
    op.create_index("ix_knowledge_chunks_embedding_input_hash", "knowledge_chunks", ["embedding_input_hash"])
    op.create_unique_constraint("uq_knowledge_chunks_vector_point_id", "knowledge_chunks", ["vector_point_id"])


def downgrade():
    bind = op.get_bind()
    op.drop_constraint("uq_knowledge_chunks_vector_point_id", "knowledge_chunks", type_="unique")
    op.drop_index("ix_knowledge_chunks_embedding_input_hash", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_embedding_status", table_name="knowledge_chunks")
    op.drop_column("knowledge_chunks", "embedding_error")
    op.drop_column("knowledge_chunks", "embedded_at")
    op.drop_column("knowledge_chunks", "vector_point_id")
    op.drop_column("knowledge_chunks", "embedding_status")
    op.drop_column("knowledge_chunks", "embedding_model")
    op.drop_column("knowledge_chunks", "embedding_input_hash")
    if bind.dialect.name == "postgresql":
        embedding_status_enum.drop(bind, checkfirst=True)
