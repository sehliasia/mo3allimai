"""make knowledge document metadata optional"""

from alembic import op
import sqlalchemy as sa

revision = "20260812_0004"
down_revision = "20260812_0003"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("knowledge_documents", "document_type", existing_type=sa.String(length=50), nullable=True)
    op.alter_column("knowledge_documents", "language", existing_type=sa.String(length=20), nullable=True)
    op.alter_column("knowledge_documents", "storage_path", new_column_name="file_path")
    op.add_column("knowledge_documents", sa.Column("mime_type", sa.String(length=100), nullable=False, server_default=sa.text("'application/pdf'")))
    op.alter_column("knowledge_documents", "mime_type", server_default=None)


def downgrade():
    bind = op.get_bind()
    missing_metadata = bind.execute(sa.text("SELECT 1 FROM knowledge_documents WHERE document_type IS NULL OR language IS NULL LIMIT 1")).first()
    if missing_metadata:
        raise RuntimeError("Cannot restore NOT NULL metadata constraints while knowledge documents have no metadata.")
    op.drop_column("knowledge_documents", "mime_type")
    op.alter_column("knowledge_documents", "file_path", new_column_name="storage_path")
    op.alter_column("knowledge_documents", "language", existing_type=sa.String(length=20), nullable=False)
    op.alter_column("knowledge_documents", "document_type", existing_type=sa.String(length=50), nullable=False)
