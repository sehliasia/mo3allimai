"""add failed knowledge document status"""

from alembic import op


revision = "20260812_0005"
down_revision = "20260812_0004"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TYPE knowledge_document_status ADD VALUE IF NOT EXISTS 'failed'")


def downgrade():
    # PostgreSQL does not support removing an enum value safely in place.
    raise RuntimeError("Downgrade is not supported: knowledge_document_status contains the 'failed' value.")
