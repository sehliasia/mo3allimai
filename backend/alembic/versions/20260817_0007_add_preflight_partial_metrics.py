"""add partial preflight metrics"""

from alembic import op
import sqlalchemy as sa


revision = "20260817_0007"
down_revision = "20260817_0006"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("knowledge_documents", sa.Column("preflight_pages_analyzed", sa.Integer(), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_analysis_failed_pages", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("knowledge_documents", "preflight_analysis_failed_pages")
    op.drop_column("knowledge_documents", "preflight_pages_analyzed")
