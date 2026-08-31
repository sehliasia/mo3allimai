"""persist compact knowledge ingestion results"""

from alembic import op
import sqlalchemy as sa


revision = "20260819_0009"
down_revision = "20260818_0008"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("knowledge_processing_jobs", sa.Column("result_summary", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("knowledge_processing_jobs", "result_summary")
