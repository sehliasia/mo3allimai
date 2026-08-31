"""create persistent knowledge processing jobs"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260818_0008"
down_revision = "20260817_0007"
branch_labels = None
depends_on = None


# These are deliberately PostgreSQL ENUMs.  They are explicitly created in
# upgrade() and create_type=False prevents create_table() from issuing a second
# CREATE TYPE for an enum that already exists after an interrupted migration.
job_type_enum = postgresql.ENUM(
    "preflight",
    "ingestion",
    name="knowledge_processing_job_type",
    create_type=False,
)
job_status_enum = postgresql.ENUM(
    "pending",
    "processing",
    "completed",
    "failed",
    name="knowledge_processing_job_status",
    create_type=False,
)


def upgrade():
    # PostgreSQL enum values cannot be removed safely on downgrade.  Extend the
    # existing document lifecycle before the worker can write these states.
    op.execute("ALTER TYPE knowledge_document_status ADD VALUE IF NOT EXISTS 'processing'")
    op.execute("ALTER TYPE knowledge_document_status ADD VALUE IF NOT EXISTS 'ready'")
    op.execute("ALTER TYPE knowledge_document_status ADD VALUE IF NOT EXISTS 'partial'")
    bind = op.get_bind()
    job_type_enum.create(bind, checkfirst=True)
    job_status_enum.create(bind, checkfirst=True)
    op.create_table(
        "knowledge_processing_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_type", job_type_enum, nullable=False),
        sa.Column("status", job_status_enum, nullable=False, server_default="pending"),
        sa.Column("stage", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_knowledge_processing_jobs_document_id", "knowledge_processing_jobs", ["document_id"])
    op.create_index("ix_knowledge_processing_jobs_status", "knowledge_processing_jobs", ["status"])
    op.execute("CREATE UNIQUE INDEX uq_knowledge_processing_jobs_active_type ON knowledge_processing_jobs (document_id, job_type) WHERE status IN ('pending', 'processing')")


def downgrade():
    op.drop_index("uq_knowledge_processing_jobs_active_type", table_name="knowledge_processing_jobs")
    op.drop_index("ix_knowledge_processing_jobs_status", table_name="knowledge_processing_jobs")
    op.drop_index("ix_knowledge_processing_jobs_document_id", table_name="knowledge_processing_jobs")
    op.drop_table("knowledge_processing_jobs")
    bind = op.get_bind()
    job_status_enum.drop(bind, checkfirst=True)
    job_type_enum.drop(bind, checkfirst=True)
