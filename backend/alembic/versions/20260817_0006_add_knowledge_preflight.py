"""add persisted knowledge document preflight metadata"""

from alembic import op
import sqlalchemy as sa


revision = "20260817_0006"
down_revision = "20260812_0005"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("knowledge_documents", sa.Column("preflight_status", sa.String(length=20), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_analyzed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_source_sha256", sa.String(length=64), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_analysis_version", sa.String(length=64), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_pages_total", sa.Integer(), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_native_good_pages", sa.Integer(), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_native_borderline_pages", sa.Integer(), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_native_bad_pages", sa.Integer(), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_ocr_candidate_page_count", sa.Integer(), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_ocr_required_page_ratio", sa.Float(), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_recommended_strategy", sa.String(length=50), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_estimated_complexity", sa.String(length=20), nullable=True))
    op.add_column("knowledge_documents", sa.Column("preflight_page_details", sa.JSON(), nullable=True))


def downgrade():
    for column in (
        "preflight_page_details", "preflight_estimated_complexity", "preflight_recommended_strategy",
        "preflight_ocr_required_page_ratio", "preflight_ocr_candidate_page_count", "preflight_native_bad_pages",
        "preflight_native_borderline_pages", "preflight_native_good_pages", "preflight_pages_total",
        "preflight_analysis_version", "preflight_source_sha256", "preflight_analyzed_at", "preflight_status",
    ):
        op.drop_column("knowledge_documents", column)
