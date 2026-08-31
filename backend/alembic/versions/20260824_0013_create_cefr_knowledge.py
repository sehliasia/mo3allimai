"""create structured CEFR knowledge tables"""

from alembic import op
import sqlalchemy as sa

revision = "20260824_0013"
down_revision = "20260822_0012"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("cefr_levels", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("code", sa.String(16), nullable=False, unique=True), sa.Column("label", sa.String(64), nullable=False), sa.Column("sort_order", sa.Integer(), nullable=False, unique=True), sa.Column("is_core_reference_level", sa.Boolean(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_table("cefr_scales", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("name", sa.String(255), nullable=False), sa.Column("normalized_name", sa.String(255), nullable=False, unique=True), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_table("cefr_descriptors", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("level_id", sa.Integer(), sa.ForeignKey("cefr_levels.id", ondelete="RESTRICT"), nullable=False), sa.Column("scale_id", sa.Integer(), sa.ForeignKey("cefr_scales.id", ondelete="RESTRICT"), nullable=False), sa.Column("descriptor_text", sa.Text()), sa.Column("normalized_text", sa.Text(), nullable=False), sa.Column("descriptor_hash", sa.String(64), nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.CheckConstraint("status IN ('AVAILABLE', 'NO_DESCRIPTOR_AVAILABLE')", name="ck_cefr_descriptor_status"), sa.UniqueConstraint("level_id", "scale_id", "descriptor_hash", name="uq_cefr_descriptor_identity"))
    op.create_index("ix_cefr_descriptors_level_scale", "cefr_descriptors", ["level_id", "scale_id"])
    op.create_table("cefr_descriptor_sources", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("descriptor_id", sa.Integer(), sa.ForeignKey("cefr_descriptors.id", ondelete="CASCADE"), nullable=False), sa.Column("document_id", sa.Integer(), sa.ForeignKey("knowledge_documents.id", ondelete="RESTRICT"), nullable=False), sa.Column("chunk_id", sa.Integer(), sa.ForeignKey("knowledge_chunks.id", ondelete="RESTRICT"), nullable=False), sa.Column("page_start", sa.Integer()), sa.Column("page_end", sa.Integer()), sa.Column("source_order", sa.Integer(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.UniqueConstraint("descriptor_id", "chunk_id", name="uq_cefr_descriptor_source_chunk"))
    op.create_index("ix_cefr_descriptor_sources_document", "cefr_descriptor_sources", ["document_id"])


def downgrade():
    op.drop_table("cefr_descriptor_sources")
    op.drop_index("ix_cefr_descriptors_level_scale", table_name="cefr_descriptors")
    op.drop_table("cefr_descriptors")
    op.drop_table("cefr_scales")
    op.drop_table("cefr_levels")
