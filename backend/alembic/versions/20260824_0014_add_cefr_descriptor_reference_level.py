"""add structured CEFR descriptor reference level"""

from alembic import op
import sqlalchemy as sa


revision = "20260824_0014"
down_revision = "20260824_0013"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("cefr_descriptors", sa.Column("reference_level_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_cefr_descriptors_reference_level",
        "cefr_descriptors",
        "cefr_levels",
        ["reference_level_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade():
    op.drop_constraint("fk_cefr_descriptors_reference_level", "cefr_descriptors", type_="foreignkey")
    op.drop_column("cefr_descriptors", "reference_level_id")
