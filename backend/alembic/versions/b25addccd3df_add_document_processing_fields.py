"""synchronise document processing fields after an interrupted rollout

Revision ID: b25addccd3df
Revises: 20260830_0017
Create Date: 2026-08-30 17:31:43.757985

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "b25addccd3df"
down_revision = "20260830_0017"
branch_labels = None
depends_on = None


def _existing_columns() -> dict[str, dict]:
    """Return the live column definitions, not assumptions from the ORM.

    The database was left at revision 0017 after a prior rollout had already
    introduced ``processing_error``.  Inspecting PostgreSQL makes this repair
    safe to run on either version of that partially-upgraded schema.
    """
    return {
        column["name"]: column
        for column in inspect(op.get_bind()).get_columns("teacher_library_documents")
    }


def upgrade() -> None:
    existing = _existing_columns()

    # A VARCHAR(1000) conversion is safe only if every existing value fits.
    # Fail before changing anything rather than truncate an error message.
    processing_error = existing.get("processing_error")
    if processing_error is not None and getattr(processing_error["type"], "length", None) != 1000:
        oversized_count = op.get_bind().execute(
            sa.text(
                "SELECT count(*) FROM teacher_library_documents "
                "WHERE char_length(processing_error) > 1000"
            )
        ).scalar_one()
        if oversized_count:
            raise RuntimeError(
                "Cannot convert teacher_library_documents.processing_error to "
                "VARCHAR(1000): existing values exceed 1000 characters. "
                "No schema changes were applied."
            )

    if "processing_stage" not in existing:
        op.add_column("teacher_library_documents", sa.Column("processing_stage", sa.String(length=32), nullable=True))
    if processing_error is None:
        op.add_column("teacher_library_documents", sa.Column("processing_error", sa.String(length=1000), nullable=True))
    elif getattr(processing_error["type"], "length", None) != 1000:
        # The pre-existing TEXT column is valid data, but differs from the
        # SQLAlchemy contract.  The preflight above guarantees this cast is
        # lossless on PostgreSQL.
        op.alter_column(
            "teacher_library_documents",
            "processing_error",
            existing_type=processing_error["type"],
            type_=sa.String(length=1000),
            existing_nullable=processing_error["nullable"],
            postgresql_using="processing_error::varchar(1000)",
        )


def downgrade() -> None:
    # This repair migration may adopt a pre-existing column.  A downgrade must
    # never remove a column whose provenance Alembic cannot prove.
    pass
