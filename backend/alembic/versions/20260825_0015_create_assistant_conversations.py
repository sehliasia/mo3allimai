"""create assistant conversation persistence tables"""

from alembic import op
import sqlalchemy as sa


revision = "20260825_0015"
down_revision = "20260824_0014"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "assistant_conversations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_assistant_conversations_user_id", "assistant_conversations", ["user_id"])
    op.create_index("ix_assistant_conversations_updated_at", "assistant_conversations", ["updated_at"])
    op.create_index("ix_assistant_conversations_user_updated", "assistant_conversations", ["user_id", "updated_at"])

    op.create_table(
        "assistant_messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.Integer(), sa.ForeignKey("assistant_conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("role IN ('USER', 'ASSISTANT')", name="ck_assistant_messages_role"),
    )
    op.create_index("ix_assistant_messages_conversation_id", "assistant_messages", ["conversation_id"])
    op.create_index("ix_assistant_messages_created_at", "assistant_messages", ["created_at"])
    op.create_index("ix_assistant_messages_conversation_created", "assistant_messages", ["conversation_id", "created_at"])

    op.create_table(
        "assistant_message_sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("message_id", sa.Integer(), sa.ForeignKey("assistant_messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("knowledge_documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("document_title", sa.String(length=255), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("descriptor_scale", sa.String(length=255), nullable=True),
        sa.Column("source_order", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("source_type IN ('cefr_structured', 'pedagogical_resource')", name="ck_assistant_message_sources_type"),
        sa.UniqueConstraint("message_id", "source_order", name="uq_assistant_message_sources_order"),
    )
    op.create_index("ix_assistant_message_sources_message_id", "assistant_message_sources", ["message_id"])


def downgrade():
    op.drop_index("ix_assistant_message_sources_message_id", table_name="assistant_message_sources")
    op.drop_table("assistant_message_sources")
    op.drop_index("ix_assistant_messages_conversation_created", table_name="assistant_messages")
    op.drop_index("ix_assistant_messages_created_at", table_name="assistant_messages")
    op.drop_index("ix_assistant_messages_conversation_id", table_name="assistant_messages")
    op.drop_table("assistant_messages")
    op.drop_index("ix_assistant_conversations_user_updated", table_name="assistant_conversations")
    op.drop_index("ix_assistant_conversations_updated_at", table_name="assistant_conversations")
    op.drop_index("ix_assistant_conversations_user_id", table_name="assistant_conversations")
    op.drop_table("assistant_conversations")
