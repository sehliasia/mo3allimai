"""Persistent, teacher-owned conversations for the pedagogical assistant."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class AssistantMessageRole(str, Enum):
    user = "USER"
    assistant = "ASSISTANT"


class AssistantConversation(Base):
    __tablename__ = "assistant_conversations"
    __table_args__ = (
        Index("ix_assistant_conversations_user_updated", "user_id", "updated_at"),
        Index("ix_assistant_conversations_user_archived_updated", "user_id", "archived_at", "updated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, index=True,
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)

    user: Mapped["User"] = relationship(back_populates="assistant_conversations")
    messages: Mapped[list["AssistantMessage"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True,
    )


class AssistantMessage(Base):
    __tablename__ = "assistant_messages"
    __table_args__ = (
        CheckConstraint("role IN ('USER', 'ASSISTANT')", name="ck_assistant_messages_role"),
        Index("ix_assistant_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("assistant_conversations.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    role: Mapped[AssistantMessageRole] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    conversation: Mapped[AssistantConversation] = relationship(back_populates="messages")
    sources: Mapped[list["AssistantMessageSource"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", passive_deletes=True,
    )


class AssistantMessageSource(Base):
    __tablename__ = "assistant_message_sources"
    __table_args__ = (
        CheckConstraint(
            "source_type IN ('cefr_structured', 'pedagogical_resource')",
            name="ck_assistant_message_sources_type",
        ),
        UniqueConstraint("message_id", "source_order", name="uq_assistant_message_sources_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("assistant_messages.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Keep historical citations when a global document is removed; never cascade from it.
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="SET NULL"), nullable=True,
    )
    document_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    descriptor_scale: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    message: Mapped[AssistantMessage] = relationship(back_populates="sources")
