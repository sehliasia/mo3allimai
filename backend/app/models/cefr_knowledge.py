"""Authoritative, deterministic CEFR structures sourced from canonical chunks."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class CEFRLevel(Base):
    __tablename__ = "cefr_levels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    is_core_reference_level: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    descriptors: Mapped[list["CEFRDescriptor"]] = relationship(back_populates="level", foreign_keys="CEFRDescriptor.level_id")


class CEFRScale(Base):
    __tablename__ = "cefr_scales"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    descriptors: Mapped[list["CEFRDescriptor"]] = relationship(back_populates="scale")


class CEFRDescriptor(Base):
    __tablename__ = "cefr_descriptors"
    __table_args__ = (
        CheckConstraint("status IN ('AVAILABLE', 'NO_DESCRIPTOR_AVAILABLE')", name="ck_cefr_descriptor_status"),
        UniqueConstraint("level_id", "scale_id", "descriptor_hash", name="uq_cefr_descriptor_identity"),
        Index("ix_cefr_descriptors_level_scale", "level_id", "scale_id"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    level_id: Mapped[int] = mapped_column(ForeignKey("cefr_levels.id", ondelete="RESTRICT"), nullable=False)
    reference_level_id: Mapped[int | None] = mapped_column(ForeignKey("cefr_levels.id", ondelete="RESTRICT"), nullable=True)
    scale_id: Mapped[int] = mapped_column(ForeignKey("cefr_scales.id", ondelete="RESTRICT"), nullable=False)
    descriptor_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    descriptor_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    level: Mapped[CEFRLevel] = relationship(back_populates="descriptors", foreign_keys=[level_id])
    reference_level: Mapped[CEFRLevel | None] = relationship(foreign_keys=[reference_level_id])
    scale: Mapped[CEFRScale] = relationship(back_populates="descriptors")
    sources: Mapped[list["CEFRDescriptorSource"]] = relationship(back_populates="descriptor", cascade="all, delete-orphan")


class CEFRDescriptorSource(Base):
    __tablename__ = "cefr_descriptor_sources"
    __table_args__ = (
        UniqueConstraint("descriptor_id", "chunk_id", name="uq_cefr_descriptor_source_chunk"),
        Index("ix_cefr_descriptor_sources_document", "document_id"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    descriptor_id: Mapped[int] = mapped_column(ForeignKey("cefr_descriptors.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[int] = mapped_column(ForeignKey("knowledge_documents.id", ondelete="RESTRICT"), nullable=False)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("knowledge_chunks.id", ondelete="RESTRICT"), nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    descriptor: Mapped[CEFRDescriptor] = relationship(back_populates="sources")
