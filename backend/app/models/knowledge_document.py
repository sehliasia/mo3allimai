from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SqlEnum, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class KnowledgeDocumentStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    partial = "partial"
    failed = "failed"


class KnowledgeProcessingJobType(str, Enum):
    preflight = "preflight"
    ingestion = "ingestion"


class KnowledgeProcessingJobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class KnowledgeChunkEmbeddingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    document_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    language: Mapped[str | None] = mapped_column(String(20), nullable=True)
    cefr_level: Mapped[str | None] = mapped_column(String(10), nullable=True)
    skill: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[KnowledgeDocumentStatus] = mapped_column(SqlEnum(KnowledgeDocumentStatus, name="knowledge_document_status"), nullable=False, default=KnowledgeDocumentStatus.pending)
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    preflight_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    preflight_analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    preflight_source_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preflight_analysis_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preflight_pages_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preflight_pages_analyzed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preflight_analysis_failed_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preflight_native_good_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preflight_native_borderline_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preflight_native_bad_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preflight_ocr_candidate_page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preflight_ocr_required_page_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    preflight_recommended_strategy: Mapped[str | None] = mapped_column(String(50), nullable=True)
    preflight_estimated_complexity: Mapped[str | None] = mapped_column(String(20), nullable=True)
    preflight_page_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    processing_jobs: Mapped[list["KnowledgeProcessingJob"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[list["KnowledgeChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class KnowledgeChunk(Base):
    """Validated, versioned chunk materialization; embeddings remain external."""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_knowledge_chunks_document_index"),
        Index("ix_knowledge_chunks_document_page", "document_id", "source_page_start"),
        Index("ix_knowledge_chunks_document_hash", "document_id", "chunk_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_for_embedding: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extraction_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quality_status: Mapped[str] = mapped_column(String(20), nullable=False)
    heading_context: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ingestion_version: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_status: Mapped[KnowledgeChunkEmbeddingStatus] = mapped_column(
        SqlEnum(KnowledgeChunkEmbeddingStatus, name="knowledge_chunk_embedding_status"),
        nullable=False,
        default=KnowledgeChunkEmbeddingStatus.pending,
        index=True,
    )
    vector_point_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    embedding_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")


class KnowledgeProcessingJob(Base):
    __tablename__ = "knowledge_processing_jobs"
    __table_args__ = (
        Index(
            "uq_knowledge_processing_jobs_active_type",
            "document_id",
            "job_type",
            unique=True,
            postgresql_where=text("status IN ('pending', 'processing')"),
            sqlite_where=text("status IN ('pending', 'processing')"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    job_type: Mapped[KnowledgeProcessingJobType] = mapped_column(SqlEnum(KnowledgeProcessingJobType, name="knowledge_processing_job_type"), nullable=False)
    status: Mapped[KnowledgeProcessingJobStatus] = mapped_column(SqlEnum(KnowledgeProcessingJobStatus, name="knowledge_processing_job_status"), nullable=False, default=KnowledgeProcessingJobStatus.pending, index=True)
    stage: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    document: Mapped[KnowledgeDocument] = relationship(back_populates="processing_jobs")
