"""Persistent, single-worker queue for Knowledge Base jobs.

The queue schedules existing services; it deliberately contains no Docling,
OCR, or chunking implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.knowledge_document import (
    KnowledgeDocument,
    KnowledgeDocumentStatus,
    KnowledgeProcessingJob,
    KnowledgeProcessingJobStatus,
    KnowledgeProcessingJobType,
)
from app.services.knowledge_ingestion_service import KnowledgeIngestionError, KnowledgeIngestionService
from app.services.knowledge_preflight_service import KnowledgePreflightService

logger = logging.getLogger(__name__)
ACTIVE_STATUSES = (KnowledgeProcessingJobStatus.pending, KnowledgeProcessingJobStatus.processing)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def job_response(job: KnowledgeProcessingJob) -> dict[str, object]:
    return {
        "id": job.id,
        "document_id": job.document_id,
        "job_type": job.job_type.value,
        "status": job.status.value,
        "stage": job.stage,
        "attempts": job.attempts,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message,
        "result_summary": job.result_summary,
    }


class KnowledgeProcessingQueue:
    def __init__(self, *, preflight_service: KnowledgePreflightService | None = None, ingestion_service: KnowledgeIngestionService | None = None):
        self.settings = get_settings()
        self.preflight_service = preflight_service or KnowledgePreflightService()
        self.ingestion_service = ingestion_service or KnowledgeIngestionService()

    @staticmethod
    def _active_job(db: Session, document_id: int, job_type: KnowledgeProcessingJobType) -> KnowledgeProcessingJob | None:
        return db.scalar(select(KnowledgeProcessingJob).where(
            KnowledgeProcessingJob.document_id == document_id,
            KnowledgeProcessingJob.job_type == job_type,
            KnowledgeProcessingJob.status.in_(ACTIVE_STATUSES),
        ))

    def enqueue(self, db: Session, document: KnowledgeDocument, job_type: KnowledgeProcessingJobType) -> tuple[KnowledgeProcessingJob | None, str | None]:
        existing = self._active_job(db, document.id, job_type)
        if existing:
            return existing, "active_job_exists"
        job = KnowledgeProcessingJob(document_id=document.id, job_type=job_type, status=KnowledgeProcessingJobStatus.pending, stage="pending")
        db.add(job)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = self._active_job(db, document.id, job_type)
            return existing, "active_job_exists"
        db.refresh(job)
        return job, None

    def enqueue_preflight(self, db: Session, document_ids: list[int]) -> dict[str, object]:
        return self._enqueue_many(db, document_ids, KnowledgeProcessingJobType.preflight)

    def enqueue_ingestion(self, db: Session, document_ids: list[int]) -> dict[str, object]:
        return self._enqueue_many(db, document_ids, KnowledgeProcessingJobType.ingestion)

    def _enqueue_many(self, db: Session, document_ids: list[int], job_type: KnowledgeProcessingJobType) -> dict[str, object]:
        jobs: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []
        for document_id in dict.fromkeys(document_ids):
            document = db.get(KnowledgeDocument, document_id)
            if document is None:
                skipped.append({"document_id": document_id, "reason": "document_not_found"})
                continue
            if job_type is KnowledgeProcessingJobType.ingestion:
                reason = self._ingestion_ineligible_reason(document)
                if reason:
                    skipped.append({"document_id": document_id, "reason": reason})
                    continue
            job, reason = self.enqueue(db, document, job_type)
            if reason:
                skipped.append({"document_id": document_id, "reason": reason})
            elif job:
                jobs.append(job_response(job))
        return {"queued": len(jobs), "skipped": len(skipped), "jobs": jobs, "skipped_documents": skipped}

    @staticmethod
    def _ingestion_ineligible_reason(document: KnowledgeDocument) -> str | None:
        if document.preflight_status not in {"complete", "partial"}:
            return "current_preflight_required"
        if not document.file_path or not Path(document.file_path).is_file():
            return "source_file_unavailable"
        return None

    def recover_stale_jobs(self, db: Session) -> int:
        threshold = _utcnow() - timedelta(minutes=self.settings.knowledge_job_stale_minutes)
        stale = db.scalars(select(KnowledgeProcessingJob).where(
            KnowledgeProcessingJob.status == KnowledgeProcessingJobStatus.processing,
            KnowledgeProcessingJob.started_at < threshold,
        )).all()
        for job in stale:
            document = job.document
            if job.attempts >= self.settings.knowledge_job_max_attempts:
                job.status = KnowledgeProcessingJobStatus.failed
                job.stage = "failed"
                job.completed_at = _utcnow()
                job.error_message = "The job exceeded the retry limit after worker recovery."
                if document and job.job_type is KnowledgeProcessingJobType.ingestion:
                    document.status = KnowledgeDocumentStatus.failed
                elif document and document.status is KnowledgeDocumentStatus.processing:
                    document.status = KnowledgeDocumentStatus.pending
            else:
                job.status = KnowledgeProcessingJobStatus.pending
                job.stage = "pending"
                job.started_at = None
                job.error_message = None
                # Queue state is the source of truth while a job is active.
                # Clear the legacy transient document state left by a crash.
                if document and document.status is KnowledgeDocumentStatus.processing:
                    document.status = KnowledgeDocumentStatus.pending
        if stale:
            db.commit()
        return len(stale)

    def process_next(self, db: Session) -> KnowledgeProcessingJob | None:
        job = db.scalar(select(KnowledgeProcessingJob).where(
            KnowledgeProcessingJob.status == KnowledgeProcessingJobStatus.pending,
        ).order_by(KnowledgeProcessingJob.created_at, KnowledgeProcessingJob.id).with_for_update(skip_locked=True).limit(1))
        if job is None:
            return None
        job.status = KnowledgeProcessingJobStatus.processing
        job.stage = "starting"
        job.attempts += 1
        job.started_at = _utcnow()
        job.error_message = None
        document = job.document
        db.commit()
        try:
            if job.job_type is KnowledgeProcessingJobType.preflight:
                job.stage = "native_analysis"
                db.commit()
                self.preflight_service.analyze(db, document)
                if document.status is KnowledgeDocumentStatus.processing:
                    document.status = KnowledgeDocumentStatus.pending
            else:
                job.stage = "extracting"
                db.commit()
                preview = self.ingestion_service.parse_preview(db, document)
                job.result_summary = self._ingestion_summary(preview)
                document.status = KnowledgeDocumentStatus.partial if preview.quality_status == "partial" else KnowledgeDocumentStatus.ready
            job.status = KnowledgeProcessingJobStatus.completed
            job.stage = "completed"
            job.completed_at = _utcnow()
            db.commit()
        except Exception as error:
            db.rollback()
            job = db.get(KnowledgeProcessingJob, job.id)
            document = db.get(KnowledgeDocument, job.document_id) if job else None
            if job:
                job.status = KnowledgeProcessingJobStatus.failed
                job.stage = "failed"
                job.completed_at = _utcnow()
                job.error_message = (
                    "Model resource unavailable. Please retry when the Docling model resource is available."
                    if isinstance(error, KnowledgeIngestionError) and str(error).startswith("Model resource unavailable.")
                    else "Processing could not be completed. Check server logs for details."
                )
                if job.job_type is KnowledgeProcessingJobType.ingestion and isinstance(error, KnowledgeIngestionError):
                    job.result_summary = error.result_summary
            if document:
                if job.job_type is KnowledgeProcessingJobType.ingestion:
                    document.status = KnowledgeDocumentStatus.failed
                elif document.status is KnowledgeDocumentStatus.processing:
                    document.status = KnowledgeDocumentStatus.pending
            db.commit()
            logger.exception("knowledge_processing_job_failed job_id=%s document_id=%s", job.id if job else None, document.id if document else None)
        return job

    @staticmethod
    def _ingestion_summary(preview: object) -> dict[str, object]:
        stats = preview.statistics
        pages = preview.page_extractions
        return {
            "quality_status": preview.quality_status,
            "pages_total": preview.pages,
            "pages_quarantined_count": stats.get("pages_quarantined", 0),
            "quarantined_page_numbers": stats.get("quarantined_page_numbers", []),
            "chunks_valid": preview.chunks_count,
            "chunks_quarantined_count": stats.get("chunks_quarantined", 0),
            "tables_count": stats.get("table_chunks_count", 0),
            "pictures_count": sum(len(page.images) for page in pages),
            "ocr_pages_count": sum(page.extraction_mode == "full_page_ocr" for page in pages),
            "ocr_failures_count": stats.get("ocr_failures", 0),
            "late_repairs_attempted": stats.get("late_repairs_attempted", 0),
            "late_repairs_accepted": stats.get("late_repairs_accepted", 0),
            "late_repairs_rejected": stats.get("late_repairs_rejected", 0),
            "warnings_count": stats.get("warnings_count", 0),
            "completed_at": _utcnow().isoformat(),
        }
