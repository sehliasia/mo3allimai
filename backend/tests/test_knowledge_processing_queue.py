from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.base import Base
from app.models.knowledge_document import (
    KnowledgeDocument,
    KnowledgeDocumentStatus,
    KnowledgeProcessingJob,
    KnowledgeProcessingJobStatus,
    KnowledgeProcessingJobType,
)
from app.api.routes.admin import list_knowledge_documents
from app.services.knowledge_processing_queue import KnowledgeProcessingQueue
from app.services.knowledge_ingestion_service import KnowledgeIngestionError


def session_for_queue(tmp_path: Path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'queue.db'}")
    Base.metadata.create_all(engine)
    return Session(engine)


def document(tmp_path: Path, number: int, *, preflight: str | None = None) -> KnowledgeDocument:
    source = tmp_path / f"{number}.pdf"
    source.write_bytes(b"%PDF-test")
    return KnowledgeDocument(
        title=f"Document {number}", original_filename=f"{number}.pdf", stored_filename=f"{number}.pdf",
        file_path=str(source), mime_type="application/pdf", file_size=9, uploaded_by=1,
        preflight_status=preflight,
    )


def test_enqueue_is_idempotent_and_processing_requires_preflight(tmp_path):
    db = session_for_queue(tmp_path)
    first, second, missing = document(tmp_path, 1), document(tmp_path, 2, preflight="complete"), document(tmp_path, 3)
    db.add_all([first, second, missing]); db.commit()
    queue = KnowledgeProcessingQueue()

    initial = queue.enqueue_preflight(db, [first.id, second.id, 999])
    repeated = queue.enqueue_preflight(db, [first.id])
    ingestion = queue.enqueue_ingestion(db, [second.id, missing.id])

    assert initial["queued"] == 2 and initial["skipped"] == 1
    assert repeated["queued"] == 0 and repeated["skipped_documents"][0]["reason"] == "active_job_exists"
    assert ingestion["queued"] == 1
    assert ingestion["skipped_documents"] == [{"document_id": missing.id, "reason": "current_preflight_required"}]


def test_worker_is_sequential_and_failure_does_not_stop_next_job(tmp_path):
    db = session_for_queue(tmp_path)
    first, second = document(tmp_path, 1), document(tmp_path, 2)
    db.add_all([first, second]); db.commit()
    preflight = Mock()
    preflight.analyze.side_effect = [RuntimeError("private error"), SimpleNamespace()]
    queue = KnowledgeProcessingQueue(preflight_service=preflight, ingestion_service=Mock())
    queue.enqueue_preflight(db, [first.id, second.id])

    failed = queue.process_next(db)
    completed = queue.process_next(db)

    assert failed.status == KnowledgeProcessingJobStatus.failed
    assert failed.error_message == "Processing could not be completed. Check server logs for details."
    assert completed.status == KnowledgeProcessingJobStatus.completed
    assert preflight.analyze.call_count == 2
    assert db.get(KnowledgeDocument, first.id).status == KnowledgeDocumentStatus.pending
    assert db.get(KnowledgeDocument, second.id).status == KnowledgeDocumentStatus.pending


def test_stale_processing_jobs_are_requeued_then_capped(tmp_path):
    db = session_for_queue(tmp_path)
    item = document(tmp_path, 1)
    db.add(item); db.commit()
    queue = KnowledgeProcessingQueue()
    queued = queue.enqueue_preflight(db, [item.id])["jobs"][0]
    job = db.get(KnowledgeProcessingJob, queued["id"])
    job.status = KnowledgeProcessingJobStatus.processing
    item.status = KnowledgeDocumentStatus.processing
    job.started_at = datetime.now(timezone.utc) - timedelta(days=1)
    db.commit()

    assert queue.recover_stale_jobs(db) == 1
    assert db.get(KnowledgeProcessingJob, job.id).status == KnowledgeProcessingJobStatus.pending
    assert db.get(KnowledgeDocument, item.id).status == KnowledgeDocumentStatus.pending
    job.status = KnowledgeProcessingJobStatus.processing
    item.status = KnowledgeDocumentStatus.processing
    job.attempts = queue.settings.knowledge_job_max_attempts
    job.started_at = datetime.now(timezone.utc) - timedelta(days=1)
    db.commit()
    queue.recover_stale_jobs(db)
    assert db.get(KnowledgeProcessingJob, job.id).status == KnowledgeProcessingJobStatus.failed
    assert db.get(KnowledgeDocument, item.id).status == KnowledgeDocumentStatus.pending


def test_one_processing_job_does_not_change_pending_documents_to_processing(tmp_path):
    db = session_for_queue(tmp_path)
    documents = [document(tmp_path, number, preflight="complete") for number in range(1, 4)]
    db.add_all(documents); db.commit()
    queue = KnowledgeProcessingQueue()
    queue.enqueue_ingestion(db, [item.id for item in documents])
    jobs = db.query(KnowledgeProcessingJob).order_by(KnowledgeProcessingJob.id).all()
    jobs[0].status = KnowledgeProcessingJobStatus.processing
    db.commit()

    assert [job.status for job in jobs] == [KnowledgeProcessingJobStatus.processing, KnowledgeProcessingJobStatus.pending, KnowledgeProcessingJobStatus.pending]
    assert [db.get(KnowledgeDocument, item.id).status for item in documents] == [KnowledgeDocumentStatus.pending] * 3


def test_failed_ingestion_persists_safe_ratio_summary(tmp_path):
    db = session_for_queue(tmp_path)
    item = document(tmp_path, 1, preflight="complete")
    db.add(item); db.commit()
    summary = {
        "quality_status": "failed", "pages_total": 31, "pages_quarantined_count": 7,
        "quarantined_page_numbers": [1, 2, 3, 4, 5, 6, 7], "failed_page_ratio": 7 / 31,
        "max_failed_page_ratio": 0.20, "failure_reason": "failed_page_ratio_exceeded",
        "chunks_valid": 12, "chunks_quarantined_count": 8, "ocr_failures_count": 2, "warnings_count": 7,
    }
    queue = KnowledgeProcessingQueue(ingestion_service=Mock())
    queue.ingestion_service.parse_preview.side_effect = KnowledgeIngestionError("failed", result_summary=summary)
    queue.enqueue_ingestion(db, [item.id])

    job = queue.process_next(db)

    assert job.status == KnowledgeProcessingJobStatus.failed
    assert job.result_summary == summary
    assert db.get(KnowledgeDocument, item.id).status == KnowledgeDocumentStatus.failed


def test_failed_ingestion_summary_is_exposed_in_admin_document_response(tmp_path):
    db = session_for_queue(tmp_path)
    item = document(tmp_path, 1)
    summary = {"quality_status": "failed", "pages_total": 31, "pages_quarantined_count": 7, "quarantined_page_numbers": [1, 2, 3, 4, 5, 6, 7], "failed_page_ratio": 7 / 31, "max_failed_page_ratio": 0.20, "failure_reason": "failed_page_ratio_exceeded", "chunks_valid": 12, "chunks_quarantined_count": 8, "ocr_failures_count": 2, "warnings_count": 7}
    db.add(item); db.commit()
    db.add(KnowledgeProcessingJob(document_id=item.id, job_type=KnowledgeProcessingJobType.ingestion, status=KnowledgeProcessingJobStatus.failed, stage="failed", result_summary=summary)); db.commit()

    response = list_knowledge_documents(None, db)

    assert response["items"][0]["ingestion_summary"] == summary
