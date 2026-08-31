from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from uuid import uuid4
import json
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status as http_status
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from app.api.dependencies import require_admin
from app.database.session import get_db
from app.models.user import User, UserRole
from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument, KnowledgeDocumentStatus, KnowledgeProcessingJob, KnowledgeProcessingJobStatus, KnowledgeProcessingJobType
from app.schemas.user import UserRead
from app.services.knowledge_ingestion_service import KnowledgeIngestionError, KnowledgeIngestionService
from app.services.knowledge_preflight_service import KnowledgePreflightService
from app.services.knowledge_processing_queue import KnowledgeProcessingQueue, job_response
from app.services.knowledge_ingestion_service import DEBUG_CHUNKS_DIRECTORY

router = APIRouter(prefix="/admin", tags=["admin"])
UPLOAD_DIRECTORY = Path(__file__).resolve().parents[3] / "uploads" / "knowledge-base"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class KnowledgeDocumentSelection(BaseModel):
    document_ids: list[int] = Field(min_length=1, max_length=100)


def title_from_filename(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ")
    return " ".join(stem.split()) or "document"


def stored_pdf_is_valid(path: Path) -> bool:
    """Validate the private copy, never the client stream or a client-supplied path."""
    if not path.is_file():
        return False
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf) > 0
        finally:
            pdf.close()
    except Exception:
        return False


def teacher_or_404(db: Session, teacher_id: int) -> User:
    teacher = db.get(User, teacher_id)
    if not teacher or teacher.role != UserRole.teacher or teacher.is_deleted: raise HTTPException(404, "Teacher not found")
    return teacher
@router.get("/teachers")
def teachers(search: str | None = None, is_active: bool | None = None, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100), sort_order: str = Query("desc", pattern="^(asc|desc)$"), _: User = Depends(require_admin), db: Session = Depends(get_db)):
    filters = [User.role == UserRole.teacher, User.is_deleted.is_(False)]
    if search: filters.append(or_(User.full_name.ilike(f"%{search.strip()}%"), User.email.ilike(f"%{search.strip()}%")))
    if is_active is not None: filters.append(User.is_active == is_active)
    total = db.scalar(select(func.count()).select_from(User).where(*filters)) or 0
    order = User.created_at.asc() if sort_order == "asc" else User.created_at.desc()
    items = db.scalars(select(User).where(*filters).order_by(order).offset((page - 1) * page_size).limit(page_size)).all()
    return {"items": [UserRead.model_validate(item).model_dump() for item in items], "page": page, "page_size": page_size, "total": total, "total_pages": ceil(total / page_size) if total else 0}
@router.get("/teachers/{teacher_id}", response_model=UserRead)
def teacher(teacher_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)): return teacher_or_404(db, teacher_id)
@router.patch("/teachers/{teacher_id}/status", response_model=UserRead)
def status(teacher_id: int, data: dict, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    teacher = teacher_or_404(db, teacher_id)
    if teacher.id == admin.id: raise HTTPException(403, "Cannot modify own account")
    if "is_active" not in data or not isinstance(data["is_active"], bool): raise HTTPException(422, "is_active is required")
    teacher.is_active = data["is_active"]; db.commit(); db.refresh(teacher); return teacher
@router.delete("/teachers/{teacher_id}", response_model=UserRead)
def delete_teacher(teacher_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    teacher = teacher_or_404(db, teacher_id)
    if teacher.id == admin.id: raise HTTPException(403, "Cannot delete own account")
    teacher.is_active = False; teacher.is_deleted = True; teacher.deleted_at = datetime.now(timezone.utc); teacher.deleted_by = admin.id; db.commit(); db.refresh(teacher); return teacher
@router.get("/statistics")
def statistics(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    base = [User.role == UserRole.teacher, User.is_deleted.is_(False)]
    total = db.scalar(select(func.count()).select_from(User).where(*base)) or 0
    active = db.scalar(select(func.count()).select_from(User).where(*base, User.is_active.is_(True))) or 0
    month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    new = db.scalar(select(func.count()).select_from(User).where(*base, User.created_at >= month)) or 0
    return {"total_teachers": total, "active_teachers": active, "inactive_teachers": total - active, "new_teachers_this_month": new}


@router.get("/knowledge-documents")
def list_knowledge_documents(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    documents = db.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())).all()
    active_jobs = db.scalars(select(KnowledgeProcessingJob).where(KnowledgeProcessingJob.status.in_([KnowledgeProcessingJobStatus.pending, KnowledgeProcessingJobStatus.processing]))).all()
    active_jobs_by_document: dict[int, list[KnowledgeProcessingJob]] = {}
    for job in active_jobs:
        active_jobs_by_document.setdefault(job.document_id, []).append(job)
    finished_ingestions = db.scalars(select(KnowledgeProcessingJob).where(KnowledgeProcessingJob.job_type == KnowledgeProcessingJobType.ingestion, KnowledgeProcessingJob.status.in_([KnowledgeProcessingJobStatus.completed, KnowledgeProcessingJobStatus.failed])).order_by(KnowledgeProcessingJob.completed_at.desc(), KnowledgeProcessingJob.id.desc())).all()
    summaries: dict[int, dict] = {}
    for job in finished_ingestions:
        if job.result_summary is not None:
            summaries.setdefault(job.document_id, job.result_summary)
    return {"items": [{"id": document.id, "title": document.title, "document_type": document.document_type, "language": document.language, "cefr_level": document.cefr_level, "skill": document.skill, "source": document.source, "description": document.description, "original_filename": document.original_filename, "mime_type": document.mime_type, "file_size": document.file_size, "status": document.status.value, "created_at": document.created_at, "active_jobs": [job_response(job) for job in active_jobs_by_document.get(document.id, [])], "ingestion_summary": summaries.get(document.id), "preflight": {"status": document.preflight_status, "pages_total": document.preflight_pages_total, "pages_analyzed": document.preflight_pages_analyzed, "analysis_failed_pages": document.preflight_analysis_failed_pages, "native_good_pages": document.preflight_native_good_pages, "native_borderline_pages": document.preflight_native_borderline_pages, "native_bad_pages": document.preflight_native_bad_pages, "ocr_candidate_page_count": document.preflight_ocr_candidate_page_count, "ocr_required_page_ratio": document.preflight_ocr_required_page_ratio, "recommended_strategy": document.preflight_recommended_strategy, "estimated_complexity": document.preflight_estimated_complexity, "analyzed_at": document.preflight_analyzed_at} if document.preflight_status else None} for document in documents]}

@router.get("/knowledge-documents/{document_id}/segments")
def knowledge_segments(document_id: int, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), source_page: int | None = Query(None, ge=1), content_type: str | None = None, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Read-only persisted-chunk QA browser; it never parses a source PDF."""
    document = db.get(KnowledgeDocument, document_id)
    if not document: raise HTTPException(404, "Knowledge document not found")
    has_persisted_chunks = bool(db.scalar(
        select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id)
    ))
    query = select(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id)
    if source_page is not None:
        query = query.where(
            KnowledgeChunk.source_page_start <= source_page,
            KnowledgeChunk.source_page_end >= source_page,
        )
    if content_type:
        query = query.where(KnowledgeChunk.content_type == content_type)
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    persisted = db.scalars(
        query.order_by(KnowledgeChunk.chunk_index).offset((page - 1) * page_size).limit(page_size)
    ).all()
    if has_persisted_chunks:
        return {
            "availability": "available",
            "items": [{
                "id": str(chunk.id), "chunk_index": chunk.chunk_index,
                "page_start": chunk.source_page_start, "page_end": chunk.source_page_end,
                "content_type": chunk.content_type, "extraction_mode": chunk.extraction_mode,
                "token_count": chunk.token_count, "headings": chunk.heading_context,
                "content": chunk.content,
            } for chunk in persisted],
            "total": total, "page": page, "page_size": page_size,
        }

    # Older successful jobs predate relational chunk persistence. This bounded,
    # document-id-derived diagnostic fallback is deliberately labelled legacy.
    artifact = DEBUG_CHUNKS_DIRECTORY / f"{document_id}.json"
    if not artifact.is_file():
        availability = "not_persisted" if document.status in {KnowledgeDocumentStatus.ready, KnowledgeDocumentStatus.partial} else "unavailable"
        return {"availability": availability, "items": [], "total": 0, "page": page, "page_size": page_size}
    try: payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return {"availability": "unavailable", "items": [], "total": 0, "page": page, "page_size": page_size}
    if payload.get("document_id") != document_id or not isinstance(payload.get("chunks"), list):
        return {"availability": "unavailable", "items": [], "total": 0, "page": page, "page_size": page_size}
    chunks = payload.get("chunks", [])
    if source_page is not None: chunks = [chunk for chunk in chunks if chunk.get("page_start") == source_page or chunk.get("page_end") == source_page]
    if content_type: chunks = [chunk for chunk in chunks if chunk.get("content_type") == content_type]
    items = [{"id": chunk.get("id"), "chunk_index": chunk.get("chunk_index"), "page_start": chunk.get("page_start"), "page_end": chunk.get("page_end"), "content_type": chunk.get("content_type"), "extraction_mode": chunk.get("metadata", {}).get("extraction_mode"), "token_count": chunk.get("token_count"), "headings": chunk.get("headings", []), "content": chunk.get("text_original", "")} for chunk in chunks]
    total = len(items); start = (page - 1) * page_size
    return {"availability": "legacy_debug", "items": items[start:start + page_size], "total": total, "page": page, "page_size": page_size}


@router.post("/knowledge-documents/preflight-selected")
def enqueue_knowledge_preflight_selection(selection: KnowledgeDocumentSelection, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Enqueue only; Docling never runs in the HTTP request."""
    return KnowledgeProcessingQueue().enqueue_preflight(db, selection.document_ids)


@router.post("/knowledge-documents/process-selected")
def enqueue_knowledge_ingestion_selection(selection: KnowledgeDocumentSelection, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Enqueue eligible documents only; this never starts a synchronous preflight."""
    return KnowledgeProcessingQueue().enqueue_ingestion(db, selection.document_ids)


@router.get("/knowledge-processing-jobs")
def list_knowledge_processing_jobs(
    document_ids: list[int] | None = Query(None),
    active_only: bool = Query(False),
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = select(KnowledgeProcessingJob).order_by(KnowledgeProcessingJob.created_at.desc(), KnowledgeProcessingJob.id.desc())
    if document_ids:
        query = query.where(KnowledgeProcessingJob.document_id.in_(document_ids))
    if active_only:
        query = query.where(KnowledgeProcessingJob.status.in_(["pending", "processing"]))
    return {"items": [job_response(job) for job in db.scalars(query).all()]}


@router.post("/knowledge-documents", status_code=http_status.HTTP_201_CREATED)
async def create_knowledge_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    document_type: str | None = Form(None),
    language: str | None = Form(None),
    cefr_level: str | None = Form(None),
    skill: str | None = Form(None),
    source: str | None = Form(None),
    description: str | None = Form(None),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    allowed_types = {"cecrl", "miftah", "guide", "program", "other"}
    allowed_languages = {"ar", "fr", "en", "es", "other"}
    if document_type is not None and document_type not in allowed_types:
        raise HTTPException(status_code=422, detail="Invalid document type")
    if language is not None and language not in allowed_languages:
        raise HTTPException(status_code=422, detail="Invalid document language")
    if title is not None and not title.strip():
        raise HTTPException(status_code=422, detail="Invalid document metadata")
    if not file.filename or file.content_type != "application/pdf":
        raise HTTPException(status_code=422, detail="A PDF file is required")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded PDF is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 25 MB limit")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid PDF")

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{uuid4().hex}.pdf"
    destination = UPLOAD_DIRECTORY / stored_filename
    original_filename = Path(file.filename).name
    try:
        destination.write_bytes(content)
        # The persisted bytes are the only source that may be accepted.  A PDF
        # signature alone is insufficient: PDFium must open at least one page.
        if not destination.is_file() or destination.read_bytes() != content or not stored_pdf_is_valid(destination):
            raise HTTPException(status_code=422, detail="Uploaded PDF could not be validated")
        document = KnowledgeDocument(
            title=title.strip() if title else title_from_filename(original_filename),
            document_type=document_type or None, language=language or None,
            cefr_level=cefr_level or None, skill=skill or None, source=source.strip() if source else None,
            description=description.strip() if description else None, original_filename=original_filename,
            stored_filename=stored_filename, file_path=str(destination), mime_type=file.content_type, file_size=len(content),
            status=KnowledgeDocumentStatus.pending, uploaded_by=admin.id,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"id": document.id, "title": document.title, "document_type": document.document_type, "language": document.language, "cefr_level": document.cefr_level, "skill": document.skill, "source": document.source, "description": document.description, "original_filename": document.original_filename, "mime_type": document.mime_type, "file_size": document.file_size, "status": document.status.value, "created_at": document.created_at}


@router.post("/knowledge-documents/{document_id}/parse-preview")
def parse_knowledge_document_preview(
    document_id: int,
    export_debug: bool = Query(False),
    force_reprocess: bool = Query(False),
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    document = db.get(KnowledgeDocument, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Knowledge document not found")
    try:
        return KnowledgeIngestionService().parse_preview(db, document, export_debug=export_debug, force_reprocess=force_reprocess).to_response(include_debug=export_debug)
    except KnowledgeIngestionError as error:
        raise HTTPException(status_code=422, detail="Document parsing failed. Check server logs for details.") from error


@router.post("/knowledge-documents/{document_id}/preflight")
def preflight_knowledge_document(
    document_id: int,
    export_debug: bool = Query(False),
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    document = db.get(KnowledgeDocument, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Knowledge document not found")
    return KnowledgePreflightService().analyze(db, document).to_response(export_debug=export_debug)
