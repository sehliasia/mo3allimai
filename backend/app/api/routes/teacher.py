from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.api.dependencies import require_teacher
from app.database.session import SessionLocal, get_db
from app.models.teacher_library import TeacherActivity, TeacherLibraryChunk, TeacherLibraryDocument, TeacherSavedResource
from app.services.qdrant_service import QdrantService
from app.services.teacher_library_ingestion_service import TeacherLibraryIngestionError, TeacherLibraryIngestionService
from app.models.user import User
from app.schemas.lesson_plan import LessonPlanGenerateIn, LessonPlanOut
from app.services.cefr_knowledge_service import CEFRKnowledgeService
from app.services.context_builder import ContextBuilder
from app.services.embedding_providers import get_embedding_provider
from app.services.lesson_plan_generation_service import LessonPlanGenerationError, LessonPlanGenerationService
from app.services.llm_providers import get_llm_provider
from app.services.pedagogical_knowledge_service import PedagogicalKnowledgeRequest, PedagogicalKnowledgeService
from app.services.retrieval_service import RetrievalService
from app.core.config import get_settings

router = APIRouter(prefix="/teacher", tags=["teacher"])
UPLOAD_ROOT = Path(__file__).resolve().parents[3] / "uploads" / "teacher-library"
ALLOWED = {".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".txt": "text/plain"}
MAX_BYTES = 10 * 1024 * 1024

def document_data(item: TeacherLibraryDocument): return {"id": item.id, "kind": "document", "title": item.title, "original_filename": item.original_filename, "mime_type": item.mime_type, "file_size": item.file_size, "status": item.status, "processing_stage": item.processing_stage, "processing_error": item.processing_error, "created_at": item.created_at}
def resource_data(item: TeacherSavedResource): return {"id": item.id, "kind": "creation", "resource_type": item.resource_type, "title": item.title, "cefr_level": item.cefr_level, "theme": item.theme, "created_at": item.created_at, "updated_at": item.updated_at}
def resource_detail_data(item: TeacherSavedResource): return {**resource_data(item), "content": item.content}

class SavedResourceIn(BaseModel):
    resource_type: str = Field(pattern="^(lesson-plan|lesson|exercises|exam|assessment|activity|summary)$")
    title: str = Field(min_length=1, max_length=255)
    cefr_level: str | None = Field(default=None, pattern="^(A1|A2|B1|B2|C1|C2)$")
    theme: str | None = Field(default=None, max_length=255)
    content: dict

class SavedResourceUpdateIn(SavedResourceIn):
    pass


@router.post("/ai/lesson-plans/generate", response_model=LessonPlanOut)
def generate_lesson_plan(payload: LessonPlanGenerateIn, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    """Retrieve optional pedagogical context then make exactly one LLM call."""
    settings = get_settings()
    try:
        knowledge = PedagogicalKnowledgeService(
            cefr=CEFRKnowledgeService(),
            retrieval=RetrievalService(provider=get_embedding_provider(), qdrant=QdrantService()),
            context_builder=ContextBuilder(max_chunks=min(4, settings.rag_context_max_chunks), max_tokens=min(1200, settings.rag_context_max_tokens)),
            settings=settings,
        )
        context = knowledge.build_context(db, PedagogicalKnowledgeRequest(
            cefr_level=payload.level, topic=payload.theme, objective=payload.general_objective,
            language=payload.language, skills=tuple(payload.skills), competencies=tuple(payload.skills),
            activity_type=payload.session_type, retrieval_top_k=4,
        ))
    except Exception:
        # Retrieval enhances the plan but never prevents a general generation.
        from app.services.pedagogical_knowledge_service import PedagogicalContext
        context = PedagogicalContext({"cefr_level": payload.level, "language": payload.language}, [], [], [], 0, 0, [], ["RAG unavailable; general generation used."], 0)
    try:
        return LessonPlanGenerationService(llm=get_llm_provider(settings), settings=settings).generate(payload, context)
    except LessonPlanGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@router.get("/library")
def list_library(teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    docs = db.scalars(select(TeacherLibraryDocument).where(TeacherLibraryDocument.owner_id == teacher.id).order_by(TeacherLibraryDocument.created_at.desc())).all()
    resources = db.scalars(select(TeacherSavedResource).where(TeacherSavedResource.owner_id == teacher.id).order_by(TeacherSavedResource.created_at.desc())).all()
    return {"items": sorted([*(document_data(x) for x in docs), *(resource_data(x) for x in resources)], key=lambda x: x["created_at"], reverse=True)}

def _ingest_private_document(document_id: int) -> None:
    """Independent DB session for FastAPI's post-response background task."""
    db = SessionLocal()
    try:
        item = db.get(TeacherLibraryDocument, document_id)
        if item is not None and item.status in {"pending", "processing"}:
            TeacherLibraryIngestionService().ingest(db, item, storage_root=UPLOAD_ROOT)
    except TeacherLibraryIngestionError:
        # The service already records a safe terminal status and logs the cause.
        pass
    finally:
        db.close()


@router.post("/library/documents", status_code=201)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    original = Path(file.filename or "").name
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED or file.content_type not in {ALLOWED[ext], "application/octet-stream"}: raise HTTPException(422, "Unsupported document type")
    content = await file.read(MAX_BYTES + 1)
    if not content or len(content) > MAX_BYTES: raise HTTPException(422, "Invalid document size")
    directory = UPLOAD_ROOT / str(teacher.id); directory.mkdir(parents=True, exist_ok=True)
    key = f"{teacher.id}/{uuid4().hex}{ext}"; destination = UPLOAD_ROOT / key
    try:
        destination.write_bytes(content)
        item = TeacherLibraryDocument(owner_id=teacher.id, title=Path(original).stem or "document", original_filename=original, mime_type=ALLOWED[ext], file_size=len(content), storage_key=key, status="pending", processing_stage="uploaded", processing_error=None)
        db.add(item); db.flush(); db.add(TeacherActivity(owner_id=teacher.id, activity_type="document_uploaded", resource_type="document", resource_id=item.id, title=item.title, metadata_json={"filename": original})); db.commit(); db.refresh(item)
        background_tasks.add_task(_ingest_private_document, item.id)
        return document_data(item)
    except Exception:
        db.rollback(); destination.unlink(missing_ok=True); raise
    finally: await file.close()

@router.post("/library/resources", status_code=201)
def save_resource(data: SavedResourceIn, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    item = TeacherSavedResource(owner_id=teacher.id, **data.model_dump()); db.add(item); db.flush(); db.add(TeacherActivity(owner_id=teacher.id, activity_type="resource_saved", resource_type=item.resource_type, resource_id=item.id, title=item.title, metadata_json={"cefr_level": item.cefr_level, "theme": item.theme})); db.commit(); db.refresh(item); return resource_data(item)

@router.get("/library/resources/{item_id}")
def get_resource(item_id: int, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    item = db.scalar(select(TeacherSavedResource).where(TeacherSavedResource.id == item_id, TeacherSavedResource.owner_id == teacher.id))
    if not item: raise HTTPException(404, "Resource not found")
    return resource_detail_data(item)

@router.patch("/library/resources/{item_id}")
def update_resource(item_id: int, data: SavedResourceUpdateIn, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    item = db.scalar(select(TeacherSavedResource).where(TeacherSavedResource.id == item_id, TeacherSavedResource.owner_id == teacher.id))
    if not item: raise HTTPException(404, "Resource not found")
    for key, value in data.model_dump().items(): setattr(item, key, value)
    db.add(TeacherActivity(owner_id=teacher.id, activity_type="resource_updated", resource_type=item.resource_type, resource_id=item.id, title=item.title, metadata_json={"cefr_level": item.cefr_level, "theme": item.theme})); db.commit(); db.refresh(item)
    return resource_detail_data(item)

@router.delete("/library/documents/{item_id}", status_code=204)
def delete_document(item_id: int, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    item = db.scalar(select(TeacherLibraryDocument).where(TeacherLibraryDocument.id == item_id, TeacherLibraryDocument.owner_id == teacher.id))
    if not item: raise HTTPException(404, "Document not found")
    point_ids = list(db.scalars(select(TeacherLibraryChunk.vector_point_id).where(TeacherLibraryChunk.document_id == item.id, TeacherLibraryChunk.vector_point_id.is_not(None))))
    try:
        if point_ids: QdrantService().delete_points(point_ids)
        (UPLOAD_ROOT / item.storage_key).unlink(missing_ok=True); db.delete(item); db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(503, "Document deletion could not be completed")

@router.delete("/library/resources/{item_id}", status_code=204)
def delete_resource(item_id: int, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    item = db.scalar(select(TeacherSavedResource).where(TeacherSavedResource.id == item_id, TeacherSavedResource.owner_id == teacher.id))
    if not item: raise HTTPException(404, "Resource not found")
    db.delete(item); db.commit()

@router.get("/history")
def history(teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    items = db.scalars(select(TeacherActivity).where(TeacherActivity.owner_id == teacher.id).order_by(TeacherActivity.created_at.desc())).all()
    return {"items": [{"id": x.id, "activity_type": x.activity_type, "resource_type": x.resource_type, "resource_id": x.resource_id, "title": x.title, "metadata": x.metadata_json, "created_at": x.created_at} for x in items]}
