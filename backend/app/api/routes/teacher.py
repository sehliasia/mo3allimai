from pathlib import Path
from uuid import uuid4
import logging
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
from app.schemas.activity_generator import ActivityGenerateIn, ActivityOut
from app.schemas.course_generator import CourseGenerateIn, CourseOut
from app.schemas.exercise_generator import (
    ExerciseAdaptIn,
    ExerciseGenerateIn,
    ExerciseItem,
    ExerciseOut,
    ExerciseSearchIn,
    ExerciseSearchItem,
    ExerciseSearchOut,
)
from app.services.cefr_knowledge_service import CEFRKnowledgeService
from app.services.context_builder import ContextBuilder
from app.services.embedding_providers import get_embedding_provider
from app.services.lesson_plan_generation_service import LessonPlanGenerationError, LessonPlanGenerationService
from app.services.activity_generation_service import ActivityGenerationError, ActivityGenerationService
from app.services.course_generation_service import CourseGenerationError, CourseGenerationService, CourseRateLimitError
from app.services.exercise_adaptation_service import ExerciseAdaptationError, ExerciseAdaptationRateLimitError, ExerciseAdaptationService
from app.services.exercise_cefr import normalize_level as _normalize_level
from app.services.exercise_extraction_service import (
    ExerciseExtractionError,
    ExerciseExtractionRateLimitError,
    ExerciseExtractionService,
)
from app.services.exercise_generation_service import ExerciseGenerationError, ExerciseGenerationService, ExerciseRateLimitError
from app.services.exercise_search_service import (
    ExerciseSearchService,
    StructuredExerciseQuery,
    build_retrieval_query,
    enrich_exercise_blocks,
    parse_query,
)
from app.services.llm_providers import get_llm_provider
from app.services.pedagogical_knowledge_service import PedagogicalKnowledgeRequest, PedagogicalKnowledgeService
from app.services.retrieval_service import RetrievalFilters, RetrievalService
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_DEDUP_PRIORITY = {"extracted": 0, "deterministic": 1}


def _exercise_dedup_key(item: ExerciseSearchItem) -> tuple[object, ...]:
    """Dedup key across the extracted and deterministic passes: document +
    normalized instruction prefix (an exercise legitimately spans chunks, so the
    identity is the instruction, not the whole prompt)."""
    prompt = " ".join((item.prompt or "").split()).casefold()[:120] or " ".join((item.title or "").split()).casefold()
    return (item.document_id, prompt)


def _merge_exercises(
    extracted: list[ExerciseSearchItem],
    deterministic: list[ExerciseSearchItem],
) -> list[ExerciseSearchItem]:
    """Prefer LLM-extracted items, fill gaps with deterministic ones, dedupe."""
    merged: list[ExerciseSearchItem] = []
    seen: set[tuple[object, ...]] = set()
    for source_key, items in ((_DEDUP_PRIORITY["extracted"], extracted), (_DEDUP_PRIORITY["deterministic"], deterministic)):
        for item in items:
            key = _exercise_dedup_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _log_exercise_search(*, query: str, parsed, response, blocks, extracted, items, extraction_calls) -> None:
    logger.info(
        "ExerciseSearch query=%r parsed_constraints=%r candidate_count=%s "
        "retrieved_chunks=%s expanded_blocks=%s candidate_exercises=%s "
        "extracted_exercises=%s deduplicated_exercises=%s final_exercises=%s "
        "retrieval_llm_calls=0 extraction_llm_calls=%s",
        query,
        parsed.to_dict() if hasattr(parsed, "to_dict") else parsed,
        getattr(response, "union_candidate_count", 0),
        getattr(response, "union_candidate_count", 0),
        len(blocks),
        len(extracted),
        len(extracted),
        max(0, len(extracted) + len(extracted) - len(items)),
        len(items),
        extraction_calls,
    )

router = APIRouter(prefix="/teacher", tags=["teacher"])
UPLOAD_ROOT = Path(__file__).resolve().parents[3] / "uploads" / "teacher-library"
ALLOWED = {".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".txt": "text/plain"}
MAX_BYTES = 10 * 1024 * 1024

def document_data(item: TeacherLibraryDocument): return {"id": item.id, "kind": "document", "title": item.title, "original_filename": item.original_filename, "mime_type": item.mime_type, "file_size": item.file_size, "status": item.status, "processing_stage": item.processing_stage, "processing_error": item.processing_error, "created_at": item.created_at}
def resource_data(item: TeacherSavedResource): return {"id": item.id, "kind": "creation", "resource_type": item.resource_type, "title": item.title, "cefr_level": item.cefr_level, "theme": item.theme, "created_at": item.created_at, "updated_at": item.updated_at}
def resource_detail_data(item: TeacherSavedResource): return {**resource_data(item), "content": item.content}

class SavedResourceIn(BaseModel):
    resource_type: str = Field(pattern="^(lesson-plan|lesson|exercises|exam|assessment|activity|course)$")
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

@router.post("/ai/activities/generate", response_model=ActivityOut)
def generate_activity(payload: ActivityGenerateIn, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
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
            cefr_level=payload.level, topic=payload.theme, objective=payload.objective,
            language=payload.language, skills=tuple(payload.skills), competencies=tuple(payload.skills),
            activity_type=payload.activity_type, retrieval_top_k=4,
        ))
    except Exception:
        # Retrieval enhances the activity but never prevents a general generation.
        from app.services.pedagogical_knowledge_service import PedagogicalContext
        context = PedagogicalContext({"cefr_level": payload.level, "language": payload.language}, [], [], [], 0, 0, [], ["RAG unavailable; general generation used."], 0)
    try:
        return ActivityGenerationService(llm=get_llm_provider(settings), settings=settings).generate(payload, context)
    except ActivityGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@router.post("/ai/courses/generate", response_model=CourseOut)
def generate_course(payload: CourseGenerateIn, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
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
            cefr_level=payload.level, topic=payload.theme, objective=payload.objective,
            language=payload.language, skills=tuple(payload.skills), competencies=tuple(payload.skills),
            activity_type="cours", retrieval_top_k=4,
        ))
    except Exception:
        # Retrieval enhances the course but never prevents a general generation.
        from app.services.pedagogical_knowledge_service import PedagogicalContext
        context = PedagogicalContext({"cefr_level": payload.level, "language": payload.language}, [], [], [], 0, 0, [], ["RAG unavailable; general generation used."], 0)
    try:
        return CourseGenerationService(llm=get_llm_provider(settings), settings=settings).generate(payload, context)
    except CourseRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except CourseGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@router.post("/ai/exercises/generate", response_model=ExerciseOut)
def generate_exercises(payload: ExerciseGenerateIn, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    """LLM-first exercise generation. An AI generation engine plans a varied,
    level-appropriate set; the RAG pedagogical context is gathered as reference
    material (reuse of vocabulary/notions, provenance-aware) rather than a hard
    constraint. Degrades to KB-only extraction when no provider is configured."""
    settings = get_settings()
    try:
        knowledge = PedagogicalKnowledgeService(
            cefr=CEFRKnowledgeService(),
            retrieval=RetrievalService(provider=get_embedding_provider(), qdrant=QdrantService()),
            context_builder=ContextBuilder(max_chunks=min(4, settings.rag_context_max_chunks), max_tokens=min(1200, settings.rag_context_max_tokens)),
            settings=settings,
        )
        context = knowledge.build_context(db, PedagogicalKnowledgeRequest(
            cefr_level=payload.level, topic=payload.theme, objective=payload.objective,
            language=payload.language, skills=tuple(payload.skills), competencies=tuple(payload.skills),
            # "auto" means the planner decides the per-exercise type mix; don't
            # constrain retrieval with an empty/meaningless activity filter.
            activity_type=None if (payload.exercise_type or "").casefold() in ("auto", "") else payload.exercise_type,
            retrieval_top_k=10,
        ))
    except Exception:
        # Retrieval defines the KB-first source; if unavailable we must not fake it.
        from app.services.pedagogical_knowledge_service import PedagogicalContext
        context = PedagogicalContext({"cefr_level": payload.level, "language": payload.language}, [], [], [], 0, 0, [], ["RAG unavailable; no source material."], 0)
    try:
        return ExerciseGenerationService(llm=get_llm_provider(settings), settings=settings).generate(payload, context)
    except ExerciseRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ExerciseGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@router.post("/ai/exercises/search", response_model=ExerciseSearchOut)
def search_exercises(payload: ExerciseSearchIn, teacher: User = Depends(require_teacher), db: Session = Depends(get_db)):
    """Deep RAG exercise search.

    Retrieval stays 0 LLM: hybrid (dense + BM25 + RRF), multi-chunk expansion
    via ContextBuilder, then an optional LLM extraction stage that semantically
    understands the passages, decides whether they truly contain exercises and
    structures them (never inventing content or provenance). A deterministic
    pass (score_exercise) runs on the same blocks as a resilient fallback and
    the two are merged/deduplicated. Constraints are applied only when stated.
    """
    settings = get_settings()
    try:
        retrieval = RetrievalService(provider=get_embedding_provider(), qdrant=QdrantService())
        context_builder = ContextBuilder(
            max_chunks=min(settings.exercise_search_block_max_chunks, 48),
            max_tokens=settings.exercise_search_block_max_tokens,
        )

        # Deterministic parse: constraints only when explicitly stated.
        parsed = parse_query(payload.query)
        level = payload.level or parsed.level
        level = _normalize_level(level) if level else None
        skills = list(payload.skills) or parsed.skills
        ex_type = payload.exercise_type or parsed.exercise_type

        # Retrieval (0 LLM): hybrid dense + BM25 + RRF.
        query = build_retrieval_query(StructuredExerciseQuery(
            raw_query=payload.query, level=level, skills=skills,
            exercise_type=ex_type, theme_tokens=parsed.theme_tokens,
        ))
        # Retrieval mirrors the shared pedagogical RAG used by every other
        # generator (courses, activities, lesson plans): it searches the whole
        # common knowledge base with no hard CEFR / owner exclusion. The
        # requested level is enforced *after* retrieval at the item level
        # (explicit_other_level -> dropped; never relabelled), which keeps the
        # strict "A2 is never presented as A1" guarantee without hiding admin
        # documents whose indexed cefr_level is unset or broader than the query.
        response = retrieval.search(
            db, query,
            top_k=settings.exercise_extraction_candidate_k,
            rerank=False,
            filters=RetrievalFilters(document_ids=list(payload.source_document_ids) or None),
        )
        pool = response.results
        built = context_builder.build(response.query, pool, db=db) if pool else None
        blocks = list(built.source_blocks) if built is not None else []
        # Exercise-specific in-memory reconstruction: re-group the already
        # retrieved chunks (same document, contiguous in chunk_index) so a
        # fragmented exercise (title chunk + directive + "الاختيارات:" options)
        # is rebuilt as one complete block before both the LLM extraction and
        # the deterministic pass. No reindexing, no new collection, and it never
        # fuses two distinct exercises ("تمرين 1" + "تمرين 2").
        if blocks and db is not None:
            blocks = enrich_exercise_blocks(
                blocks, db=db, max_tokens=settings.exercise_search_block_max_tokens,
            )

        # LLM extraction stage (after retrieval; retrieval stays 0 LLM).
        extraction_calls = 0
        extracted_items: list[ExerciseSearchItem] = []
        if blocks and settings.exercise_extraction_required:
            try:
                extractor = ExerciseExtractionService(llm=get_llm_provider(settings), settings=settings)
                extracted_items, extraction_calls = extractor.extract(
                    blocks, request=payload, parsed_level=level,
                )
            except ExerciseExtractionRateLimitError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            except ExerciseExtractionError as exc:
                logger.warning("exercise_extraction_failed query=%r err=%r", payload.query, str(exc))

        # Deterministic pass on the same blocks (never a second retrieval).
        det_service = ExerciseSearchService(retrieval=retrieval, context_builder=context_builder, settings=settings)
        deterministic = det_service.search(
            db, payload, expanded_blocks=blocks, prebuilt_response=response,
        )

        merged = _merge_exercises(extracted_items, list(deterministic.items))
        # Re-rank extracted-first when present; otherwise keep deterministic order.
        items = merged if extraction_calls else list(deterministic.items)
        out = deterministic.model_copy(deep=True)
        out.items = items[: payload.limit]
        out.total = len(out.items)
        out.meta.llm_calls = extraction_calls
        out.meta.retrieval_llm_calls = 0
        out.meta.extraction_llm_calls = extraction_calls
        out.meta.candidate_blocks = len(blocks)
        out.meta.expanded_blocks = len(blocks)
        out.meta.extracted_blocks = len(extracted_items)
        out.meta.deduplicated_count = max(0, (len(extracted_items) + len(deterministic.items)) - len(merged))
        _log_exercise_search(
            query=payload.query, parsed=parsed, response=response, blocks=blocks,
            extracted=extracted_items, items=out.items, extraction_calls=extraction_calls,
        )
        return out
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="La recherche d'exercices n'a pas pu être effectuée.") from exc

@router.post("/ai/exercises/adapt", response_model=ExerciseItem)
def adapt_exercise(payload: ExerciseAdaptIn, teacher: User = Depends(require_teacher)):
    """Explicit AI adaptation of one KB exercise towards a target level. The
    result keeps status "adapted_from_kb" plus original level and provenance."""
    settings = get_settings()
    try:
        service = ExerciseAdaptationService(llm=get_llm_provider(settings), settings=settings)
        return service.adapt(payload)
    except ExerciseAdaptationRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ExerciseAdaptationError as exc:
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
