"""Authenticated, single-turn pedagogical assistant endpoint."""

from dataclasses import asdict
import asyncio
import json
import logging
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.dependencies import require_teacher
from app.core.config import get_settings
from app.database.session import get_db
from app.models.user import User
from app.schemas.assistant import (
    AssistantChatIn,
    AssistantChatOut,
    AssistantConversationDetailOut,
    AssistantConversationListOut,
    AssistantMessageRegenerateIn,
)
from app.services.assistant_conversation_service import (
    AssistantConversationNotFoundError,
    AssistantConversationPersistenceError,
    AssistantConversationService,
)
from app.services.assistant_chat_service import (
    AssistantChatRequest,
    AssistantChatService,
    AssistantChatServiceError,
    AssistantChatValidationError,
)
from app.services.cefr_knowledge_service import CEFRKnowledgeService
from app.services.context_builder import ContextBuilder
from app.services.embedding_providers import get_embedding_provider
from app.services.llm_providers import get_arabic_review_fallback_provider, get_llm_provider
from app.services.pedagogical_knowledge_service import PedagogicalKnowledgeService
from app.services.qdrant_service import QdrantService
from app.services.reranker_providers import get_reranker_provider
from app.services.retrieval_service import RetrievalService
from app.services.personal_retrieval_service import PersonalRetrievalService
from app.services.retrieval_pipeline import resolve_effective_retrieval_pipeline


router = APIRouter(prefix="/assistant", tags=["assistant"])
logger = logging.getLogger(__name__)


def get_assistant_chat_service() -> AssistantChatService:
    """Create lightweight boundaries only; providers/models remain lazy until used."""
    settings = get_settings()
    pipeline = resolve_effective_retrieval_pipeline(settings)
    retrieval = RetrievalService(
        provider=get_embedding_provider(), qdrant=QdrantService(),
        mode=pipeline.retrieval_mode,
        pedagogical_ranking_enabled=pipeline.pedagogical_ranking,
        reranker=get_reranker_provider() if pipeline.reranker else None,
    )
    knowledge = PedagogicalKnowledgeService(
        cefr=CEFRKnowledgeService(), retrieval=retrieval,
        context_builder=ContextBuilder(
            max_chunks=settings.rag_context_max_chunks, max_tokens=settings.rag_context_max_tokens,
        ),
        settings=settings,
    )
    llm = get_llm_provider(settings)
    return AssistantChatService(
        knowledge=knowledge,
        llm=llm,
        review_fallback_llm=get_arabic_review_fallback_provider(settings),
        personal_retrieval=PersonalRetrievalService(qdrant=QdrantService()),
        settings=settings,
    )


def get_assistant_conversation_service() -> AssistantConversationService:
    """Provide the isolated persistence boundary without constructing RAG services."""
    return AssistantConversationService()


def _source_data(source) -> dict:
    return {
        "source_type": source.source_type,
        "document_id": source.document_id,
        "document_title": source.document_title,
        "page_start": source.page_start,
        "page_end": source.page_end,
        "descriptor_scale": source.descriptor_scale,
    }


def _detail_data(conversation) -> dict:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "archived_at": conversation.archived_at,
        "messages": [
            {
                "id": message.id,
                "role": message.role.value if hasattr(message.role, "value") else message.role,
                "content": message.content,
                "created_at": message.created_at,
                "sources": [_source_data(source) for source in sorted(
                    message.sources, key=lambda item: item.source_order,
                )],
            }
            for message in sorted(conversation.messages, key=lambda item: (item.created_at, item.id))
        ],
    }


@router.post(
    "/chat", response_model=AssistantChatOut,
    summary="Ask the pedagogical assistant",
    description="Return one grounded pedagogical answer with safe source metadata.",
)
def chat(
    payload: AssistantChatIn,
    teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
    service: AssistantChatService = Depends(get_assistant_chat_service),
    conversations: AssistantConversationService = Depends(get_assistant_conversation_service),
):
    """Persist one teacher-owned exchange; prior messages are not yet prompt context."""
    try:
        conversation = (
            conversations.get_owned_conversation(
                db, user_id=teacher.id, conversation_id=payload.conversation_id,
            )
            if payload.conversation_id is not None else None
        )
    except AssistantConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    history = (
        conversations.get_recent_history(
            db,
            user_id=teacher.id,
            conversation_id=conversation.id,
            max_messages=get_settings().assistant_history_max_messages,
            max_chars=get_settings().assistant_history_max_chars,
        )
        if conversation is not None else ()
    )
    try:
        pending = conversations.begin_exchange(
            db, user_id=teacher.id, user_content=payload.message, conversation=conversation,
        )
    except AssistantConversationPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "ASSISTANT_PERSISTENCE_ERROR", "message": "Impossible d’enregistrer cet échange."},
        ) from exc
    try:
        assistant_request = AssistantChatRequest(
            message=payload.message, cefr_level=payload.cefr_level, skills=tuple(payload.skills),
            language=payload.language, topic=payload.topic, objective=payload.objective, top_k=payload.top_k,
            mode=payload.mode, document_ids=tuple(payload.document_ids),
        )
        response = service.answer(db, assistant_request, history=history, **({"owner_id": teacher.id} if payload.mode == "user_documents" else {}))
    except AssistantChatValidationError as exc:
        conversations.rollback_exchange(db)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "ASSISTANT_VALIDATION_ERROR", "message": str(exc)},
        ) from exc
    except AssistantChatServiceError as exc:
        conversations.rollback_exchange(db)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": exc.code, "message": "Le service d’IA est momentanément indisponible."},
        ) from exc
    except Exception:
        # Keep the unit of work clean even for an unexpected generation failure.
        conversations.rollback_exchange(db)
        raise
    try:
        exchange = conversations.complete_exchange(
            db,
            pending=pending,
            assistant_response=response,
        )
    except AssistantConversationPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "ASSISTANT_PERSISTENCE_ERROR", "message": "Impossible d’enregistrer cet échange."},
        ) from exc
    return {**asdict(response), **asdict(exchange)}


def _stream_event(event: dict[str, object]) -> bytes:
    return (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@router.post("/chat/stream", summary="Stream a pedagogical assistant answer")
async def chat_stream(
    payload: AssistantChatIn,
    request: Request,
    teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
    service: AssistantChatService = Depends(get_assistant_chat_service),
    conversations: AssistantConversationService = Depends(get_assistant_conversation_service),
):
    """NDJSON stream; only the final accumulated content is persisted once."""
    try:
        conversation = conversations.get_owned_conversation(db, user_id=teacher.id, conversation_id=payload.conversation_id) if payload.conversation_id is not None else None
        history = conversations.get_recent_history(db, user_id=teacher.id, conversation_id=conversation.id, max_messages=get_settings().assistant_history_max_messages, max_chars=get_settings().assistant_history_max_chars) if conversation else ()
        pending = conversations.begin_exchange(db, user_id=teacher.id, user_content=payload.message, conversation=conversation)
        assistant_request = AssistantChatRequest(
            message=payload.message, cefr_level=payload.cefr_level, skills=tuple(payload.skills), language=payload.language,
            topic=payload.topic, objective=payload.objective, top_k=payload.top_k,
            mode=payload.mode, document_ids=tuple(payload.document_ids),
        )
        logger.info(
            "[ASSISTANT] request_received message_chars=%s language=%s mode=%s",
            len(payload.message), payload.language, payload.mode,
        )
        prepared = service.prepare_stream(db, assistant_request, history=history, **({"owner_id": teacher.id} if payload.mode == "user_documents" else {}))
    except AssistantConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    except AssistantChatValidationError as exc:
        conversations.rollback_exchange(db)
        raise HTTPException(status_code=422, detail={"code": "ASSISTANT_VALIDATION_ERROR", "message": str(exc)}) from exc
    except AssistantChatServiceError as exc:
        conversations.rollback_exchange(db)
        raise HTTPException(status_code=503, detail={"code": exc.code, "message": "Le service d’IA est momentanément indisponible."}) from exc
    except Exception:
        conversations.rollback_exchange(db)
        raise

    async def events():
        started = perf_counter(); first_token_ms: int | None = None; content = ""; completed = False; persisted = False
        logger.info("assistant_stream_started conversation_id=%s", pending.conversation.id)
        try:
            yield _stream_event({"type": "start", "conversation_id": pending.conversation.id, "message_id": pending.user_message.id})
            logger.info(
                "[ASSISTANT] llm_call_started conversation_id=%s system_prompt_chars=%s user_prompt_chars=%s",
                pending.conversation.id, len(prepared.system_prompt or ""), len(prepared.user_prompt or ""),
            )
            provider_stream = service.stream_answer(prepared)
            try:
                async for delta in provider_stream:
                    if await request.is_disconnected():
                        logger.info("assistant_stream_cancelled conversation_id=%s chars=%s", pending.conversation.id, len(content))
                        break
                    if first_token_ms is None:
                        first_token_ms = int((perf_counter() - started) * 1000)
                        logger.info("assistant_stream_first_token_ms conversation_id=%s elapsed_ms=%s", pending.conversation.id, first_token_ms)
                    logger.info("[ASSISTANT] llm_chunk_received conversation_id=%s chars=%s", pending.conversation.id, len(delta))
                    content += delta
                    yield _stream_event({"type": "delta", "content": delta})
                    logger.info("[ASSISTANT] sse_chunk_sent conversation_id=%s chars=%s", pending.conversation.id, len(delta))
                else:
                    completed = True
            finally:
                await provider_stream.aclose()
            if content.strip():
                response = service.stream_response(prepared, content)
                exchange = conversations.complete_exchange(db, pending=pending, assistant_response=response)
                persisted = True
                yield _stream_event({"type": "done" if completed else "stopped", "assistant_message_id": exchange.assistant_message_id, "sources": [asdict(source) for source in response.sources], "diagnostics": asdict(response.diagnostics)})
            else:
                db.commit()  # Preserve the user message only; never write an empty assistant response.
                persisted = True
                if not await request.is_disconnected():
                    yield _stream_event({"type": "done" if completed else "stopped"})
        except asyncio.CancelledError:
            logger.info("assistant_stream_cancelled conversation_id=%s chars=%s", pending.conversation.id, len(content))
            if content.strip() and not persisted:
                conversations.complete_exchange(db, pending=pending, assistant_response=service.stream_response(prepared, content))
            elif not persisted:
                db.commit()
            raise
        except Exception:
            if content.strip():
                try:
                    conversations.complete_exchange(db, pending=pending, assistant_response=service.stream_response(prepared, content)); persisted = True
                except Exception:
                    conversations.rollback_exchange(db)
            elif not persisted:
                conversations.rollback_exchange(db)
            if not await request.is_disconnected():
                yield _stream_event({"type": "error", "message": "Le flux de réponse a été interrompu."})
        finally:
            logger.info("[ASSISTANT] generation_completed conversation_id=%s chars=%s", pending.conversation.id, len(content))
            logger.info("assistant_stream_completed conversation_id=%s duration_ms=%s chars=%s completed=%s", pending.conversation.id, int((perf_counter() - started) * 1000), len(content), completed)

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/conversations", response_model=AssistantConversationListOut, summary="List assistant conversations")
def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    archived: bool = Query(default=False),
    teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
    conversations: AssistantConversationService = Depends(get_assistant_conversation_service),
):
    summaries, total = conversations.list_owned_conversations(
        db, user_id=teacher.id, limit=limit, offset=offset, archived=archived,
    )
    return {
        "items": [
            {
                "id": item.conversation.id,
                "title": item.conversation.title,
                "created_at": item.conversation.created_at,
                "updated_at": item.conversation.updated_at,
                "archived_at": item.conversation.archived_at,
                "message_count": item.message_count,
            }
            for item in summaries
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/conversations/{conversation_id}",
    response_model=AssistantConversationDetailOut,
    summary="Get assistant conversation",
)
def get_conversation(
    conversation_id: int,
    teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
    conversations: AssistantConversationService = Depends(get_assistant_conversation_service),
):
    try:
        conversation = conversations.get_owned_conversation_with_messages(
            db, user_id=teacher.id, conversation_id=conversation_id,
        )
    except AssistantConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    return _detail_data(conversation)


@router.patch("/conversations/{conversation_id}/messages/{message_id}/regenerate", response_model=AssistantChatOut)
def regenerate_latest_user_message(
    conversation_id: int, message_id: int, payload: AssistantMessageRegenerateIn,
    teacher: User = Depends(require_teacher), db: Session = Depends(get_db),
    service: AssistantChatService = Depends(get_assistant_chat_service),
    conversations: AssistantConversationService = Depends(get_assistant_conversation_service),
):
    try:
        conversation, _target = conversations.get_editable_latest_user_message(
            db, user_id=teacher.id, conversation_id=conversation_id, message_id=message_id,
        )
        history = conversations.get_history_before_message(db, conversation_id=conversation.id, message_id=message_id)
        response = service.answer(db, AssistantChatRequest(message=payload.message), history=history)
        exchange = conversations.replace_latest_user_exchange(
            db, user_id=teacher.id, conversation_id=conversation.id, message_id=message_id,
            user_content=payload.message, assistant_response=response,
        )
    except AssistantConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    except AssistantConversationPersistenceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "ASSISTANT_EDIT_CONFLICT", "message": "Impossible de modifier ce message."}) from exc
    except AssistantChatValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail={"code": "ASSISTANT_VALIDATION_ERROR", "message": str(exc)}) from exc
    except AssistantChatServiceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": exc.code, "message": "Le service d’IA est momentanément indisponible."}) from exc
    return {**asdict(response), **asdict(exchange)}


@router.patch(
    "/conversations/{conversation_id}/archive",
    response_model=AssistantConversationDetailOut,
    summary="Archive assistant conversation",
)
def archive_conversation(
    conversation_id: int,
    teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
    conversations: AssistantConversationService = Depends(get_assistant_conversation_service),
):
    try:
        conversation = conversations.archive_owned_conversation(
            db, user_id=teacher.id, conversation_id=conversation_id,
        )
    except AssistantConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    except AssistantConversationPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "ASSISTANT_PERSISTENCE_ERROR", "message": "Impossible d’archiver cette conversation."},
        ) from exc
    return _detail_data(conversations.get_owned_conversation_with_messages(
        db, user_id=teacher.id, conversation_id=conversation.id,
    ))


@router.patch(
    "/conversations/{conversation_id}/restore",
    response_model=AssistantConversationDetailOut,
    summary="Restore assistant conversation",
)
def restore_conversation(
    conversation_id: int,
    teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
    conversations: AssistantConversationService = Depends(get_assistant_conversation_service),
):
    try:
        conversation = conversations.restore_owned_conversation(
            db, user_id=teacher.id, conversation_id=conversation_id,
        )
    except AssistantConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    except AssistantConversationPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "ASSISTANT_PERSISTENCE_ERROR", "message": "Impossible de restaurer cette conversation."},
        ) from exc
    return _detail_data(conversations.get_owned_conversation_with_messages(
        db, user_id=teacher.id, conversation_id=conversation.id,
    ))


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete assistant conversation",
)
def delete_conversation(
    conversation_id: int,
    teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
    conversations: AssistantConversationService = Depends(get_assistant_conversation_service),
):
    try:
        conversations.delete_owned_conversation(db, user_id=teacher.id, conversation_id=conversation_id)
    except AssistantConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    except AssistantConversationPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "ASSISTANT_PERSISTENCE_ERROR", "message": "Impossible de supprimer cette conversation."},
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
