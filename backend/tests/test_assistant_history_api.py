from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.api.dependencies import get_current_user, require_teacher
from app.api.routes.assistant import get_assistant_conversation_service
from app.database.base import Base
from app.database.session import get_db
from app.main import app
from app.models.assistant_conversation import AssistantConversation, AssistantMessage, AssistantMessageSource
from app.models.knowledge_document import KnowledgeDocument, KnowledgeDocumentStatus
from app.models.user import User, UserRole
from app.services.assistant_chat_service import AssistantChatDiagnostics, AssistantChatResponse, AssistantChatSource
from app.services.assistant_conversation_service import AssistantConversationService


def _response(document_id: int) -> AssistantChatResponse:
    return AssistantChatResponse(
        answer="Réponse sauvegardée.",
        sources=[
            AssistantChatSource("pedagogical_resource", document_id, "Lesson plans", 5, 5, None),
            AssistantChatSource("cefr_structured", document_id, None, 76, 76, "Interaction orale générale"),
        ],
        diagnostics=AssistantChatDiagnostics(
            requested_cefr_level="A1", output_language="fr", retrieved_count=2, selected_count=2,
            source_count=2, requires_vision_count=0, warnings=[], provider_model="fake", finish_reason="stop",
        ),
    )


@pytest.fixture
def history_client(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'assistant-history.db'}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    teacher = User(full_name="Teacher A", email="teacher-a@example.test", password_hash="hash", role=UserRole.teacher)
    other_teacher = User(full_name="Teacher B", email="teacher-b@example.test", password_hash="hash", role=UserRole.teacher)
    session.add_all([teacher, other_teacher])
    session.flush()
    document = KnowledgeDocument(
        title="Lesson plans", original_filename="lesson.pdf", stored_filename="history-lesson.pdf",
        file_path="private/history-lesson.pdf", mime_type="application/pdf", file_size=1,
        status=KnowledgeDocumentStatus.ready, uploaded_by=teacher.id,
    )
    session.add(document)
    session.commit()

    app.dependency_overrides.clear()
    app.dependency_overrides[require_teacher] = lambda: teacher
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_assistant_conversation_service] = AssistantConversationService
    try:
        yield TestClient(app), session, teacher, other_teacher, document
    finally:
        app.dependency_overrides.clear()
        session.close()
        engine.dispose()


def _exchange(session, teacher_id, document_id, message):
    return AssistantConversationService().persist_exchange(
        session, user_id=teacher_id, user_content=message, assistant_response=_response(document_id),
    )


def test_list_returns_only_owned_lightweight_summaries_with_deterministic_pagination(history_client):
    client, session, teacher, other_teacher, document = history_client
    older = _exchange(session, teacher.id, document.id, "Conversation plus ancienne")
    newer = _exchange(session, teacher.id, document.id, "Conversation plus récente")
    _exchange(session, other_teacher.id, document.id, "Conversation privée d'un autre professeur")
    session.get(AssistantConversation, older.conversation_id).updated_at = datetime(2026, 8, 1, 9, 0, 0)
    session.get(AssistantConversation, newer.conversation_id).updated_at = datetime(2026, 8, 2, 9, 0, 0)
    session.commit()

    response = client.get("/api/assistant/conversations?limit=1&offset=0")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2 and body["limit"] == 1 and body["offset"] == 0
    assert body["items"] == [{
        "id": newer.conversation_id,
        "title": "Conversation plus récente",
        "created_at": body["items"][0]["created_at"],
        "updated_at": body["items"][0]["updated_at"],
        "archived_at": None,
        "message_count": 2,
    }]
    assert "user_id" not in body["items"][0]


def test_empty_history_is_a_valid_empty_response(history_client):
    client, _session, _teacher, _other_teacher, _document = history_client
    response = client.get("/api/assistant/conversations")
    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_detail_reconstructs_messages_and_sources_from_postgres_only(history_client):
    client, session, teacher, _other_teacher, document = history_client
    exchange = _exchange(session, teacher.id, document.id, "Question enregistrée")

    response = client.get(f"/api/assistant/conversations/{exchange.conversation_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == exchange.conversation_id
    assert [(message["id"], message["role"], message["content"]) for message in body["messages"]] == [
        (exchange.user_message_id, "USER", "Question enregistrée"),
        (exchange.assistant_message_id, "ASSISTANT", "Réponse sauvegardée."),
    ]
    assert body["messages"][0]["sources"] == []
    assert [(source["source_type"], source["page_start"], source["descriptor_scale"]) for source in body["messages"][1]["sources"]] == [
        ("pedagogical_resource", 5, None),
        ("cefr_structured", 76, "Interaction orale générale"),
    ]
    assert "chunk_id" not in response.text and "user_id" not in response.text


@pytest.mark.parametrize("conversation_id", [999999, None])
def test_foreign_and_nonexistent_detail_share_the_safe_404(history_client, conversation_id):
    client, session, teacher, other_teacher, document = history_client
    foreign = _exchange(session, other_teacher.id, document.id, "Privé")
    requested_id = foreign.conversation_id if conversation_id is None else conversation_id

    response = client.get(f"/api/assistant/conversations/{requested_id}")

    assert response.status_code == 404 and response.json() == {"detail": "Conversation not found."}


def test_delete_owned_conversation_cascades_only_to_messages_and_sources(history_client):
    client, session, teacher, _other_teacher, document = history_client
    exchange = _exchange(session, teacher.id, document.id, "À supprimer")

    response = client.delete(f"/api/assistant/conversations/{exchange.conversation_id}")

    assert response.status_code == 204 and response.content == b""
    assert session.get(AssistantConversation, exchange.conversation_id) is None
    assert session.scalars(select(AssistantMessage)).all() == []
    assert session.scalars(select(AssistantMessageSource)).all() == []
    assert session.get(KnowledgeDocument, document.id) is not None


def test_foreign_and_nonexistent_delete_share_the_safe_404(history_client):
    client, session, _teacher, other_teacher, document = history_client
    foreign = _exchange(session, other_teacher.id, document.id, "Privé")

    foreign_response = client.delete(f"/api/assistant/conversations/{foreign.conversation_id}")
    missing_response = client.delete("/api/assistant/conversations/999999")

    assert foreign_response.status_code == missing_response.status_code == 404
    assert foreign_response.json() == missing_response.json() == {"detail": "Conversation not found."}


def test_archive_restore_and_listing_are_owned_and_keep_persisted_history(history_client):
    client, session, teacher, other_teacher, document = history_client
    exchange = _exchange(session, teacher.id, document.id, "À archiver")
    foreign = _exchange(session, other_teacher.id, document.id, "Privé")

    response = client.patch(f"/api/assistant/conversations/{exchange.conversation_id}/archive")
    assert response.status_code == 200 and response.json()["archived_at"] is not None
    assert client.get("/api/assistant/conversations").json()["items"] == []
    archived = client.get("/api/assistant/conversations?archived=true").json()
    assert [item["id"] for item in archived["items"]] == [exchange.conversation_id]
    assert len(response.json()["messages"]) == 2
    assert client.patch(f"/api/assistant/conversations/{foreign.conversation_id}/archive").status_code == 404

    restored = client.patch(f"/api/assistant/conversations/{exchange.conversation_id}/restore")
    assert restored.status_code == 200 and restored.json()["archived_at"] is None
    assert [item["id"] for item in client.get("/api/assistant/conversations").json()["items"]] == [exchange.conversation_id]


def test_history_routes_keep_existing_teacher_authentication_policy(history_client):
    client, _session, _teacher, _other_teacher, _document = history_client
    app.dependency_overrides.pop(require_teacher)
    unauthenticated = client.get("/api/assistant/conversations")
    assert unauthenticated.status_code == 401

    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(role=UserRole.admin)
    admin = client.get("/api/assistant/conversations")
    assert admin.status_code == 403


def test_openapi_exposes_all_history_operations(history_client):
    client, _session, _teacher, _other_teacher, _document = history_client
    schema = app.openapi()
    assert "get" in schema["paths"]["/api/assistant/conversations"]
    assert "get" in schema["paths"]["/api/assistant/conversations/{conversation_id}"]
    assert "delete" in schema["paths"]["/api/assistant/conversations/{conversation_id}"]
    assert "patch" in schema["paths"]["/api/assistant/conversations/{conversation_id}/archive"]
    assert "patch" in schema["paths"]["/api/assistant/conversations/{conversation_id}/restore"]
