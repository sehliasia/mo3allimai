import json

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from app.api.dependencies import get_current_user, require_teacher
from app.api.routes.assistant import get_assistant_chat_service, get_assistant_conversation_service
from app.database.session import get_db
from app.main import app
from app.services.assistant_chat_service import (
    AssistantChatHistoryMessage,
    AssistantChatDiagnostics,
    AssistantChatRequest,
    AssistantChatResponse,
    AssistantChatService,
    AssistantChatServiceError,
    AssistantChatSource,
    PreparedAssistantStream,
)
from app.services.assistant_conversation_service import (
    AssistantConversationNotFoundError,
    AssistantConversationPersistenceError,
    PersistedAssistantExchange,
)
from app.models.user import UserRole


class FakeAssistantService:
    def __init__(self, response: AssistantChatResponse | None = None, error: Exception | None = None) -> None:
        self.requests: list[AssistantChatRequest] = []
        self.histories = []
        self.response = response or _response()
        self.error = error

    def answer(self, _db, request: AssistantChatRequest, *, history=()) -> AssistantChatResponse:
        # The HTTP adapter must use the core validation contract rather than duplicate it.
        request = AssistantChatService._validate(request)
        self.requests.append(request)
        self.histories.append(history)
        if self.error:
            raise self.error
        return self.response


class FakeStreamingAssistantService(FakeAssistantService):
    def __init__(self, chunks=(), error: Exception | None = None) -> None:
        super().__init__()
        self.chunks = list(chunks)
        self.error = error
        self.prepared = []

    def prepare_stream(self, _db, request: AssistantChatRequest, *, history=()):
        request = AssistantChatService._validate(request)
        self.requests.append(request)
        self.histories.append(history)
        prepared = PreparedAssistantStream(
            request=request, system_prompt="system", user_prompt="user",
            sources=self.response.sources, diagnostics=self.response.diagnostics,
        )
        self.prepared.append(prepared)
        return prepared

    async def stream_answer(self, _prepared):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def stream_response(self, prepared, content):
        return AssistantChatResponse(content, prepared.sources, prepared.diagnostics)


class FakeConversationService:
    def __init__(self, *, error: Exception | None = None, history=()) -> None:
        self.error = error
        self.history = history
        self.lookups = []
        self.exchanges = []
        self.rollbacks = 0

    def get_owned_conversation(self, _db, *, user_id, conversation_id):
        self.lookups.append((user_id, conversation_id))
        if self.error:
            raise self.error
        return SimpleNamespace(id=conversation_id, user_id=user_id)

    def begin_exchange(self, _db, **kwargs):
        self.exchanges.append(kwargs)
        if self.error:
            raise self.error
        conversation = kwargs["conversation"] or SimpleNamespace(id=12, user_id=kwargs["user_id"])
        return SimpleNamespace(conversation=conversation, user_message=SimpleNamespace(id=100))

    def get_recent_history(self, _db, **_kwargs):
        return self.history

    def complete_exchange(self, _db, **kwargs):
        self.exchanges.append(kwargs)
        if self.error:
            raise self.error
        return PersistedAssistantExchange(12, 100, 101)

    def rollback_exchange(self, _db):
        self.rollbacks += 1


def _response(*, answer: str = "Réponse fondée.") -> AssistantChatResponse:
    return AssistantChatResponse(
        answer=answer,
        sources=[AssistantChatSource(
            source_type="pedagogical_resource", document_id=15, document_title="Lesson plans",
            page_start=4, page_end=4, cefr_scale=None,
        )],
        diagnostics=AssistantChatDiagnostics(
            requested_cefr_level="A1", output_language="fr", retrieved_count=3,
            selected_count=2, source_count=1, requires_vision_count=0,
            warnings=[], provider_model="fake-llm", finish_reason="stop",
        ),
    )


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    app.dependency_overrides[require_teacher] = lambda: SimpleNamespace(id=7)
    app.dependency_overrides[get_db] = lambda: object()
    yield TestClient(app)
    app.dependency_overrides.clear()


def _use_service(service: FakeAssistantService) -> None:
    app.dependency_overrides[get_assistant_chat_service] = lambda: service


def _use_conversations(service: FakeConversationService) -> None:
    app.dependency_overrides[get_assistant_conversation_service] = lambda: service


def test_chat_is_registered_protected_and_returns_only_safe_grounded_fields(client):
    service = FakeAssistantService()
    _use_service(service)
    conversations = FakeConversationService()
    _use_conversations(conversations)

    response = client.post("/api/assistant/chat", json={
        "message": "Quels objectifs pour une leçon ?", "cefr_level": "A1",
        "skills": ["speaking"], "language": "fr", "top_k": 8,
    })

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": 12, "user_message_id": 100, "assistant_message_id": 101,
        "answer": "Réponse fondée.",
        "sources": [{
            "source_type": "pedagogical_resource", "document_id": 15,
            "document_title": "Lesson plans", "page_start": 4, "page_end": 4,
            "cefr_scale": None,
        }],
        "diagnostics": {
            "requested_cefr_level": "A1", "output_language": "fr", "retrieved_count": 3,
            "selected_count": 2, "source_count": 1, "requires_vision_count": 0,
            "warnings": [], "provider_model": "fake-llm", "finish_reason": "stop",
            "history_messages_used": 0, "history_chars_used": 0,
        },
    }
    assert service.requests == [AssistantChatRequest(
        message="Quels objectifs pour une leçon ?", cefr_level="A1", skills=("speaking",),
        language="fr", topic=None, objective=None, top_k=8,
    )]
    assert conversations.exchanges[0]["user_id"] == 7
    assert conversations.exchanges[0]["conversation"] is None


def test_chat_requires_a_teacher_bearer_authentication(client):
    app.dependency_overrides.pop(require_teacher)
    _use_service(FakeAssistantService())
    _use_conversations(FakeConversationService())

    response = client.post("/api/assistant/chat", json={"message": "Question"})

    assert response.status_code == 401


@pytest.mark.parametrize("payload", [
    {"message": ""},
    {"message": "Question", "cefr_level": "A0"},
    {"message": "Question", "skills": ["invented"]},
    {"message": "Question", "top_k": 21},
    {"message": "Question", "user_id": 99},
])
def test_chat_rejects_invalid_or_out_of_scope_input(client, payload):
    _use_service(FakeAssistantService())
    _use_conversations(FakeConversationService())
    response = client.post("/api/assistant/chat", json=payload)
    assert response.status_code == 422


def test_chat_accepts_a_request_without_a_cefr_level(client):
    service = FakeAssistantService()
    _use_service(service)
    _use_conversations(FakeConversationService())

    response = client.post("/api/assistant/chat", json={"message": "Aidez-moi à préparer une activité", "language": "ar"})

    assert response.status_code == 200
    assert service.requests[0].cefr_level is None
    assert service.requests[0].language == "ar"


def test_chat_returns_the_service_insufficient_knowledge_response(client):
    service = FakeAssistantService(response=_response(answer="Les sources disponibles ne permettent pas de répondre précisément à cette question."))
    _use_service(service)
    _use_conversations(FakeConversationService())

    response = client.post("/api/assistant/chat", json={"message": "Question inconnue"})

    assert response.status_code == 200
    assert response.json()["answer"].startswith("Les sources disponibles")


def test_chat_json_round_trip_preserves_markdown_without_double_escaping(client):
    answer = "**Titre**\n\n### Vocabulaire\n\n| Arabe | Français |\n| --- | --- |\n| أمي | Ma mère |"
    _use_service(FakeAssistantService(response=_response(answer=answer)))
    _use_conversations(FakeConversationService())

    response = client.post("/api/assistant/chat", json={"message": "Question"})

    assert response.status_code == 200
    assert response.json()["answer"] == answer


def test_chat_hides_provider_details_when_the_service_is_unavailable(client):
    _use_service(FakeAssistantService(error=AssistantChatServiceError(
        "ASSISTANT_PROVIDER_ERROR", "secret provider trace", status_code=503,
    )))
    conversations = FakeConversationService()
    _use_conversations(conversations)

    response = client.post("/api/assistant/chat", json={"message": "Question"})

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "ASSISTANT_PROVIDER_ERROR",
        "message": "Le service d’IA est momentanément indisponible.",
    }
    assert "secret" not in response.text
    assert conversations.rollbacks == 1
    assert len(conversations.exchanges) == 1


def test_admin_access_follows_the_existing_teacher_only_rule(client):
    app.dependency_overrides.pop(require_teacher)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(role=UserRole.admin)
    _use_conversations(FakeConversationService())
    _use_service(FakeAssistantService())

    response = client.post("/api/assistant/chat", json={"message": "Question"})

    assert response.status_code == 403


def test_chat_appends_only_to_a_conversation_owned_by_the_current_teacher(client):
    conversations = FakeConversationService()
    _use_conversations(conversations)
    service = FakeAssistantService()
    _use_service(service)

    response = client.post("/api/assistant/chat", json={"message": "Question", "conversation_id": 12})

    assert response.status_code == 200
    assert conversations.lookups == [(7, 12)]
    assert conversations.exchanges[0]["conversation"].id == 12
    assert service.histories == [()]


def test_chat_loads_only_previous_owned_history_before_staging_current_message(client):
    previous = (AssistantChatHistoryMessage("USER", "Activité A1 sur la famille"),)
    conversations = FakeConversationService(history=previous)
    service = FakeAssistantService()
    _use_conversations(conversations)
    _use_service(service)

    response = client.post("/api/assistant/chat", json={"message": "Rends-la plus simple.", "conversation_id": 12})

    assert response.status_code == 200
    assert service.histories == [previous]
    assert all(item.content != "Rends-la plus simple." for item in service.histories[0])


def test_chat_hides_nonexistent_or_other_users_conversation(client):
    _use_conversations(FakeConversationService(error=AssistantConversationNotFoundError()))
    _use_service(FakeAssistantService())

    response = client.post("/api/assistant/chat", json={"message": "Question", "conversation_id": 999})

    assert response.status_code == 404


def test_persistence_failure_returns_safe_error_after_rollback(client):
    _use_conversations(FakeConversationService(error=AssistantConversationPersistenceError()))
    _use_service(FakeAssistantService())

    response = client.post("/api/assistant/chat", json={"message": "Question"})

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "ASSISTANT_PERSISTENCE_ERROR",
        "message": "Impossible d’enregistrer cet échange.",
    }


def test_openapi_exposes_the_authenticated_chat_operation(client):
    schema = app.openapi()
    operation = schema["paths"]["/api/assistant/chat"]["post"]
    assert operation["security"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]


def _stream_events(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_chat_stream_emits_ordered_deltas_and_persists_once_at_completion(client):
    service = FakeStreamingAssistantService(["Bon", "jour"])
    conversations = FakeConversationService()
    _use_service(service)
    _use_conversations(conversations)

    response = client.post("/api/assistant/chat/stream", json={"message": "Question"})

    assert response.status_code == 200
    assert _stream_events(response)[:3] == [
        {"type": "start", "conversation_id": 12, "message_id": 100},
        {"type": "delta", "content": "Bon"},
        {"type": "delta", "content": "jour"},
    ]
    assert _stream_events(response)[-1]["type"] == "done"
    assert len([item for item in conversations.exchanges if "assistant_response" in item]) == 1


def test_chat_stream_returns_a_safe_event_and_does_not_persist_an_empty_assistant_on_provider_error(client):
    service = FakeStreamingAssistantService([AssistantChatServiceError("ASSISTANT_PROVIDER_ERROR", "secret")])
    conversations = FakeConversationService()
    _use_service(service)
    _use_conversations(conversations)

    response = client.post("/api/assistant/chat/stream", json={"message": "Question"})

    events = _stream_events(response)
    assert response.status_code == 200
    assert events[-1] == {"type": "error", "message": "Le flux de réponse a été interrompu."}
    assert "secret" not in response.text
    assert not [item for item in conversations.exchanges if "assistant_response" in item]
    assert conversations.rollbacks == 1
