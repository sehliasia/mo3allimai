import pytest
from dataclasses import replace
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.models.assistant_conversation import AssistantConversation, AssistantMessage, AssistantMessageRole, AssistantMessageSource
from app.models.knowledge_document import KnowledgeDocument, KnowledgeDocumentStatus
from app.models.user import User, UserRole
from app.services.assistant_chat_service import (
    AssistantChatDiagnostics,
    AssistantChatResponse,
    AssistantChatSource,
    clean_internal_rag_references,
)
from app.services.assistant_conversation_service import (
    AssistantConversationNotFoundError,
    AssistantConversationPersistenceError,
    AssistantConversationService,
)


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'assistant-conversations.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    teacher = User(full_name="Teacher A", email="teacher@example.test", password_hash="hash", role=UserRole.teacher)
    session.add(teacher)
    session.flush()
    document = KnowledgeDocument(
        title="Lesson plans", original_filename="lesson.pdf", stored_filename="lesson-internal.pdf",
        file_path="private/lesson.pdf", mime_type="application/pdf", file_size=1,
        status=KnowledgeDocumentStatus.ready, uploaded_by=teacher.id,
    )
    session.add(document)
    session.commit()
    try:
        yield session, teacher, document
    finally:
        session.close()
        engine.dispose()


def _response(document_id: int) -> AssistantChatResponse:
    return AssistantChatResponse(
        answer="Réponse fondée.",
        sources=[
            AssistantChatSource("cefr_structured", document_id, None, 76, 76, "Interaction orale générale"),
            AssistantChatSource("pedagogical_resource", document_id, "Lesson plans", 4, 5, None),
        ],
        diagnostics=AssistantChatDiagnostics(
            requested_cefr_level="A1", output_language="fr", retrieved_count=2, selected_count=2,
            source_count=2, requires_vision_count=0, warnings=[], provider_model="fake", finish_reason="stop",
        ),
    )


def test_new_exchange_is_atomic_owned_and_persists_ordered_safe_sources(db):
    session, teacher, document = db
    service = AssistantConversationService()

    exchange = service.persist_exchange(
        session, user_id=teacher.id, user_content="  Propose-moi\nune activité orale sur la famille. ",
        assistant_response=_response(document.id),
    )

    conversation = session.get(AssistantConversation, exchange.conversation_id)
    messages = session.scalars(select(AssistantMessage).where(
        AssistantMessage.conversation_id == conversation.id,
    ).order_by(AssistantMessage.id)).all()
    sources = session.scalars(select(AssistantMessageSource).where(
        AssistantMessageSource.message_id == exchange.assistant_message_id,
    ).order_by(AssistantMessageSource.source_order)).all()
    assert conversation.user_id == teacher.id
    assert conversation.title == "Activité orale – Famille"
    assert [(item.id, item.role, item.content) for item in messages] == [
        (exchange.user_message_id, AssistantMessageRole.user, "  Propose-moi\nune activité orale sur la famille. "),
        (exchange.assistant_message_id, AssistantMessageRole.assistant, "Réponse fondée."),
    ]
    assert [(item.source_order, item.source_type, item.document_id, item.descriptor_scale) for item in sources] == [
        (0, "cefr_structured", document.id, "Interaction orale générale"),
        (1, "pedagogical_resource", document.id, None),
    ]


def test_persistence_receives_the_already_clean_teacher_facing_answer(db):
    session, teacher, document = db
    service = AssistantConversationService()
    answer = clean_internal_rag_references("Exercice (voir Resource‑6) : أمي اسمها فاطمة.")

    exchange = service.persist_exchange(
        session,
        user_id=teacher.id,
        user_content="Question",
        assistant_response=replace(_response(document.id), answer=answer),
    )

    persisted = session.get(AssistantMessage, exchange.assistant_message_id)
    assert persisted.content == "Exercice : أمي اسمها فاطمة."
    assert "Resource‑6" not in persisted.content


def test_owned_conversation_can_be_reused_but_is_hidden_from_another_teacher(db):
    session, teacher, document = db
    service = AssistantConversationService()
    first = service.persist_exchange(
        session, user_id=teacher.id, user_content="Première question", assistant_response=_response(document.id),
    )

    owned = service.get_owned_conversation(session, user_id=teacher.id, conversation_id=first.conversation_id)
    second = service.persist_exchange(
        session, user_id=teacher.id, user_content="Deuxième question", assistant_response=_response(document.id), conversation=owned,
    )

    assert second.conversation_id == first.conversation_id
    assert owned.title == "Première question"
    assert len(session.scalars(select(AssistantMessage).where(
        AssistantMessage.conversation_id == first.conversation_id,
    )).all()) == 4
    with pytest.raises(AssistantConversationNotFoundError):
        service.get_owned_conversation(session, user_id=teacher.id + 1, conversation_id=first.conversation_id)


def test_archive_and_restore_are_owned_reversible_and_preserve_messages_and_sources(db):
    session, teacher, document = db
    service = AssistantConversationService()
    exchange = service.persist_exchange(
        session, user_id=teacher.id, user_content="Activité A1 sur la famille", assistant_response=_response(document.id),
    )

    archived = service.archive_owned_conversation(
        session, user_id=teacher.id, conversation_id=exchange.conversation_id,
    )
    active, active_total = service.list_owned_conversations(session, user_id=teacher.id, limit=10, offset=0)
    archived_rows, archived_total = service.list_owned_conversations(
        session, user_id=teacher.id, limit=10, offset=0, archived=True,
    )

    assert archived.archived_at is not None
    assert active == [] and active_total == 0
    assert [row.conversation.id for row in archived_rows] == [exchange.conversation_id] and archived_total == 1
    assert len(session.scalars(select(AssistantMessage).where(AssistantMessage.conversation_id == exchange.conversation_id)).all()) == 2
    assert len(session.scalars(select(AssistantMessageSource)).all()) == 2
    with pytest.raises(AssistantConversationNotFoundError):
        service.restore_owned_conversation(session, user_id=teacher.id + 1, conversation_id=exchange.conversation_id)

    restored = service.restore_owned_conversation(session, user_id=teacher.id, conversation_id=exchange.conversation_id)
    assert restored.archived_at is None


def test_persistence_failure_rolls_back_the_entire_exchange(db, monkeypatch):
    session, teacher, document = db
    service = AssistantConversationService()

    def fail_commit():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(session, "commit", fail_commit)
    with pytest.raises(AssistantConversationPersistenceError):
        service.persist_exchange(
            session, user_id=teacher.id, user_content="Question", assistant_response=_response(document.id),
        )

    assert session.scalars(select(AssistantConversation)).all() == []
    assert session.scalars(select(AssistantMessage)).all() == []
    assert session.scalars(select(AssistantMessageSource)).all() == []


def test_title_is_deterministic_whitespace_normalized_and_bounded():
    title = AssistantConversationService.derive_title("  un\n titre  " + "x" * 100)
    assert title.startswith("un titre ")
    assert len(title) == 80 and not title.endswith("…")


def test_title_uses_resolved_level_skill_and_subject_for_french_and_arabic_requests():
    assert AssistantConversationService.derive_title(
        "Propose-moi une activité orale A1 sur la famille."
    ) == "Activité orale A1 – Famille"
    assert AssistantConversationService.derive_title(
        "اقترح نشاطًا شفهيًا للمستوى A1 حول الأسرة."
    ) == "نشاط شفهي A1 – الأسرة"
    assert AssistantConversationService.derive_title(
        "Propose un jeu de rôle B1 sur un problème pendant un voyage."
    ) == "Jeu de rôle B1 – Problème pendant un voyage"
    assert AssistantConversationService.derive_title(
        "Prépare une activité de compréhension orale A2 sur les achats au marché."
    ) == "Compréhension orale A2 – Achats au marché"


def test_recent_history_is_owned_bounded_chronological_and_never_loads_sources(db):
    session, teacher, document = db
    service = AssistantConversationService()
    first = service.persist_exchange(
        session, user_id=teacher.id, user_content="Ancien message utilisateur", assistant_response=_response(document.id),
    )
    owned = service.get_owned_conversation(session, user_id=teacher.id, conversation_id=first.conversation_id)
    service.persist_exchange(
        session, user_id=teacher.id, user_content="Message utilisateur récent", assistant_response=_response(document.id), conversation=owned,
    )

    history = service.get_recent_history(
        session, user_id=teacher.id, conversation_id=first.conversation_id, max_messages=3, max_chars=1000,
    )

    assert [(item.role, item.content) for item in history] == [
        ("ASSISTANT", "Réponse fondée."),
        ("USER", "Message utilisateur récent"),
        ("ASSISTANT", "Réponse fondée."),
    ]
    trimmed = service.get_recent_history(
        session, user_id=teacher.id, conversation_id=first.conversation_id, max_messages=4, max_chars=25,
    )
    assert [(item.role, item.content) for item in trimmed] == [("ASSISTANT", "Réponse fondée.")]
    assert service.get_recent_history(
        session, user_id=teacher.id + 1, conversation_id=first.conversation_id, max_messages=8, max_chars=1000,
    ) == ()
