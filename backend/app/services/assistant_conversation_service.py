"""Transactional persistence boundary for assistant exchanges.

This service deliberately has no RAG or provider dependency. It records only a
completed exchange returned by the assistant chat service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models.assistant_conversation import (
    AssistantConversation,
    AssistantMessage,
    AssistantMessageRole,
    AssistantMessageSource,
)
from app.services.assistant_chat_service import AssistantChatResponse
from app.services.assistant_chat_service import AssistantChatHistoryMessage
from app.services.chat_parameter_resolver import ChatParameterResolver


class AssistantConversationNotFoundError(LookupError):
    """Raised for a missing conversation or one owned by somebody else."""


class AssistantConversationPersistenceError(RuntimeError):
    """A safe, typed boundary for a failed database unit of work."""


@dataclass(frozen=True)
class PersistedAssistantExchange:
    conversation_id: int
    user_message_id: int
    assistant_message_id: int


@dataclass(frozen=True)
class PendingAssistantExchange:
    conversation: AssistantConversation
    user_message: AssistantMessage


@dataclass(frozen=True)
class AssistantConversationSummary:
    conversation: AssistantConversation
    message_count: int


class AssistantConversationService:
    """Enforce ownership and atomically write one successful assistant exchange."""

    _TITLE_LIMIT = 80

    _SKILL_TITLES = {
        "fr": {"speaking": "Activité orale", "listening": "Compréhension orale", "reading": "Lecture", "writing": "Écriture"},
        "ar": {"speaking": "نشاط شفهي", "listening": "فهم شفهي", "reading": "قراءة", "writing": "كتابة"},
        "en": {"speaking": "Speaking activity", "listening": "Listening", "reading": "Reading", "writing": "Writing"},
        "es": {"speaking": "Actividad oral", "listening": "Comprensión oral", "reading": "Lectura", "writing": "Escritura"},
    }
    _ROLE_PLAY_TITLES = {"fr": "Jeu de rôle", "ar": "لعبة أدوار", "en": "Role play", "es": "Juego de rol"}
    _GENERIC_TITLES = {"fr": "Activité", "ar": "نشاط", "en": "Activity", "es": "Actividad"}

    @classmethod
    def _subject(cls, message: str, language: str) -> str | None:
        normalized = re.sub(r"\s+", " ", message).strip(" .?!؟")
        patterns = {
            "fr": (r"(?:sur|autour de|pour)\s+(.+)",),
            "ar": (r"(?:حول|عن)\s+(.+)",),
            "en": (r"(?:about|on|for)\s+(.+)",),
            "es": (r"(?:sobre|para)\s+(.+)",),
        }
        for pattern in patterns.get(language, ()):
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                subject = match.group(1).strip(" .?!؟")
                subject = re.sub(r"^(?:la |le |les |un |une |the |el |la )", "", subject, flags=re.IGNORECASE)
                return subject[:40].strip() or None
        return None

    @classmethod
    def derive_title(cls, message: str) -> str:
        normalized = re.sub(r"\s+", " ", message).strip()
        resolved = ChatParameterResolver().resolve(
            message=normalized, cefr_level=None, skills=(), language=None,
        )
        language = resolved.response_language or "fr"
        folded = normalized.casefold()
        is_role_play = any(marker in folded for marker in ("jeu de rôle", "role play", "role-play", "لعبة أدوار"))
        subject = cls._subject(normalized, language)
        if not is_role_play and resolved.cefr_level is None and subject is None:
            return normalized[: cls._TITLE_LIMIT].rstrip()
        title_skill = resolved.skills[0] if resolved.skills else (
            "speaking" if any(marker in folded for marker in ("نشاطًا شفهي", "نشاط شفهي")) else None
        )
        title = cls._ROLE_PLAY_TITLES[language] if is_role_play else (
            cls._SKILL_TITLES[language].get(title_skill)
            if title_skill else cls._GENERIC_TITLES[language]
        )
        if resolved.cefr_level:
            title += f" {resolved.cefr_level}"
        if subject:
            title += f" – {subject[:1].upper() + subject[1:]}"
        if title != cls._GENERIC_TITLES[language] or len(normalized) > cls._TITLE_LIMIT:
            return title[: cls._TITLE_LIMIT].rstrip()
        return normalized[: cls._TITLE_LIMIT].rstrip()

    @staticmethod
    def get_owned_conversation(
        db: Session, *, user_id: int, conversation_id: int,
    ) -> AssistantConversation:
        conversation = db.scalar(select(AssistantConversation).where(
            AssistantConversation.id == conversation_id,
            AssistantConversation.user_id == user_id,
        ))
        if conversation is None:
            # Deliberately indistinguishable from absent, so ownership is not leaked.
            raise AssistantConversationNotFoundError("Conversation not found.")
        return conversation

    def list_owned_conversations(
        self, db: Session, *, user_id: int, limit: int, offset: int, archived: bool = False,
    ) -> tuple[list[AssistantConversationSummary], int]:
        """List only lightweight owned summaries; message bodies are never loaded."""
        message_count = func.count(AssistantMessage.id).label("message_count")
        rows = db.execute(
            select(AssistantConversation, message_count)
            .outerjoin(AssistantMessage, AssistantMessage.conversation_id == AssistantConversation.id)
            .where(
                AssistantConversation.user_id == user_id,
                AssistantConversation.archived_at.is_not(None) if archived else AssistantConversation.archived_at.is_(None),
            )
            .group_by(AssistantConversation.id)
            .order_by(AssistantConversation.updated_at.desc(), AssistantConversation.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        total = db.scalar(select(func.count()).select_from(AssistantConversation).where(
            AssistantConversation.user_id == user_id,
            AssistantConversation.archived_at.is_not(None) if archived else AssistantConversation.archived_at.is_(None),
        )) or 0
        return [AssistantConversationSummary(conversation, int(count)) for conversation, count in rows], int(total)

    def archive_owned_conversation(self, db: Session, *, user_id: int, conversation_id: int) -> AssistantConversation:
        conversation = self.get_owned_conversation(db, user_id=user_id, conversation_id=conversation_id)
        try:
            conversation.archived_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(conversation)
            return conversation
        except Exception as exc:
            db.rollback()
            raise AssistantConversationPersistenceError("Unable to archive assistant conversation.") from exc

    def restore_owned_conversation(self, db: Session, *, user_id: int, conversation_id: int) -> AssistantConversation:
        conversation = self.get_owned_conversation(db, user_id=user_id, conversation_id=conversation_id)
        try:
            conversation.archived_at = None
            db.commit()
            db.refresh(conversation)
            return conversation
        except Exception as exc:
            db.rollback()
            raise AssistantConversationPersistenceError("Unable to restore assistant conversation.") from exc

    def get_owned_conversation_with_messages(
        self, db: Session, *, user_id: int, conversation_id: int,
    ) -> AssistantConversation:
        """Fetch the complete persisted history in bounded eager-load queries."""
        conversation = db.scalar(
            select(AssistantConversation)
            .options(selectinload(AssistantConversation.messages).selectinload(AssistantMessage.sources))
            .where(
                AssistantConversation.id == conversation_id,
                AssistantConversation.user_id == user_id,
            )
        )
        if conversation is None:
            raise AssistantConversationNotFoundError("Conversation not found.")
        return conversation

    def get_recent_history(
        self,
        db: Session,
        *,
        user_id: int,
        conversation_id: int,
        max_messages: int,
        max_chars: int,
    ) -> tuple[AssistantChatHistoryMessage, ...]:
        """Read a bounded owned history without loading citations or knowledge rows.

        The latest ``max_messages`` rows are selected, then the oldest are
        discarded until their complete contents fit ``max_chars``. Messages are
        never sliced, and the surviving window is chronological.
        """
        rows = db.execute(
            select(AssistantMessage.role, AssistantMessage.content)
            .join(AssistantConversation, AssistantMessage.conversation_id == AssistantConversation.id)
            .where(
                AssistantConversation.id == conversation_id,
                AssistantConversation.user_id == user_id,
            )
            .order_by(AssistantMessage.created_at.desc(), AssistantMessage.id.desc())
            .limit(max_messages)
        ).all()
        messages = [
            AssistantChatHistoryMessage(
                role=role.value if hasattr(role, "value") else role,
                content=content,
            )
            for role, content in reversed(rows)
        ]
        while messages and sum(len(message.content) for message in messages) > max_chars:
            messages.pop(0)
        return tuple(messages)

    def get_history_before_message(self, db: Session, *, conversation_id: int, message_id: int) -> tuple[AssistantChatHistoryMessage, ...]:
        rows = db.execute(select(AssistantMessage.role, AssistantMessage.content).where(
            AssistantMessage.conversation_id == conversation_id, AssistantMessage.id < message_id,
        ).order_by(AssistantMessage.id)).all()
        return tuple(AssistantChatHistoryMessage(role=role.value if hasattr(role, "value") else role, content=content) for role, content in rows)

    def replace_latest_user_exchange(
        self, db: Session, *, user_id: int, conversation_id: int, message_id: int,
        user_content: str, assistant_response: AssistantChatResponse,
    ) -> PersistedAssistantExchange:
        conversation, target = self.get_editable_latest_user_message(db, user_id=user_id, conversation_id=conversation_id, message_id=message_id)
        ordered = sorted(conversation.messages, key=lambda item: item.id)
        try:
            for message in [item for item in ordered if item.id > target.id]:
                db.delete(message)
            target.content = user_content
            db.flush()
            assistant_message = AssistantMessage(conversation_id=conversation.id, role=AssistantMessageRole.assistant, content=assistant_response.answer)
            db.add(assistant_message)
            db.flush()
            db.add_all([AssistantMessageSource(message_id=assistant_message.id, source_type=source.source_type, document_id=source.document_id, document_title=source.document_title, page_start=source.page_start, page_end=source.page_end, descriptor_scale=source.cefr_scale, source_order=index) for index, source in enumerate(assistant_response.sources)])
            conversation.updated_at = datetime.now(timezone.utc)
            db.commit()
            return PersistedAssistantExchange(conversation.id, target.id, assistant_message.id)
        except Exception as exc:
            db.rollback()
            raise AssistantConversationPersistenceError("Unable to replace assistant exchange.") from exc

    def get_editable_latest_user_message(self, db: Session, *, user_id: int, conversation_id: int, message_id: int) -> tuple[AssistantConversation, AssistantMessage]:
        conversation = self.get_owned_conversation_with_messages(db, user_id=user_id, conversation_id=conversation_id)
        ordered = sorted(conversation.messages, key=lambda item: item.id)
        target = next((message for message in ordered if message.id == message_id), None)
        if target is None or target.role != AssistantMessageRole.user:
            raise AssistantConversationNotFoundError("Conversation message not found.")
        if any(message.role == AssistantMessageRole.user for message in ordered if message.id > target.id):
            raise AssistantConversationPersistenceError("Only the latest user message can be edited.")
        return conversation, target

    def delete_owned_conversation(self, db: Session, *, user_id: int, conversation_id: int) -> None:
        """Delete only an owned conversation; database cascades leave knowledge data intact."""
        conversation = self.get_owned_conversation(db, user_id=user_id, conversation_id=conversation_id)
        try:
            db.delete(conversation)
            db.commit()
        except Exception as exc:
            db.rollback()
            raise AssistantConversationPersistenceError("Unable to delete assistant conversation.") from exc

    def begin_exchange(
        self,
        db: Session,
        *,
        user_id: int,
        user_content: str,
        conversation: AssistantConversation | None = None,
    ) -> PendingAssistantExchange:
        """Stage the new conversation/user message without committing it."""
        try:
            if conversation is None:
                conversation = AssistantConversation(user_id=user_id, title=self.derive_title(user_content))
                db.add(conversation)
                db.flush()

            user_message = AssistantMessage(
                conversation_id=conversation.id,
                role=AssistantMessageRole.user,
                content=user_content,
            )
            db.add(user_message)
            db.flush()
            return PendingAssistantExchange(conversation=conversation, user_message=user_message)
        except Exception as exc:
            db.rollback()
            raise AssistantConversationPersistenceError("Unable to persist assistant exchange.") from exc

    def complete_exchange(
        self,
        db: Session,
        *,
        pending: PendingAssistantExchange,
        assistant_response: AssistantChatResponse,
    ) -> PersistedAssistantExchange:
        """Add the completed assistant answer/sources and commit the staged exchange."""
        try:
            assistant_message = AssistantMessage(
                conversation_id=pending.conversation.id,
                role=AssistantMessageRole.assistant,
                content=assistant_response.answer,
            )
            db.add(assistant_message)
            db.flush()

            db.add_all([
                AssistantMessageSource(
                    message_id=assistant_message.id,
                    source_type=source.source_type,
                    document_id=source.document_id,
                    document_title=source.document_title,
                    page_start=source.page_start,
                    page_end=source.page_end,
                    descriptor_scale=source.cefr_scale,
                    source_order=source_order,
                )
                for source_order, source in enumerate(assistant_response.sources)
            ])
            pending.conversation.updated_at = datetime.now(timezone.utc)
            db.commit()
            return PersistedAssistantExchange(
                conversation_id=pending.conversation.id,
                user_message_id=pending.user_message.id,
                assistant_message_id=assistant_message.id,
            )
        except Exception as exc:
            db.rollback()
            raise AssistantConversationPersistenceError("Unable to persist assistant exchange.") from exc

    def persist_exchange(
        self,
        db: Session,
        *,
        user_id: int,
        user_content: str,
        assistant_response: AssistantChatResponse,
        conversation: AssistantConversation | None = None,
    ) -> PersistedAssistantExchange:
        """Convenience composition used by non-HTTP callers and focused tests."""
        pending = self.begin_exchange(
            db, user_id=user_id, user_content=user_content, conversation=conversation,
        )
        return self.complete_exchange(db, pending=pending, assistant_response=assistant_response)

    @staticmethod
    def rollback_exchange(db: Session) -> None:
        """Discard a staged message when answer generation fails."""
        db.rollback()
