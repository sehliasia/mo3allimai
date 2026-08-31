"""Public, single-turn HTTP schemas for the pedagogical assistant."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AssistantChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    conversation_id: int | None = Field(default=None, ge=1)
    cefr_level: str | None = None
    skills: list[str] = Field(default_factory=list)
    language: str | None = None
    topic: str | None = None
    objective: str | None = None
    top_k: int = 8
    mode: Literal["knowledge_base", "user_documents"] = "knowledge_base"
    document_ids: list[int] = Field(default_factory=list)


class AssistantMessageRegenerateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


class AssistantChatSourceOut(BaseModel):
    source_type: Literal["cefr_structured", "pedagogical_resource", "personal_document"]
    document_id: int
    document_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    cefr_scale: str | None = None


class AssistantChatDiagnosticsOut(BaseModel):
    requested_cefr_level: str | None = None
    output_language: str
    retrieved_count: int
    selected_count: int
    source_count: int
    requires_vision_count: int
    warnings: list[str]
    provider_model: str | None = None
    finish_reason: str | None = None
    history_messages_used: int = 0
    history_chars_used: int = 0


class AssistantChatOut(BaseModel):
    conversation_id: int
    user_message_id: int
    assistant_message_id: int
    answer: str
    sources: list[AssistantChatSourceOut]
    diagnostics: AssistantChatDiagnosticsOut


class AssistantConversationSummaryOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    message_count: int


class AssistantConversationListOut(BaseModel):
    items: list[AssistantConversationSummaryOut]
    total: int
    limit: int
    offset: int


class AssistantHistorySourceOut(BaseModel):
    source_type: Literal["cefr_structured", "pedagogical_resource"]
    document_id: int | None = None
    document_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    descriptor_scale: str | None = None


class AssistantHistoryMessageOut(BaseModel):
    id: int
    role: Literal["USER", "ASSISTANT"]
    content: str
    created_at: datetime
    sources: list[AssistantHistorySourceOut]


class AssistantConversationDetailOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    messages: list[AssistantHistoryMessageOut]
