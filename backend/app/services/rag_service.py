"""Grounded text RAG orchestration; generation is isolated behind LLMProvider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.services.context_builder import ContextBuilder
from app.services.llm_provider import LLMProvider, LLMProviderError
from app.services.rag_prompt_builder import RAGPromptBuilder
from app.services.retrieval_service import RetrievalFilters, RetrievalService


class RAGServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, provider_message: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_message = provider_message


@dataclass(frozen=True)
class RAGSource:
    source_number: int
    document_id: int
    document_title: str
    page_start: int | None
    page_end: int | None
    chunk_ids: list[int]


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[RAGSource]
    used_chunk_ids: list[int]
    document_ids: list[int]
    has_requires_vision: bool
    warnings: list[str]
    model: str | None
    finish_reason: str | None
    output_token_count: int | None
    retrieval_metadata: dict[str, Any]


class RAGService:
    def __init__(self, *, retrieval: RetrievalService, context_builder: ContextBuilder, prompt_builder: RAGPromptBuilder, llm: LLMProvider, settings: Settings | None = None) -> None:
        self.retrieval, self.context_builder, self.prompt_builder, self.llm = retrieval, context_builder, prompt_builder, llm
        self.settings = settings or get_settings()

    @staticmethod
    def _insufficient(language: str) -> str:
        return {
            "Arabic": "المصادر المتاحة لا توفر معلومات كافية للإجابة بدقة عن هذا السؤال.",
            "French": "Les sources disponibles ne permettent pas de répondre précisément à cette question.",
        }.get(language, "The available sources do not provide enough information to answer this question accurately.")

    def answer_query(self, db: Session, query: str, *, document_ids: list[int] | None = None, language: str | None = None, top_k: int | None = None, use_reranker: bool | None = None) -> RAGResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise RAGServiceError("A non-empty query is required.")
        use_reranker = self.settings.rag_reranker_enabled if use_reranker is None else use_reranker
        retrieval_response = self.retrieval.search(
            db, normalized_query, top_k=top_k or self.settings.rag_retrieval_top_k, rerank=use_reranker,
            filters=RetrievalFilters(document_ids=document_ids),
        )
        context = self.context_builder.build(normalized_query, retrieval_response.results, db=db)
        sources = [RAGSource(block.source_number, block.document_id, block.document_title, block.page_start, block.page_end, block.chunk_ids) for block in context.source_blocks]
        metadata = {"retrieval_model": retrieval_response.model, "reranking_applied": retrieval_response.reranking_applied, "candidate_top_k": retrieval_response.candidate_top_k, "stale_references_skipped": retrieval_response.stale_references_skipped}
        answer_language = self.prompt_builder.normalize_output_language(language) if language else self.prompt_builder.detect_language(normalized_query)
        if not context.source_blocks:
            return RAGResponse(self._insufficient(answer_language), sources, [], [], False, context.warnings, None, None, None, metadata)
        prompt = self.prompt_builder.build(query=normalized_query, context=context, output_language=language)
        try:
            result = self.llm.generate(system_prompt=prompt.system_prompt, user_prompt=prompt.user_prompt, temperature=self.settings.rag_llm_temperature, max_tokens=self.settings.rag_llm_max_tokens)
        except LLMProviderError as exc:
            raise RAGServiceError(
                "Grounded answer generation failed.", status_code=exc.status_code, provider_message=exc.provider_message
            ) from exc
        return RAGResponse(
            result.text, sources, context.included_chunk_ids, list(dict.fromkeys(source.document_id for source in sources)),
            context.has_requires_vision, context.warnings, result.model, result.finish_reason, result.output_token_count, metadata,
        )
