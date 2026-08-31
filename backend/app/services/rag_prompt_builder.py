"""Strict, source-grounded prompts for text-only pedagogical RAG."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.context_builder import RAGContext


@dataclass(frozen=True)
class RAGPrompt:
    system_prompt: str
    user_prompt: str
    answer_language: str


class RAGPromptBuilder:
    @staticmethod
    def normalize_output_language(language: str) -> str:
        return {"ar": "Arabic", "arabic": "Arabic", "fr": "French", "french": "French", "en": "English", "english": "English"}.get(language.casefold(), language)

    @staticmethod
    def detect_language(query: str) -> str:
        if any("\u0600" <= char <= "\u06ff" for char in query):
            return "Arabic"
        lowered = query.casefold()
        if any(token in lowered for token in ("quel", "quelle", "quels", "quelles", "comment", "pourquoi", " le ", " la ", " des ", "é", "à", "ç")):
            return "French"
        return "English"

    def build(self, *, query: str, context: RAGContext, output_language: str | None = None) -> RAGPrompt:
        answer_language = self.normalize_output_language(output_language) if output_language else self.detect_language(query)
        system = (
            "You are a grounded pedagogical assistant. Answer only from the supplied sources. "
            "Do not invent facts, citations, document identifiers, page numbers, or image details. "
            "If the sources are insufficient, say so clearly. Preserve pedagogical terminology. "
            f"Answer in {answer_language}."
        )
        if context.has_requires_vision:
            system += " Some sources include visual content that was not interpreted; use only their supplied text and do not infer image facts."
        return RAGPrompt(system_prompt=system, user_prompt=f"QUESTION:\n{query}\n\nSOURCES:\n{context.context_text}", answer_language=answer_language)
