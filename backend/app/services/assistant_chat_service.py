"""Single-turn, knowledge-base-only core for the future pedagogical assistant."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.services.llm_provider import LLMGenerationOptions, LLMProvider, LLMProviderError, LLMRetryPolicy
from app.services.arabic_linguistic_review_service import ArabicLinguisticReviewService
from app.services.chat_parameter_resolver import ChatParameterResolver
from app.services.pedagogical_knowledge_service import (
    PedagogicalContext,
    PedagogicalKnowledgeRequest,
    PedagogicalKnowledgeService,
    SUPPORTED_CEFR_LEVELS,
)
from app.services.personal_retrieval_service import PersonalDocumentAccessError, PersonalRetrievalService


class AssistantChatValidationError(ValueError):
    """An assistant request is incomplete or outside its bounded input contract."""


class AssistantChatServiceError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


# Keep this set intentionally narrow: models sometimes substitute a typographic
# dash in internal labels, but ordinary punctuation in teacher-facing prose must
# remain untouched.
_INTERNAL_RAG_DASH = r"[-\u2010\u2011\u2013\u2014]"
_INTERNAL_RAG_REFERENCE = (
    rf"(?:resource|ressource|cefr(?:{_INTERNAL_RAG_DASH}missing)?)"
    rf"\s*{_INTERNAL_RAG_DASH}\s*\d+"
)
_INTERNAL_RAG_PARENTHETICAL = re.compile(
    rf"\s*\((?:voir|cf\.?|see)\s+{_INTERNAL_RAG_REFERENCE}\)",
    re.IGNORECASE,
)
_INTERNAL_RAG_BRACKETED = re.compile(rf"[ \t]*\[{_INTERNAL_RAG_REFERENCE}\]", re.IGNORECASE)
_INTERNAL_RAG_CITATION_CUE = re.compile(
    rf"\b(?:voir|cf\.?|see)\s+{_INTERNAL_RAG_REFERENCE}\b", re.IGNORECASE,
)
_INTERNAL_RAG_BARE = re.compile(rf"\b{_INTERNAL_RAG_REFERENCE}\b", re.IGNORECASE)


def clean_internal_rag_references(answer: str) -> str:
    """Remove only internal registry labels from teacher-facing generated prose."""
    answer = _INTERNAL_RAG_PARENTHETICAL.sub("", answer)
    answer = _INTERNAL_RAG_BRACKETED.sub("", answer)
    answer = _INTERNAL_RAG_CITATION_CUE.sub("", answer)
    return _INTERNAL_RAG_BARE.sub("", answer)


@dataclass(frozen=True)
class AssistantChatRequest:
    message: str
    cefr_level: str | None = None
    skills: tuple[str, ...] = ()
    language: str | None = None
    topic: str | None = None
    objective: str | None = None
    top_k: int = 8
    mode: Literal["knowledge_base", "user_documents"] = "knowledge_base"
    document_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class AssistantChatSource:
    source_type: Literal["cefr_structured", "pedagogical_resource", "personal_document"]
    document_id: int
    document_title: str | None
    page_start: int | None
    page_end: int | None
    cefr_scale: str | None


@dataclass(frozen=True)
class AssistantChatHistoryMessage:
    """Trusted, persisted conversation memory; never accepted from the client."""

    role: Literal["USER", "ASSISTANT"]
    content: str


@dataclass(frozen=True)
class AssistantChatDiagnostics:
    requested_cefr_level: str | None
    output_language: str
    retrieved_count: int
    selected_count: int
    source_count: int
    requires_vision_count: int
    warnings: list[str]
    provider_model: str | None
    finish_reason: str | None
    history_messages_used: int = 0
    history_chars_used: int = 0


@dataclass(frozen=True)
class AssistantChatResponse:
    answer: str
    sources: list[AssistantChatSource]
    diagnostics: AssistantChatDiagnostics


@dataclass(frozen=True)
class PreparedAssistantStream:
    request: AssistantChatRequest
    system_prompt: str | None
    user_prompt: str | None
    sources: list[AssistantChatSource]
    diagnostics: AssistantChatDiagnostics
    immediate_answer: str | None = None


class AssistantChatPromptBuilder:
    _LANGUAGE_NAMES = {"ar": "Arabic", "fr": "French", "en": "English", "es": "Spanish"}

    @classmethod
    def language_name(cls, language: str) -> str:
        return cls._LANGUAGE_NAMES[language]

    def build(
        self,
        *,
        request: AssistantChatRequest,
        context: PedagogicalContext,
        history: tuple[AssistantChatHistoryMessage, ...] = (),
    ) -> tuple[str, str]:
        lines: list[str] = ["STRUCTURED CEFR KNOWLEDGE:"]
        for index, descriptor in enumerate(context.cefr_descriptors, start=1):
            label = f"CEFR-{index}"
            detail = descriptor.descriptor_text or "No descriptor is available for this scale."
            lines.append(
                f"[{label}] level={descriptor.level}; scale={descriptor.scale}; status={descriptor.status}; "
                f"reference_level={descriptor.reference_level}; text={detail}"
            )
        for index, missing in enumerate(context.cefr_missing, start=1):
            lines.append(
                f"[CEFR-MISSING-{index}] level={missing.level}; scale={missing.scale}; status={missing.status}"
            )
        lines.append("PEDAGOGICAL RESOURCES:")
        for block in context.resource_blocks:
            label = f"RESOURCE-{block.source_number}"
            section = " > ".join(block.heading_context) if block.heading_context else "none"
            image_note = " Image present but not interpreted." if block.requires_vision else ""
            lines.append(f"[{label}] section={section}; content={block.content}{image_note}")
        if context.warnings:
            lines.append("WARNINGS: " + " | ".join(context.warnings))
        system = (
            "You are a pedagogical assistant for teachers of Arabic. Use the resolved response language for all "
            "teacher-facing prose: headings, explanations, instructions, pedagogical advice, assessment descriptions, "
            "and conclusions. If the resolved response language is Arabic, write the entire teacher-facing response "
            "naturally in Arabic, without unnecessary French or English labels. The target teaching language is Arabic. "
            "Unless the teacher "
            "explicitly requests another target language, write learner-facing vocabulary, target utterances, "
            "dialogues, and production models in Arabic. Do not translate every Arabic example back into the "
            "response language. Use pedagogically appropriate learner terminology: in French, default to apprenant(s), "
            "use élève(s) only for an explicit primary or secondary school context, and étudiant(s) only for a clearly "
            "higher-education or adult-student context. In Arabic, default to المتعلم/المتعلمون when age or status is "
            "unspecified, use التلميذ/التلاميذ for an explicit school-pupil context, and الطالب/الطلاب only where a student "
            "context genuinely calls for it. Never mechanically translate student, étudiant, or learner into طالب. "
            "Whenever you generate Arabic linguistic material, use natural Modern Standard Arabic: "
            "use grammatically complete and idiomatic sentences, correct gender, pronoun/reference and possessive forms, "
            "natural verb/preposition combinations, and vocabulary suited to the requested CEFR level. Before finalizing "
            "learner-facing Arabic examples, silently check grammatical correctness, natural MSA usage, intended meaning, "
            "learner-level suitability, and coherence with the requested topic and situation. Prefer short reliable "
            "sentences to needlessly complex ones. Do not translate teacher-language phrasing word for word into Arabic; "
            "use idiomatic MSA instead. If diacritics are used, they must be accurate; otherwise prefer clean unvowelled "
            "Arabic. For A1, keep one communicative function at a time, a limited lexical load, and strong repetition; "
            "for A2 allow simple connected exchanges; for B1 retain explanations, opinions, clarification, reasons, "
            "suggestions, and simple problem solving. In an Arabic-only response, do not add unnecessary French or English "
            "glosses or labels unless the teacher explicitly requests bilingual material or a translation. Respect the "
            "user's explicit topic exactly; retrieved context is evidence, never "
            "permission to substitute a nearby generic theme. For speaking, prioritize oral interaction or production "
            "(short question-answer patterns, repetition with communicative use, pair work or role play); reading and "
            "writing can only be supporting steps. For A1, use familiar vocabulary, short utterances, clear scaffolding, "
            "limited lexical load, and simple question-answer patterns. "
            "Answer only from the supplied knowledge when making source-grounded claims. Structured CEFR knowledge "
            "is authoritative for proficiency constraints and must never be overridden by a resource. Use retrieved "
            "pedagogical resources for concrete teaching content when available. Do not invent CEFR descriptors, sources, "
            "pages, or visual interpretations, and do not attribute an ungrounded Arabic example to a source. If a CEFR "
            "scale is missing or unavailable, state that distinction accurately. Conversation history is only "
            "conversational memory: previous assistant messages are never authoritative knowledge or sources. Current "
            "structured CEFR and current retrieved resources override conversation history. Internal registry labels such as "
            "[CEFR-1], [CEFR-MISSING-1], [RESOURCE-2], Resource-6, chunk IDs, and Qdrant IDs are for grounding only. "
            "Never repeat them, including in parenthetical citations, in teacher-facing prose. "
            "First respect the explicit pedagogical request type; do not force every answer into a generic lesson "
            "template. For an explicit role play, prioritize a realistic situation, a communicative mission, "
            "complementary Role Card A and Role Card B information when useful, Arabic target vocabulary and sentence "
            "patterns appropriate to the scenario, including useful Arabic expressions, "
            "brief preparation, genuine learner interaction, an optional useful complication, and observable success "
            "criteria. A model opening can support the interaction, but must not replace the information gap or become "
            "the whole activity. For a listening activity, organize around pre-listening, an input, comprehension tasks, "
            "and a useful post-listening step. For reading, use pre-reading, an Arabic text, comprehension, and an "
            "appropriate follow-up. For vocabulary teaching, use discovery, meaning, pronunciation, guided practice, "
            "and reuse. For speaking, use linguistic support, interaction, and production. For writing, use preparation, "
            "a model or support, guided production, and revision. For a pedagogical method question, give a step-by-step "
            "procedure; for a general teacher question, give conversational pedagogical advice rather than an activity "
            "template. Differentiate the requested format by CEFR level: A1 needs short predictable heavily scaffolded "
            "exchanges and sentence frames; A2 allows short connected exchanges and modest choice; B1 favors less scripted "
            "problem solving, clarification, opinions, suggestions, and simple negotiation; higher levels permit more "
            "spontaneity and nuance. Include only sections that serve the teacher's request. Provide complete but concise, "
            "classroom-ready answers; avoid redundant explanation and repeated tables. When retrieved pedagogical context "
            "contains concrete activities, exercises, dialogues, or classroom tasks relevant to the request, use them as "
            "primary inspiration for classroom design, adapt them to the requested level, topic, and skill, and never copy "
            "them blindly or claim unsupported direct provenance. CEFR evidence controls level; concrete resources inform "
            "classroom design."
            + f" Answer in {self.language_name(request.language)}."
        )
        history_lines = [f"{message.role}: {message.content}" for message in history]
        user = "CURRENT PEDAGOGICAL KNOWLEDGE:\n" + "\n".join(lines)
        if history_lines:
            user += "\n\nCONVERSATION HISTORY:\n" + "\n".join(history_lines)
        user += "\n\nCURRENT USER MESSAGE:\n" + request.message.strip()
        return system, user


class AssistantChatService:
    """Composes Phase 6C context and the existing LLM provider for one chat turn."""

    _LANGUAGES = {"ar", "fr", "en", "es"}
    _MIN_TOP_K = 1
    _MAX_TOP_K = 20

    def __init__(
        self, *, knowledge: PedagogicalKnowledgeService, llm: LLMProvider,
        settings: Settings | None = None, prompt_builder: AssistantChatPromptBuilder | None = None,
        reviewer: ArabicLinguisticReviewService | None = None, personal_retrieval: PersonalRetrievalService | None = None,
        review_fallback_llm: LLMProvider | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.llm = llm
        self.settings = settings or get_settings()
        self.prompt_builder = prompt_builder or AssistantChatPromptBuilder()
        self.reviewer = reviewer or ArabicLinguisticReviewService(
            llm=llm,
            fallback_llm=review_fallback_llm,
            max_tokens=getattr(self.settings, "arabic_review_max_output_tokens", 3000),
            retry_policy=LLMRetryPolicy(
                max_retries=getattr(self.settings, "arabic_review_max_retries", 4),
                max_wait_seconds=getattr(self.settings, "arabic_review_max_wait_seconds", 20.0),
                retry_base_delay=getattr(self.settings, "arabic_review_retry_base_delay", 2.0),
                retry_max_delay=getattr(self.settings, "arabic_review_retry_max_delay", 8.0),
                retry_jitter_seconds=getattr(self.settings, "arabic_review_retry_jitter_seconds", 0.25),
            ),
            generation_options=LLMGenerationOptions(
                reasoning_effort=getattr(self.settings, "arabic_review_reasoning_effort", "medium"),
                include_reasoning=getattr(self.settings, "arabic_review_include_reasoning", False),
            ),
        )
        self.personal_retrieval = personal_retrieval

    @classmethod
    def _validate(cls, request: AssistantChatRequest) -> AssistantChatRequest:
        message = request.message.strip()
        if not message:
            raise AssistantChatValidationError("message is required.")
        language = request.language.strip().casefold() if request.language else None
        if language is not None and language not in cls._LANGUAGES:
            raise AssistantChatValidationError("language must be one of: ar, fr, en, es.")
        level = request.cefr_level.strip().upper() if request.cefr_level else None
        if level and level not in SUPPORTED_CEFR_LEVELS:
            raise AssistantChatValidationError("Unsupported CEFR level.")
        if not cls._MIN_TOP_K <= request.top_k <= cls._MAX_TOP_K:
            raise AssistantChatValidationError(f"top_k must be between {cls._MIN_TOP_K} and {cls._MAX_TOP_K}.")
        # Phase 6C owns canonical skill aliases; unknown values remain explicit
        # input errors rather than silently becoming unusable retrieval hints.
        _, unknown_skills = cls._normalize_skills(request.skills)
        if unknown_skills:
            raise AssistantChatValidationError("Unsupported pedagogical skills: " + ", ".join(unknown_skills))
        if request.mode not in {"knowledge_base", "user_documents"}:
            raise AssistantChatValidationError("Unsupported assistant mode.")
        if request.mode == "user_documents" and not request.document_ids:
            raise AssistantChatValidationError("At least one personal document is required.")
        return AssistantChatRequest(
            message=message, cefr_level=level, skills=request.skills, language=language,
            topic=request.topic.strip() if request.topic and request.topic.strip() else None,
            objective=request.objective.strip() if request.objective and request.objective.strip() else None,
            top_k=request.top_k, mode=request.mode, document_ids=tuple(sorted(set(request.document_ids))),
        )

    def _personal_prepared(self, db: Session, request: AssistantChatRequest, *, owner_id: int, history: tuple[AssistantChatHistoryMessage, ...]) -> PreparedAssistantStream:
        if self.personal_retrieval is None:
            raise AssistantChatServiceError("PERSONAL_RETRIEVAL_UNAVAILABLE", "Personal retrieval is unavailable.")
        try:
            results, timings = self.personal_retrieval.search(db, query=request.message, owner_id=owner_id, document_ids=list(request.document_ids), top_k=request.top_k)
        except PersonalDocumentAccessError as exc:
            raise AssistantChatValidationError(str(exc)) from exc
        sources = [AssistantChatSource("personal_document", item.document_id, item.document_title, item.page_number, item.page_number, None) for item in results]
        diagnostics = AssistantChatDiagnostics(request.cefr_level, request.language or "fr", len(results), len(results), len(sources), 0, [], self.llm.model_id, None, len(history), sum(len(item.content) for item in history))
        if not results:
            return PreparedAssistantStream(request, None, None, [], diagnostics, self._insufficient(request.language or "fr"))
        system = "You are a pedagogical assistant. Answer only from the supplied private teacher documents. " + f"Answer in {self.prompt_builder.language_name(request.language or 'fr')}."
        evidence = "\n\n".join(f"[PERSONAL-{i}] document={item.document_title}; page={item.page_number}; content={item.content}" for i, item in enumerate(results, 1))
        prompt = f"PRIVATE TEACHER DOCUMENTS:\n{evidence}\n\nQUESTION:\n{request.message}"
        return PreparedAssistantStream(request, system, prompt, sources, diagnostics)

    @staticmethod
    def _normalize_skills(skills: tuple[str, ...]) -> tuple[list[str], list[str]]:
        # Delegates to the existing Phase 6C normalizer without duplicating its aliases.
        return PedagogicalKnowledgeService._normal_skills(skills)

    @staticmethod
    def _insufficient(language: str) -> str:
        return {
            "ar": "المصادر المتاحة لا توفر معلومات كافية للإجابة بدقة عن هذا السؤال.",
            "fr": "Les sources disponibles ne permettent pas de répondre précisément à cette question.",
            "es": "Las fuentes disponibles no permiten responder con precisión a esta pregunta.",
        }.get(language, "The available sources do not provide enough information to answer this question accurately.")

    @staticmethod
    def _sources(context: PedagogicalContext) -> list[AssistantChatSource]:
        sources: list[AssistantChatSource] = []
        seen: set[tuple[object, ...]] = set()
        for descriptor in context.cefr_descriptors:
            for provenance in descriptor.sources:
                source = AssistantChatSource(
                    "cefr_structured", provenance.document_id, None, provenance.page_start,
                    provenance.page_end, descriptor.scale,
                )
                key = (source.source_type, source.document_id, source.page_start, source.page_end, source.cefr_scale)
                if key not in seen:
                    sources.append(source); seen.add(key)
        for block in context.resource_blocks:
            source = AssistantChatSource(
                "pedagogical_resource", block.document_id, block.document_title,
                block.page_start, block.page_end, None,
            )
            key = (source.source_type, source.document_id, source.page_start, source.page_end, source.cefr_scale)
            if key not in seen:
                sources.append(source); seen.add(key)
        return sources

    @staticmethod
    def _retrieval_topic(
        request: AssistantChatRequest, history: tuple[AssistantChatHistoryMessage, ...],
    ) -> tuple[str, bool]:
        previous_user_messages = [
            ChatParameterResolver.historical_semantic_text(message.content)
            for message in history
            if message.role == "USER"
        ]
        previous_user_messages = [message for message in previous_user_messages if message]
        if not previous_user_messages:
            return request.topic or request.message, False
        current = request.message
        if request.topic and request.topic.casefold() != request.message.casefold():
            current += "\nTopic: " + request.topic
        return (
            "Current request:\n" + current
            + "\n\nRecent pedagogical context:\n"
            + "\n".join(previous_user_messages),
            True,
        )

    def answer(
        self,
        db: Session,
        request: AssistantChatRequest,
        *,
        history: tuple[AssistantChatHistoryMessage, ...] = (),
        owner_id: int | None = None,
    ) -> AssistantChatResponse:
        request = self._validate(request)
        if request.mode == "user_documents":
            if owner_id is None:
                raise AssistantChatServiceError("PERSONAL_RETRIEVAL_UNAVAILABLE", "Personal retrieval requires an owner.")
            prepared = self._personal_prepared(db, request, owner_id=owner_id, history=history)
            if prepared.immediate_answer is not None:
                return self.stream_response(prepared, prepared.immediate_answer)
            try:
                result = self.llm.generate(system_prompt=prepared.system_prompt, user_prompt=prepared.user_prompt, temperature=self.settings.rag_llm_temperature, max_tokens=getattr(self.settings, "assistant_llm_max_output_tokens", self.settings.rag_llm_max_tokens))
            except LLMProviderError as exc:
                raise AssistantChatServiceError("ASSISTANT_PROVIDER_ERROR", "The assistant provider is unavailable.", status_code=exc.status_code) from exc
            reviewed = self.reviewer.review(content=result.text, response_language=request.language or "fr", cefr_level=request.cefr_level, request_context=request.message)
            return AssistantChatResponse(clean_internal_rag_references(reviewed.content), prepared.sources, prepared.diagnostics)
        resolved = ChatParameterResolver().resolve(
            message=request.message,
            cefr_level=request.cefr_level,
            skills=request.skills,
            language=request.language,
            user_history=tuple(message.content for message in history if message.role == "USER"),
        )
        request = AssistantChatRequest(
            message=request.message,
            cefr_level=resolved.cefr_level,
            skills=resolved.skills,
            language=resolved.response_language or "fr",
            topic=request.topic,
            objective=request.objective,
            top_k=request.top_k, mode=request.mode, document_ids=request.document_ids,
        )
        history_chars = sum(len(message.content) for message in history)
        retrieval_topic, topic_is_context = self._retrieval_topic(request, history)
        context = self.knowledge.build_context(db, PedagogicalKnowledgeRequest(
            cefr_level=request.cefr_level, topic=retrieval_topic,
            objective=request.objective, language=request.language, skills=request.skills,
            retrieval_top_k=request.top_k, topic_is_context=topic_is_context,
        ))
        sources = self._sources(context)
        has_knowledge = bool(context.cefr_descriptors or context.cefr_missing or context.resource_blocks)
        if not has_knowledge and not history:
            return AssistantChatResponse(
                answer=self._insufficient(request.language), sources=[],
                diagnostics=AssistantChatDiagnostics(
                    request.cefr_level, request.language, context.retrieved_count, context.selected_count,
                    0, context.requires_vision_count, context.warnings, None, None,
                    len(history), history_chars,
                ),
            )
        system_prompt, user_prompt = self.prompt_builder.build(request=request, context=context, history=history)
        try:
            result = self.llm.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=self.settings.rag_llm_temperature,
                max_tokens=getattr(self.settings, "assistant_llm_max_output_tokens", self.settings.rag_llm_max_tokens),
            )
        except LLMProviderError as exc:
            raise AssistantChatServiceError("ASSISTANT_PROVIDER_ERROR", "The assistant provider is unavailable.", status_code=exc.status_code) from exc
        reviewed = self.reviewer.review(
            content=result.text,
            response_language=request.language,
            cefr_level=request.cefr_level,
            request_context=request.topic or request.message,
        )
        return AssistantChatResponse(
            answer=clean_internal_rag_references(reviewed.content), sources=sources,
            diagnostics=AssistantChatDiagnostics(
                request.cefr_level, request.language, context.retrieved_count, context.selected_count,
                len(sources), context.requires_vision_count, context.warnings,
                result.model, result.finish_reason, len(history), history_chars,
            ),
        )

    def prepare_stream(
        self, db: Session, request: AssistantChatRequest, *, history: tuple[AssistantChatHistoryMessage, ...] = (), owner_id: int | None = None,
    ) -> PreparedAssistantStream:
        """Run the unchanged RAG path once, stopping immediately before LLM generation."""
        request = self._validate(request)
        if request.mode == "user_documents":
            if owner_id is None:
                raise AssistantChatServiceError("PERSONAL_RETRIEVAL_UNAVAILABLE", "Personal retrieval requires an owner.")
            return self._personal_prepared(db, request, owner_id=owner_id, history=history)
        resolved = ChatParameterResolver().resolve(
            message=request.message, cefr_level=request.cefr_level, skills=request.skills,
            language=request.language, user_history=tuple(message.content for message in history if message.role == "USER"),
        )
        request = AssistantChatRequest(
            message=request.message, cefr_level=resolved.cefr_level, skills=resolved.skills,
            language=resolved.response_language or "fr", topic=request.topic, objective=request.objective, top_k=request.top_k, mode=request.mode, document_ids=request.document_ids,
        )
        history_chars = sum(len(message.content) for message in history)
        retrieval_topic, topic_is_context = self._retrieval_topic(request, history)
        context = self.knowledge.build_context(db, PedagogicalKnowledgeRequest(
            cefr_level=request.cefr_level, topic=retrieval_topic, objective=request.objective,
            language=request.language, skills=request.skills, retrieval_top_k=request.top_k, topic_is_context=topic_is_context,
        ))
        sources = self._sources(context)
        diagnostics = AssistantChatDiagnostics(
            request.cefr_level, request.language, context.retrieved_count, context.selected_count,
            len(sources), context.requires_vision_count, context.warnings, self.llm.model_id, None,
            len(history), history_chars,
        )
        if not (context.cefr_descriptors or context.cefr_missing or context.resource_blocks) and not history:
            return PreparedAssistantStream(request, None, None, [], diagnostics, self._insufficient(request.language))
        system_prompt, user_prompt = self.prompt_builder.build(request=request, context=context, history=history)
        return PreparedAssistantStream(request, system_prompt, user_prompt, sources, diagnostics)

    async def stream_answer(self, prepared: PreparedAssistantStream):
        """Real provider stream. Arabic review remains a complete-answer gate for non-streaming V1."""
        if prepared.immediate_answer is not None:
            yield prepared.immediate_answer
            return
        stream = getattr(self.llm, "stream_generate", None)
        if stream is None:
            raise AssistantChatServiceError("ASSISTANT_PROVIDER_ERROR", "Streaming is unavailable.")
        upstream = stream(
            system_prompt=prepared.system_prompt, user_prompt=prepared.user_prompt,
            temperature=self.settings.rag_llm_temperature,
            max_tokens=getattr(self.settings, "assistant_llm_max_output_tokens", self.settings.rag_llm_max_tokens),
            generation_options=LLMGenerationOptions(reasoning_effort="low", include_reasoning=False),
        )
        try:
            async for delta in upstream:
                yield delta
        finally:
            # Ensure a client disconnect closes the provider's HTTP streaming
            # context immediately instead of merely abandoning its iterator.
            close = getattr(upstream, "aclose", None)
            if close is not None:
                await close()

    @staticmethod
    def stream_response(prepared: PreparedAssistantStream, content: str) -> AssistantChatResponse:
        return AssistantChatResponse(
            answer=clean_internal_rag_references(content), sources=prepared.sources, diagnostics=prepared.diagnostics,
        )
