import asyncio
from types import SimpleNamespace

import pytest

from app.services.assistant_chat_service import (
    AssistantChatHistoryMessage,
    AssistantChatPromptBuilder,
    AssistantChatRequest,
    AssistantChatService,
    AssistantChatServiceError,
    AssistantChatValidationError,
    clean_internal_rag_references,
)
from app.services.llm_provider import FakeLLMProvider, LLMProviderError
from app.services.llm_provider import LLMResult
from app.services.pedagogical_knowledge_service import (
    CEFRSourceProvenance,
    PedagogicalCEFRDescriptor,
    PedagogicalCEFRMissing,
    PedagogicalContext,
    PedagogicalKnowledgeService,
    PedagogicalResourceBlock,
)


def _context(*, with_cefr=True, with_resource=True, status="AVAILABLE", missing=False, vision=False):
    descriptors = [PedagogicalCEFRDescriptor(
        level="A1", scale="Interaction orale générale", status=status,
        descriptor_text="Peut gérer des interactions simples." if status == "AVAILABLE" else None,
        reference_level="C1" if status == "NO_DESCRIPTOR_AVAILABLE" else None,
        sources=[CEFRSourceProvenance(19, 76, 76, 205, 0)],
    )] if with_cefr else []
    resources = [PedagogicalResourceBlock(
        source_number=1, document_id=15, document_title="Lesson plans", chunk_ids=[42], page_start=4,
        page_end=4, heading_context=["Goals"], content_type="paragraph", structural_quality="structured",
        content="Use family vocabulary in simple oral interactions.", requires_vision=vision,
        image_not_interpreted=vision, vector_scores=[0.9], reranker_scores=[None], original_ranks=[1], reranked_ranks=[None],
    )] if with_resource else []
    return PedagogicalContext(
        request_summary={"cefr_level": "A1" if with_cefr else None, "language": "fr"},
        cefr_descriptors=descriptors,
        cefr_missing=[PedagogicalCEFRMissing("A1", "Production orale générale")] if missing else [],
        resource_blocks=resources, retrieved_count=len(resources), selected_count=len(resources),
        sources=[], warnings=[], requires_vision_count=int(vision),
    )


class FakeKnowledge:
    def __init__(self, context): self.context, self.calls = context, []
    def build_context(self, db, request): self.calls.append((db, request)); return self.context


class ErrorProvider:
    model_id = "error-provider"
    def generate(self, **_kwargs): raise LLMProviderError("unavailable", status_code=503, provider_message="safe")


class SequenceProvider:
    model_id = "sequence-provider"
    def __init__(self, outcomes): self.outcomes, self.calls = list(outcomes), []
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception): raise outcome
        return LLMResult(text=outcome, model=self.model_id)


class StreamSequenceProvider:
    model_id = "stream-sequence-provider"

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = []

    async def stream_generate(self, **kwargs):
        self.calls.append(kwargs)
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class CloseTrackingStreamProvider(StreamSequenceProvider):
    def __init__(self):
        super().__init__(["Bon", "jour"])
        self.closed = False

    async def stream_generate(self, **kwargs):
        self.calls.append(kwargs)
        try:
            yield "Bon"
            yield "jour"
        finally:
            self.closed = True


def _service(context, llm=None):
    knowledge = FakeKnowledge(context)
    llm = llm or FakeLLMProvider("Réponse pédagogique fondée.")
    service = AssistantChatService(
        knowledge=knowledge, llm=llm, settings=SimpleNamespace(
            rag_llm_temperature=0.2,
            rag_llm_max_tokens=1200,
            assistant_llm_max_output_tokens=1800,
        ),
    )
    return service, knowledge, llm


def test_valid_question_combines_cefr_resources_and_safe_sources():
    service, knowledge, llm = _service(_context())
    response = service.answer(None, AssistantChatRequest("Comment parler de la famille ?", cefr_level="a1", skills=("speaking",), language="fr"))
    request = knowledge.calls[0][1]
    assert request.cefr_level == "A1" and request.topic == "Comment parler de la famille ?"
    assert response.answer == "Réponse pédagogique fondée."
    assert [(source.source_type, source.document_id, source.page_start) for source in response.sources] == [
        ("cefr_structured", 19, 76), ("pedagogical_resource", 15, 4)
    ]
    assert "CEFR-1" in llm.calls[0]["user_prompt"] and "RESOURCE-1" in llm.calls[0]["user_prompt"]
    assert "42" not in llm.calls[0]["user_prompt"]
    assert llm.calls[0]["max_tokens"] == 1800


def test_arabic_reviewer_uses_its_own_completion_budget_without_changing_primary_generation():
    provider = SequenceProvider(["هذا مثال عربي صالح وطويل بما يكفي.", "هذا مثال عربي صالح وطويل بما يكفي."])
    service, _knowledge, _llm = _service(_context(), provider)

    response = service.answer(None, AssistantChatRequest("اقترح نشاطًا عن الأسرة.", language="ar"))

    assert response.answer == "هذا مثال عربي صالح وطويل بما يكفي."
    assert len(provider.calls) == 2
    assert provider.calls[0]["max_tokens"] == 1800 and "generation_options" not in provider.calls[0]
    assert provider.calls[1]["max_tokens"] == 3000
    assert provider.calls[1]["generation_options"].reasoning_effort == "medium"
    assert provider.calls[1]["generation_options"].include_reasoning is False


def test_internal_rag_reference_cleanup_is_narrow_and_preserves_teacher_content():
    original = (
        "**Activité**\n\n"
        "Utilisez le guide peda CE1 (voir Resource-6). [CEFR-1]\n"
        "(cf. RESOURCE-2) CEFR-MISSING-1\n"
        "**Arabe** : أمي اسمها فاطمة. Le CECRL reste pertinent."
    )

    cleaned = clean_internal_rag_references(original)

    assert "Resource-6" not in cleaned
    assert "RESOURCE-2" not in cleaned
    assert "CEFR-1" not in cleaned and "CEFR-MISSING-1" not in cleaned
    assert "**Activité**" in cleaned
    assert "guide peda CE1" in cleaned
    assert "أمي اسمها فاطمة." in cleaned
    assert "CECRL reste pertinent" in cleaned
    assert clean_internal_rag_references("Voir Resource-6 pour la suite.") == " pour la suite."


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Exercice 48 (voir Resource-6)", "Exercice 48"),
        ("Exercice 48 (voir Resource‑6)", "Exercice 48"),
        ("Exercice 48 (cf. RESOURCE–2)", "Exercice 48"),
        ("Activité [CEFR—1]", "Activité"),
        ("[CEFR‑MISSING—1]", ""),
        ("Resource‑4", ""),
    ],
)
def test_internal_rag_reference_cleanup_supports_common_unicode_dashes(value, expected):
    assert clean_internal_rag_references(value) == expected


def test_internal_rag_reference_cleanup_regression_for_non_breaking_hyphen():
    reference = "Resource‑6"
    assert ord(reference[len("Resource")]) == 0x2011

    cleaned = clean_internal_rag_references(
        "Exercice 48 du cahier d’activités (voir Resource‑6)"
    )

    assert cleaned == "Exercice 48 du cahier d’activités"


def test_internal_rag_reference_cleanup_preserves_normal_typographic_dashes():
    answer = (
        "niveau A1–A2\n"
        "activité orale – 10 minutes\n"
        "arabe — français\n"
        "guide peda CE1\n"
        "أمي اسمها فاطمة."
    )

    assert clean_internal_rag_references(answer) == answer


def test_service_cleans_generated_prose_but_keeps_structured_sources_unchanged():
    raw = "**Titre**\n\nExercice (voir Resource-6)\n[CEFR-1] أمي اسمها فاطمة."
    service, _knowledge, _llm = _service(_context(), FakeLLMProvider(raw))

    response = service.answer(None, AssistantChatRequest("Propose une activité A1."))

    assert response.answer == "**Titre**\n\nExercice\n أمي اسمها فاطمة."
    assert [(source.source_type, source.document_id) for source in response.sources] == [
        ("cefr_structured", 19), ("pedagogical_resource", 15),
    ]


def test_real_stream_path_yields_provider_deltas_and_sanitizes_only_after_completion():
    provider = StreamSequenceProvider(["**Activité**\n", "Exercice (voir Resource-6)"])
    service, _knowledge, _llm = _service(_context(), provider)
    prepared = service.prepare_stream(None, AssistantChatRequest("Propose une activité A1."))

    async def collect():
        return [chunk async for chunk in service.stream_answer(prepared)]

    chunks = asyncio.run(collect())
    response = service.stream_response(prepared, "".join(chunks))

    assert chunks == ["**Activité**\n", "Exercice (voir Resource-6)"]
    assert provider.calls[0]["max_tokens"] == 1800
    assert provider.calls[0]["generation_options"] == LLMGenerationOptions(reasoning_effort="low", include_reasoning=False)
    assert response.answer == "**Activité**\nExercice"
    assert [(source.source_type, source.document_id) for source in response.sources] == [
        ("cefr_structured", 19), ("pedagogical_resource", 15),
    ]


def test_streaming_does_not_invoke_the_complete_answer_arabic_review_gate():
    provider = StreamSequenceProvider(["نشاط شفهي بسيط."])
    service, _knowledge, _llm = _service(_context(), provider)
    prepared = service.prepare_stream(None, AssistantChatRequest("اقترح نشاطًا A1.", language="ar"))

    async def collect():
        return [chunk async for chunk in service.stream_answer(prepared)]

    assert asyncio.run(collect()) == ["نشاط شفهي بسيط."]
    assert len(provider.calls) == 1


def test_closing_the_assistant_stream_closes_the_upstream_provider_stream():
    provider = CloseTrackingStreamProvider()
    service, _knowledge, _llm = _service(_context(), provider)
    prepared = service.prepare_stream(None, AssistantChatRequest("Propose une activité A1."))

    async def consume_then_cancel():
        stream = service.stream_answer(prepared)
        assert await anext(stream) == "Bon"
        await stream.aclose()

    asyncio.run(consume_then_cancel())
    assert provider.closed is True


def test_review_runs_before_the_existing_internal_reference_sanitizer_and_keeps_sources_unchanged():
    provider = SequenceProvider([
        "**Titre**\n\nأبي يعمل طبيب. [RESOURCE-1]",
        "**Titre**\n\nأبي طبيب. [RESOURCE-1]",
    ])
    service, _knowledge, _llm = _service(_context(), provider)

    response = service.answer(None, AssistantChatRequest("اقترح نشاطًا A1 حول الأسرة."))

    assert len(provider.calls) == 2
    assert response.answer == "**Titre**\n\nأبي طبيب."
    assert "[RESOURCE-1]" not in response.answer
    assert [(source.source_type, source.document_id) for source in response.sources] == [
        ("cefr_structured", 19), ("pedagogical_resource", 15),
    ]


def test_clean_answer_without_registry_labels_is_unchanged():
    answer = "**Titre**\n\nUne activité CECRL avec أمي اسمها فاطمة."
    assert clean_internal_rag_references(answer) == answer


def test_question_without_cefr_level_does_not_invent_a_default_level():
    service, knowledge, _llm = _service(_context(with_cefr=False))
    service.answer(None, AssistantChatRequest("Comment introduire une activité ?", language="en"))
    request = knowledge.calls[0][1]
    assert request.cefr_level is None and request.topic == "Comment introduire une activité ?"


def test_current_message_parameters_are_resolved_before_fresh_retrieval():
    service, knowledge, _llm = _service(_context())

    service.answer(None, AssistantChatRequest("Propose-moi une activité orale A1 sur la famille."))

    retrieval_request = knowledge.calls[0][1]
    assert retrieval_request.cefr_level == "A1"
    assert retrieval_request.skills == ("speaking",)
    assert retrieval_request.language == "fr"


def test_recent_user_parameters_fill_gaps_but_current_message_replaces_them():
    service, knowledge, _llm = _service(_context())
    history = (
        AssistantChatHistoryMessage("USER", "Propose une activité orale A1 sur la famille."),
        AssistantChatHistoryMessage("ASSISTANT", "Je suggère plutôt le niveau B2 et la lecture."),
    )

    service.answer(None, AssistantChatRequest("Maintenant niveau A2 et compréhension orale."), history=history)

    retrieval_request = knowledge.calls[0][1]
    assert retrieval_request.cefr_level == "A2"
    assert retrieval_request.skills == ("listening",)
    assert "famille" in retrieval_request.topic.casefold()
    assert "A1" not in retrieval_request.topic
    assert "activité orale" not in retrieval_request.topic.casefold()
    assert retrieval_request.topic_is_context is True

    resource_query = PedagogicalKnowledgeService._resource_semantic_query(
        retrieval_request,
        retrieval_request.cefr_level,
        list(retrieval_request.skills),
    )
    assert "niveau A2" in resource_query
    assert "famille" in resource_query.casefold()
    assert "compréhension orale" in resource_query and "écoute" in resource_query
    assert "A1" not in resource_query
    assert "expression orale" not in resource_query
    assert "interaction orale" not in resource_query
    assert "production orale" not in resource_query


def test_explicit_api_parameters_override_message_and_user_history():
    service, knowledge, _llm = _service(_context())
    history = (AssistantChatHistoryMessage("USER", "activité orale A1"),)

    service.answer(
        None,
        AssistantChatRequest(
            "Propose une activité de lecture A2.",
            cefr_level="B1",
            skills=("speaking",),
            language="es",
        ),
        history=history,
    )

    retrieval_request = knowledge.calls[0][1]
    assert retrieval_request.cefr_level == "B1"
    assert retrieval_request.skills == ("speaking",)
    assert retrieval_request.language == "es"


def test_no_descriptor_and_missing_scale_are_preserved_in_prompt():
    service, _knowledge, llm = _service(_context(status="NO_DESCRIPTOR_AVAILABLE", missing=True))
    service.answer(None, AssistantChatRequest("Que proposer ?", cefr_level="A1", language="fr"))
    prompt = llm.calls[0]["user_prompt"]
    assert "status=NO_DESCRIPTOR_AVAILABLE" in prompt
    assert "status=NO_STRUCTURED_DESCRIPTOR_FOUND" in prompt
    assert "No descriptor is available" in prompt


def test_empty_context_returns_localized_insufficient_answer_without_llm_call():
    service, _knowledge, llm = _service(_context(with_cefr=False, with_resource=False))
    response = service.answer(None, AssistantChatRequest("Aidez-moi", language="ar"))
    assert "المصادر المتاحة" in response.answer and response.sources == []
    assert llm.calls == [] and response.diagnostics.provider_model is None


def test_requested_language_and_vision_warning_are_preserved_without_visual_claims():
    service, _knowledge, llm = _service(_context(vision=True))
    response = service.answer(None, AssistantChatRequest("Question", language="es"))
    assert "Answer in Spanish." in llm.calls[0]["system_prompt"]
    assert "Image present but not interpreted" in llm.calls[0]["user_prompt"]
    assert response.diagnostics.requires_vision_count == 1


def test_family_speaking_activity_prompt_keeps_arabic_target_material_and_a1_scaffolding():
    service, _knowledge, llm = _service(_context())

    service.answer(None, AssistantChatRequest(
        "Propose-moi une activité orale A1 sur la famille.",
        cefr_level="A1", skills=("speaking",), language="fr", topic="la famille",
    ))

    system = llm.calls[0]["system_prompt"]
    assert "Answer in French." in system
    assert "target teaching language is Arabic" in system
    assert "Respect the user's explicit topic exactly" in system
    assert "For speaking, prioritize oral interaction or production" in system
    assert "For A1, use familiar vocabulary, short utterances, clear scaffolding" in system
    assert "Arabic target vocabulary and sentence patterns" in system
    assert "CEFR knowledge is authoritative for proficiency constraints" in system
    assert "Use retrieved pedagogical resources for concrete teaching content" in system
    assert "[CEFR-1], [CEFR-MISSING-1], [RESOURCE-2], Resource-6" in system


def test_arabic_target_examples_are_requested_even_when_teacher_explanation_is_french():
    service, _knowledge, llm = _service(_context())

    service.answer(None, AssistantChatRequest(
        "Propose une activité sur la famille.", cefr_level="A1", language="fr",
    ))

    system = llm.calls[0]["system_prompt"]
    assert "Use the resolved response language for all teacher-facing prose" in system
    assert "write learner-facing vocabulary, target utterances, dialogues, and production models in Arabic" in system
    assert "Do not translate every Arabic example back into the response language" in system


@pytest.mark.parametrize(
    ("message", "expected_language", "language_name"),
    [
        ("اقترح نشاطًا شفهيًا للمستوى A1 حول الأسرة.", "ar", "Arabic"),
        ("كيف يمكنني تدريس مفردات المدرسة للمستوى A1 بطريقة ممتعة؟", "ar", "Arabic"),
        ("Propose une activité orale A1 sur la famille.", "fr", "French"),
        ("Suggest an A1 speaking activity about family.", "en", "English"),
        ("Propón una actividad oral A1 sobre la familia.", "es", "Spanish"),
        ("Réponds en arabe et propose une activité orale A1 sur la famille.", "ar", "Arabic"),
    ],
)
def test_resolved_response_language_controls_all_teacher_facing_prompt_prose(
    message, expected_language, language_name
):
    service, knowledge, llm = _service(_context())

    service.answer(None, AssistantChatRequest(message))

    request = knowledge.calls[0][1]
    system = llm.calls[0]["system_prompt"]
    assert request.language == expected_language
    assert f"Answer in {language_name}." in system
    assert "Use the resolved response language for all teacher-facing prose" in system
    assert "target teaching language is Arabic" in system


def test_activity_in_arabic_keeps_french_teacher_response_without_an_explicit_override():
    service, knowledge, llm = _service(_context())

    service.answer(None, AssistantChatRequest("Propose une activité A1 en arabe sur la famille."))

    assert knowledge.calls[0][1].language == "fr"
    assert "Answer in French." in llm.calls[0]["system_prompt"]
    assert "write learner-facing vocabulary, target utterances, dialogues, and production models in Arabic" in (
        llm.calls[0]["system_prompt"]
    )


def test_prompt_hardens_arabic_learner_material_without_a_second_generation_step():
    service, _knowledge, llm = _service(_context())

    service.answer(None, AssistantChatRequest(
        "اقترح نشاطًا شفهيًا للمستوى A1 حول الأسرة.",
        cefr_level="A1",
        skills=("speaking",),
    ))

    system = llm.calls[0]["system_prompt"]
    assert len(llm.calls) == 1
    assert "natural Modern Standard Arabic" in system
    assert "correct gender, pronoun/reference and possessive forms" in system
    assert "silently check grammatical correctness, natural MSA usage" in system
    assert "coherence with the requested topic and situation" in system
    assert "Do not translate teacher-language phrasing word for word into Arabic" in system
    assert "For A1, keep one communicative function at a time" in system
    assert "for B1 retain explanations, opinions, clarification, reasons" in system
    assert "do not add unnecessary French or English glosses or labels" in system
    assert "أبي طبيب." not in system and "جدتي تحب الزراعة." not in system


def test_prompt_uses_contextual_pedagogical_learner_terminology_without_mechanical_translation():
    service, _knowledge, llm = _service(_context())

    service.answer(None, AssistantChatRequest("Propose une activité A1 sur la famille.", language="fr"))

    system = llm.calls[0]["system_prompt"]
    assert "in French, default to apprenant(s)" in system
    assert "élève(s) only for an explicit primary or secondary school context" in system
    assert "étudiant(s) only for a clearly higher-education or adult-student context" in system
    assert "default to المتعلم/المتعلمون when age or status is unspecified" in system
    assert "use التلميذ/التلاميذ for an explicit school-pupil context" in system
    assert "Never mechanically translate student, étudiant, or learner into طالب" in system


def test_arabic_quality_regression_examples_remain_test_only_guidance():
    bad_profession = "أبي يعمل طبيب."
    preferred_profession = "أبي طبيب."
    bad_possessive = "جديّة تحب الزراعة."
    preferred_possessive = "جدتي تحب الزراعة."

    assert bad_profession != preferred_profession
    assert bad_possessive != preferred_possessive


def test_general_teacher_question_is_not_forced_into_an_activity_template():
    builder = AssistantChatPromptBuilder()
    system, _user = builder.build(
        request=AssistantChatRequest("Quel est le niveau visé ?", cefr_level="A1", language="fr"),
        context=_context(),
    )

    assert "general teacher question, give conversational pedagogical advice" in system
    assert "generic lesson template" in system


def test_role_play_prompt_prioritizes_authentic_mission_roles_and_information_gap():
    builder = AssistantChatPromptBuilder()
    system, _user = builder.build(
        request=AssistantChatRequest(
            "Propose un jeu de rôle B1 en arabe sur un problème de voyage.",
            cefr_level="B1", skills=("speaking",), language="fr",
        ),
        context=_context(),
    )

    assert "realistic situation, a communicative mission" in system
    assert "complementary Role Card A and Role Card B information" in system
    assert "must not replace the information gap" in system
    assert "B1 favors less scripted problem solving, clarification, opinions, suggestions, and simple negotiation" in system
    assert "Answer in French." in system
    assert "target teaching language is Arabic" in system


def test_prompt_plans_listening_reading_and_vocabulary_without_role_play_overreach():
    builder = AssistantChatPromptBuilder()
    system, _user = builder.build(
        request=AssistantChatRequest("Comment enseigner le vocabulaire des voyages ?", language="fr"),
        context=_context(),
    )

    assert "listening activity, organize around pre-listening" in system
    assert "For reading, use pre-reading, an Arabic text, comprehension" in system
    assert "For vocabulary teaching, use discovery, meaning, pronunciation, guided practice, and reuse" in system
    assert "explicit role play" in system
    assert "Internal registry labels" in system
    assert "concrete activities, exercises, dialogues, or classroom tasks" in system
    assert "CEFR evidence controls level; concrete resources inform classroom design" in system


def test_follow_up_uses_prior_user_context_for_fresh_retrieval_and_full_history_for_prompt():
    service, knowledge, llm = _service(_context())
    history = (
        AssistantChatHistoryMessage("USER", "Propose-moi une activité orale A1 sur la famille."),
        AssistantChatHistoryMessage("ASSISTANT", "Voici une activité orale de vingt minutes."),
    )

    response = service.answer(None, AssistantChatRequest("Rends-la plus facile.", language="fr"), history=history)

    retrieval_request = knowledge.calls[0][1]
    prompt = llm.calls[0]["user_prompt"]
    assert "Recent pedagogical context:" in retrieval_request.topic
    assert "activité orale A1" not in retrieval_request.topic
    assert "sur la famille" in retrieval_request.topic
    assert "Voici une activité" not in retrieval_request.topic
    assert retrieval_request.topic.startswith("Current request:\nRends-la plus facile.")
    assert retrieval_request.cefr_level == "A1"
    assert retrieval_request.skills == ("speaking",)
    assert "CONVERSATION HISTORY:" in prompt
    assert "USER: Propose-moi une activité" in prompt
    assert "ASSISTANT: Voici une activité" in prompt
    assert prompt.endswith("CURRENT USER MESSAGE:\nRends-la plus facile.")
    assert response.diagnostics.history_messages_used == 2
    assert response.diagnostics.history_chars_used == sum(len(item.content) for item in history)
    assert [(source.document_id, source.page_start) for source in response.sources] == [(19, 76), (15, 4)]


def test_current_message_is_primary_over_sanitized_historical_context():
    service, knowledge, _llm = _service(_context())
    history = (AssistantChatHistoryMessage("USER", "Propose une activité orale A1 sur la famille."),)

    service.answer(None, AssistantChatRequest("Fais plutôt une activité A2 de lecture sur l'école."), history=history)

    retrieval_request = knowledge.calls[0][1]
    assert retrieval_request.topic.startswith("Current request:\nFais plutôt une activité A2 de lecture sur l'école.")
    assert "famille" in retrieval_request.topic.casefold()
    assert "A1" not in retrieval_request.topic
    assert "activité orale" not in retrieval_request.topic.casefold()
    assert retrieval_request.cefr_level == "A2"
    assert retrieval_request.skills == ("reading",)


def test_history_can_support_a_safe_transformation_when_fresh_context_is_empty():
    service, knowledge, llm = _service(_context(with_cefr=False, with_resource=False))
    history = (AssistantChatHistoryMessage("ASSISTANT", "Activité précédente : discussion en binômes."),)

    response = service.answer(None, AssistantChatRequest("Réduis-la à dix minutes.", language="fr"), history=history)

    assert knowledge.calls and llm.calls
    assert response.answer == "Réponse pédagogique fondée."


def test_current_structured_cefr_unavailability_remains_authoritative_over_history():
    service, _knowledge, llm = _service(_context(status="NO_DESCRIPTOR_AVAILABLE"))
    history = (AssistantChatHistoryMessage("ASSISTANT", "Un ancien message prétend qu'un descripteur est disponible."),)

    service.answer(None, AssistantChatRequest("Que dit le CECRL ?", cefr_level="A1", language="fr"), history=history)

    assert "status=NO_DESCRIPTOR_AVAILABLE" in llm.calls[0]["user_prompt"]
    assert "previous assistant messages are never authoritative" in llm.calls[0]["system_prompt"]


def test_current_missing_structured_cefr_fact_remains_authoritative_over_history():
    service, _knowledge, llm = _service(_context(with_cefr=False, missing=True))
    history = (AssistantChatHistoryMessage("ASSISTANT", "Un ancien message invente un descripteur."),)

    service.answer(None, AssistantChatRequest("Que dit le CECRL ?", cefr_level="A1", language="fr"), history=history)

    assert "status=NO_STRUCTURED_DESCRIPTOR_FOUND" in llm.calls[0]["user_prompt"]
    assert "Current structured CEFR and current retrieved resources override conversation history" in llm.calls[0]["system_prompt"]


def test_provider_errors_are_typed_and_do_not_expose_provider_message():
    service, _knowledge, _llm = _service(_context(), ErrorProvider())
    with pytest.raises(AssistantChatServiceError) as error:
        service.answer(None, AssistantChatRequest("Question", language="fr"))
    assert error.value.code == "ASSISTANT_PROVIDER_ERROR" and str(error.value) == "The assistant provider is unavailable."
    assert "safe" not in str(error.value)


@pytest.mark.parametrize("chat_request", [
    AssistantChatRequest("", language="fr"),
    AssistantChatRequest("x", cefr_level="A0", language="fr"),
    AssistantChatRequest("x", skills=("invented",), language="fr"),
    AssistantChatRequest("x", language="de"),
    AssistantChatRequest("x", top_k=21, language="fr"),
])
def test_invalid_input_is_rejected_before_knowledge_or_llm(chat_request):
    service, knowledge, llm = _service(_context())
    with pytest.raises(AssistantChatValidationError):
        service.answer(None, chat_request)
    assert knowledge.calls == [] and llm.calls == []


def test_request_has_no_persistence_private_document_or_streaming_fields():
    names = set(AssistantChatRequest.__dataclass_fields__)
    assert {"user_id", "conversation_id", "document_ids", "stream"}.isdisjoint(names)
