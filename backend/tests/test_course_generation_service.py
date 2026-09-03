import json

import pytest

from app.schemas.course_generator import CourseGenerateIn
from app.services.course_generation_service import CourseGenerationError, CourseGenerationService, CourseRateLimitError
from app.services.llm_provider import FakeLLMProvider, LLMProviderError, LLMResult
from app.services.pedagogical_knowledge_service import PedagogicalContext


def _context():
    return PedagogicalContext({"cefr_level": "A1", "language": "ar"}, [], [], [], 0, 0, [], [], 0)


def _course(level="B1", duration=60, title="Cours : Le voyage", content=None):
    if content is None:
        content = [{"title": "Le lexique du voyage", "body": "Présentation du vocabulaire utile pour voyager."}]
    return {
        "title": title, "level": level, "theme": "Le voyage", "duration": duration,
        "objectives": ["Utiliser le vocabulaire du voyage", "Décrire un trajet"],
        "skills": ["Expression orale"],
        "vocabulary": ["مطار", "طائرة", "سفر"],
        "expressions": ["أين تطير؟", "كم تستغرق الرحلة؟"],
        "introduction": "Aujourd'hui nous apprenons à parler du voyage et à poser des questions simples.",
        "grammar": [{"title": "Les questions avec أين", "body": "On construit la question avec أين + verbe.", "examples": [{"title": "Exemple", "body": "أين تسكن؟"}]}],
        "content": content,
        "dialogue": {"context": "Dans un aéroport.", "lines": ["مسافر: أين تطير؟", "موظف: أسافر إلى باريس."]},
        "comprehension": [{"title": "Vrai ou faux ?", "instructions": "Réponds d'après le dialogue.", "example": None}],
        "guided_practice": [{"title": "Complète la question", "instructions": "Complète avec أين.", "example": {"title": "Exemple", "body": "أين تسكن؟"}}],
        "communicative_practice": [{"title": "En binôme", "instructions": "Pose et réponds à la question أين."}],
        "production": [{"title": "Présente ton trajet", "instructions": "Décris un déplacement en deux ou trois phrases."}],
        "summary": ["L'apprenant sait décrire un trajet", "Il sait poser une question simple"],
        "homework": "Réviser le vocabulaire et préparer un court dialogue.",
    }


def test_a1_family_course_fr_is_validated():
    course = _course(level="A1", duration=45, title="Cours : La famille")
    course["theme"] = "Famille"
    llm = FakeLLMProvider(json.dumps(course, ensure_ascii=False))
    request = CourseGenerateIn(level="A1", theme="Famille", objective="Présenter les membres de sa famille", duration_minutes=45, language="fr")
    result = CourseGenerationService(llm=llm).generate(request, _context())
    assert result.level == "A1"
    assert result.duration == 45
    assert result.vocabulary
    assert result.content
    assert result.introduction != ""
    assert len(result.summary) >= 1
    assert result.dialogue is not None and result.dialogue.lines
    assert llm.calls[0]["max_tokens"] == 4096
    assert llm.calls[0]["generation_options"].reasoning_effort == "low"
    assert "Expert en didactique" in llm.calls[0]["system_prompt"] or "expert en didactique" in llm.calls[0]["system_prompt"]


def test_a1_family_short_is_validated_and_uses_short_structures():
    course = _course(level="A1", duration=15, title="درس: الأسرة")
    course.update({"theme": "الأسرة", "vocabulary": ["أب", "أم", "أخ", "أخت"]})
    course["grammar"] = [{
        "title": "دلالة هذا وهذه",
        "body": "هذا للمذكر وهذه للمؤنث.",
        "examples": [{"title": "جملة قصيرة", "body": "هذا أبي."}, {"title": "جملة قصيرة", "body": "هذه أمي."}],
    }]
    llm = FakeLLMProvider(json.dumps(course, ensure_ascii=False))
    request = CourseGenerateIn(level="A1", theme="الأسرة", objective="تعلم مفردات الأسرة", duration_minutes=15, language="ar")
    result = CourseGenerationService(llm=llm).generate(request, _context())
    assert result.theme == "الأسرة"
    assert result.title == "درس: الأسرة"
    assert result.vocabulary == ["أب", "أم", "أخ", "أخت"]
    assert result.guided_practice and result.production
    assert result.summary


def test_a2_school_course_is_validated():
    course = _course(level="A2", duration=60, title="درس: المدرسة")
    course.update({
        "theme": "المدرسة", "vocabulary": ["مدرسة", "فصل", "كتاب", "معلم"],
        "content": [{"title": "في الصف", "body": "نصف يومنا في المدرسة بجمل قصيرة.", "examples": []}],
    })
    llm = FakeLLMProvider(json.dumps(course, ensure_ascii=False))
    result = CourseGenerationService(llm=llm).generate(
        CourseGenerateIn(level="A2", theme="المدرسة", objective="الحديث عن المدرسة", duration_minutes=60, language="ar"),
        _context(),
    )
    assert result.level == "A2"
    assert result.vocabulary
    assert result.introduction != ""
    assert result.comprehension


def test_b1_travel_course_is_validated():
    llm = FakeLLMProvider(json.dumps(_course(level="B1", duration=60), ensure_ascii=False))
    request = CourseGenerateIn(level="B1", theme="Voyage", objective="Développer l'expression orale", duration_minutes=60, language="fr")
    result = CourseGenerationService(llm=llm).generate(request, _context())
    assert result.level == "B1"
    assert result.duration == 60
    assert result.grammar[0].title == "Les questions avec أين"


def test_b1_connectors_and_justification_are_encouraged_in_prompt():
    prompt = CourseGenerationService._build_system_prompt()
    assert "لأن" in prompt
    assert "لكن" in prompt
    assert "بعد ذلك" in prompt
    assert "لذلك" in prompt
    assert "جمل واضحة ومترابطة" in prompt


def test_a1_explicitly_never_asks_for_complex_sentences_in_prompt():
    prompt = CourseGenerationService._build_system_prompt()
    a1_region = prompt[prompt.index("A1 :"):prompt.index("A2 :")]
    # The A1 guidance must explicitly forbid demanding complex sentences
    # for low levels (not demand them), and must never present them as a goal.
    assert "jamaIs <em>" not in a1_region
    assert "phraseS COURTES" in a1_region or "phrases COURTES" in a1_region or "phrases courtes" in a1_region
    assert "ne demande JAMAIS" in a1_region or "ne demande jamais" in a1_region or "JAMAIS" in a1_region
    assert "جمل معقدة" in a1_region and "phrases complexes" in a1_region


def test_b2_culture_course_is_validated_and_encourages_argumentation():
    course = _course(level="B2", duration=90, title="درس: الثقافة المغربية")
    course.update({
        "theme": "الثقافة المغربية",
        "objectives": ["Argumenter sur les traditions", "Comparer des pratiques culturelles"],
        "vocabulary": ["تقاليد", "عادات", "تراث", "مناسبات"],
        "grammar": [{"title": "Le comparatif", "body": "Présentation du comparatif et de la nuance.", "examples": [{"title": "Exemple", "body": "هذا التراث أغنى من ذاك."}]}],
    })
    request = CourseGenerateIn(level="B2", theme="الثقافة المغربية", objective="Argumenter et défendre une position", duration_minutes=90, language="ar")
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False))).generate(request, _context())
    assert result.level == "B2"
    assert result.content
    assert result.grammar[0].title == "Le comparatif"
    prompt = CourseGenerationService._build_system_prompt()
    assert "argumentation" in prompt and "comparaison" in prompt


def test_rejects_incomplete_course_form():
    with pytest.raises(Exception):
        CourseGenerateIn()  # missing required theme/objective at Pydantic level


def test_rejects_an_incomplete_json_object():
    with pytest.raises(CourseGenerationError, match="incomplet ou invalide"):
        CourseGenerationService._json_object('{"title": "درس')


def test_works_without_relevant_rag_documents():
    llm = FakeLLMProvider(json.dumps(_course(), ensure_ascii=False))
    context = PedagogicalContext({"cefr_level": "B1", "language": "ar"}, [], [], [], 0, 0, [], ["No relevant document"], 0)
    result = CourseGenerationService(llm=llm).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        context,
    )
    assert result.rag_sources_used == 0


def test_rejects_invalid_llm_json_shape(caplog):
    invalid = {"titre": "Cours", "clés": ["invalides"]}
    service = CourseGenerationService(llm=FakeLLMProvider(json.dumps(invalid, ensure_ascii=False)))
    with pytest.raises(CourseGenerationError, match="format attendu"):
        service.generate(
            CourseGenerateIn(level="A1", theme="الأسرة", objective="تعلم المفردات", duration_minutes=45, language="ar"),
            _context(),
        )
    assert "course_schema_validation_failed" in caplog.text


def test_rejects_an_llm_timeout_as_a_provider_error():
    from app.services.llm_provider import LLMProviderError

    class TimedOutLLM(FakeLLMProvider):
        def generate(self, **_kwargs):
            raise LLMProviderError("provider timed out", provider_message="Timeout du fournisseur")

    service = CourseGenerationService(llm=TimedOutLLM())
    with pytest.raises(CourseGenerationError, match="Timeout du fournisseur"):
        service.generate(
            CourseGenerateIn(level="A1", theme="Famille", objective="Vocabulaire", duration_minutes=45),
            _context(),
        )


def test_rejects_a_header_duration_that_differs_from_the_request():
    course = _course(duration=30)
    service = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False)))
    with pytest.raises(CourseGenerationError, match="durée annoncée"):
        service.generate(
            CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
            _context(),
        )


def test_rejects_a_level_mismatch():
    course = _course(level="B2")
    service = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False)))
    with pytest.raises(CourseGenerationError, match="niveau annoncé"):
        service.generate(
            CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
            _context(),
        )


def test_rejects_a_theme_mismatch():
    course = _course()
    course["theme"] = "Un autre sujet"
    service = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False)))
    with pytest.raises(CourseGenerationError, match="thème"):
        service.generate(
            CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
            _context(),
        )


def test_normalizes_duration_strings_before_pydantic_validation():
    course = _course()
    course["duration"] = "60 minutes"
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False))).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        _context(),
    )
    assert result.duration == 60


def test_extracts_a_complete_json_object_from_json_markdown_and_trailing_text():
    raw = "Voici :\n```json\n" + json.dumps(_course(), ensure_ascii=False) + "\n```\nFin."
    parsed = CourseGenerationService._json_object(raw)
    assert parsed["title"] == "Cours : Le voyage"


def test_summary_is_always_a_list_of_points():
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(_course(), ensure_ascii=False))).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        _context(),
    )
    assert isinstance(result.summary, list)
    assert all(isinstance(item, str) for item in result.summary)
    assert result.summary


def test_summary_normalized_from_a_single_string():
    course = _course()
    course["summary"] = "L'apprenant sait décrire un trajet."
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False))).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        _context(),
    )
    assert isinstance(result.summary, list)
    assert result.summary[0] == "L'apprenant sait décrire un trajet."


def test_homework_and_dialogue_optional():
    course = _course()
    course.pop("homework")
    course["dialogue"] = None
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False))).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        _context(),
    )
    assert result.homework is None
    assert result.dialogue is None


def test_content_grammar_and_exercise_lists_are_typed():
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(_course(), ensure_ascii=False))).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        _context(),
    )
    assert isinstance(result.grammar, list)
    assert isinstance(result.content, list)
    assert isinstance(result.comprehension, list)
    assert isinstance(result.guided_practice, list)
    assert isinstance(result.communicative_practice, list)
    assert isinstance(result.production, list)
    assert result.grammar[0].title
    assert result.grammar[0].body
    assert result.guided_practice[0].example is not None and result.guided_practice[0].example.body


def test_legacy_grammar_shape_is_normalized_to_sections():
    course = _course()
    course["grammar"] = [{"topic": "Les questions avec أين", "explanation": "On construit la question avec أين + verbe.", "example": "أين تسكن؟"}]
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False))).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        _context(),
    )
    assert result.grammar[0].title == "Les questions avec أين"
    assert result.grammar[0].body == "On construit la question avec أين + verbe."
    assert result.grammar[0].examples[0].body == "أين تسكن؟"


def test_legacy_string_dialogue_is_normalized_to_lines():
    course = _course()
    course["dialogue"] = "مسافر: أين تطير؟ / موظف: أسافر إلى باريس."
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False))).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        _context(),
    )
    assert result.dialogue is not None
    assert len(result.dialogue.lines) == 2


def test_rejects_a_complex_sentence_requirement_at_b1():
    course = _course(level="B1", duration=60)
    course["summary"] = ["L'apprenant rédige des phrases complexes."]
    service = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False)))
    with pytest.raises(CourseGenerationError, match="phrases complexes"):
        service.generate(
            CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
            _context(),
        )


def test_complex_sentence_requirement_still_allowed_above_b1():
    course = _course(level="C1", duration=90)
    course["theme"] = "Culture"
    course["summary"] = ["L'apprenant produit des phrases complexes et nuancées."]
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False))).generate(
        CourseGenerateIn(level="C1", theme="Culture", objective="Argumenter", duration_minutes=90),
        _context(),
    )
    assert result.level == "C1"


def test_prompt_contains_cefr_adaptation_and_progression_and_expert_role():
    prompt = CourseGenerationService._build_system_prompt()
    assert "expert en didactique de la langue arabe" in prompt
    assert "CECRL" in prompt
    assert "contenu pédagogique" in prompt
    assert "argumentation" in prompt
    for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
        assert level in prompt
    for stage in ("Comprendre", "Observer", "Apprendre", "Pratiquer", "Réutiliser", "Produire", "Bilan"):
        assert stage in prompt
    assert "introduction" in prompt
    assert "guided_practice" in prompt and "communicative_practice" in prompt
    assert "title, level, theme, duration" in prompt


def test_rag_sources_count_and_provider_model_are_recorded():
    from app.services.pedagogical_knowledge_service import PedagogicalResourceBlock
    course = _course()
    blocks = [
        {"document_title": "Manuel de vocabulaire", "page_start": 1, "page_end": 2, "content": "Lexique du voyage..." * 100},
        {"document_title": "Guide conversation", "page_start": 3, "page_end": 4, "content": "Dialogues utiles..." * 100},
    ]
    resource_blocks = [
        PedagogicalResourceBlock(source_number=1, document_id=1, document_title=b["document_title"], chunk_ids=[1], page_start=b["page_start"], page_end=b["page_end"], heading_context=[], content_type="text", structural_quality=None, content=b["content"], requires_vision=False, image_not_interpreted=False, vector_scores=[], reranker_scores=[], original_ranks=[], reranked_ranks=[])
        for b in blocks
    ]
    context = PedagogicalContext({"cefr_level": "B1", "language": "ar"}, [], [], resource_blocks, 0, 0, [], [], 0)
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False))).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        context,
    )
    assert result.rag_sources_used == 2
    assert result.provider_model


def test_json_object_selects_the_course_not_an_earlier_reasoning_fragment():
    full_course = _course(level="A1", duration=30, title="تقديم أفراد الأسرة")
    reasoning_fragment = {
        "topic": "الضمائر المتصلة",
        "explanation": "تضاف إلى الاسم لتحديد المتكلم أو المخاطب.",
        "example": "أبي يعمل في المدرسة.",
    }
    raw = (
        "Voici mon raisonnement : "
        + json.dumps(reasoning_fragment, ensure_ascii=False)
        + " puis le cours final : "
        + json.dumps(full_course, ensure_ascii=False)
    )
    payload = CourseGenerationService._json_object(raw)
    assert payload.get("title") == "تقديم أفراد الأسرة"
    assert "topic" not in payload


def test_json_object_returns_single_valid_course_directly():
    full_course = _course(level="A1", duration=30)
    payload = CourseGenerationService._json_object(json.dumps(full_course, ensure_ascii=False))
    assert payload.get("title") == full_course["title"]
    assert payload.get("level") == "A1"


class QueuedLLM(FakeLLMProvider):
    """Return successive responses in order for retry tests."""

    def __init__(self, responses):
        super().__init__(responses[0])
        self._queue = list(responses)

    def generate(self, **kwargs):
        self.calls.append({
            "system_prompt": kwargs.get("system_prompt"),
            "user_prompt": kwargs.get("user_prompt"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "retry_policy": kwargs.get("retry_policy"),
            "generation_options": kwargs.get("generation_options"),
        })
        return LLMResult(text=self._queue.pop(0), model=self.model_id)


def _incomplete_course():
    course = _course(level="A1", duration=30)
    del course["skills"]
    del course["vocabulary"]
    return course


def test_retry_complete_first_call_no_retry():
    course = _course(level="A1", duration=30, title="Cours : La famille")
    course["theme"] = "Famille"
    llm = FakeLLMProvider(json.dumps(course, ensure_ascii=False))
    result = CourseGenerationService(llm=llm).generate(
        CourseGenerateIn(level="A1", theme="Famille", objective="Présenter la famille", duration_minutes=30, language="fr"),
        _context(),
    )
    assert result.level == "A1"
    assert len(llm.calls) == 1


def test_retry_incomplete_then_complete():
    llm = QueuedLLM([
        json.dumps(_incomplete_course(), ensure_ascii=False),
        json.dumps(_course(level="A1", duration=30), ensure_ascii=False),
    ])
    result = CourseGenerationService(llm=llm).generate(
        CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30),
        _context(),
    )
    assert result.level == "A1"
    assert result.summary
    assert len(llm.calls) == 2
    assert "incomplète" in llm.calls[1]["user_prompt"]


def test_retry_incomplete_then_incomplete_raises_cleanly():
    llm = QueuedLLM([
        json.dumps(_incomplete_course(), ensure_ascii=False),
        json.dumps(_incomplete_course(), ensure_ascii=False),
    ])
    service = CourseGenerationService(llm=llm)
    with pytest.raises(CourseGenerationError):
        service.generate(
            CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30),
            _context(),
        )
    assert len(llm.calls) == 2


def test_retry_wrong_type_does_not_retry():
    course = _course(level="A1", duration=30)
    course["skills"] = {"expression": "orale"}
    llm = FakeLLMProvider(json.dumps(course, ensure_ascii=False))
    service = CourseGenerationService(llm=llm)
    with pytest.raises(CourseGenerationError):
        service.generate(
            CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30),
            _context(),
        )
    assert len(llm.calls) == 1


def test_retry_rag_sources_unchanged():
    from app.services.pedagogical_knowledge_service import PedagogicalResourceBlock

    def block(title, content):
        return PedagogicalResourceBlock(
            source_number=1, document_id=1, document_title=title, chunk_ids=[1],
            page_start=1, page_end=2, heading_context=[], content_type="text",
            structural_quality=None, content=content, requires_vision=False,
            image_not_interpreted=False, vector_scores=[], reranker_scores=[],
            original_ranks=[], reranked_ranks=[],
        )

    blocks = [
        block("Manuel", "Lexique du voyage..." * 100),
        block("Guide", "Dialogues..." * 100),
    ]
    context = PedagogicalContext({"cefr_level": "A1", "language": "ar"}, [], [], blocks, 0, 0, [], [], 0)
    llm = FakeLLMProvider(json.dumps(_course(level="A1", duration=30), ensure_ascii=False))
    result = CourseGenerationService(llm=llm).generate(
        CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30),
        context,
    )
    assert len(llm.calls) == 1
    assert result.rag_sources_used == 2


def test_no_text_duplication_between_sections():
    """The same exercise/instruction must not be duplicated verbatim between
    two practice sections (requirement: no duplicated text across sections)."""
    course = _course()
    course["guided_practice"] = [{"title": "Complète la question", "instructions": "Complète avec أين.", "example": None}]
    course["communicative_practice"] = [{"title": "Complète la question", "instructions": "Complète avec أين.", "example": None}]
    service = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False)))
    with pytest.raises(CourseGenerationError, match="différencier"):
        service.generate(
            CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
            _context(),
        )


def test_language_fr_produces_arabic_learning_content_with_french_guidance():
    """For language=fr, the web service only checks the flag reach the prompt
    (content quality is not structurally verifiable by the unit test); this
    asserts the french language and the arabic learning content coexist."""
    course = _course(level="A1", duration=45, title="Cours : La famille")
    course["theme"] = "Famille"
    llm = FakeLLMProvider(json.dumps(course, ensure_ascii=False))
    request = CourseGenerateIn(level="A1", theme="Famille", objective="Présenter les membres de sa famille", duration_minutes=45, language="fr")
    result = CourseGenerationService(llm=llm).generate(request, _context())
    assert result.level == "A1"
    assert result.content[0].body
    user = json.loads(llm.calls[0]["user_prompt"])
    assert user["request"]["language"] == "fr"


def test_objective_is_optional_and_absent_when_not_provided():
    """The objective must be optional: when not provided it is serialized as
    null in the request instead of forcing the frontend to send one."""
    course = _course(level="A1", duration=30, title="Cours : La famille")
    course["theme"] = "Famille"
    llm = FakeLLMProvider(json.dumps(course, ensure_ascii=False))
    result = CourseGenerationService(llm=llm).generate(
        CourseGenerateIn(level="A1", theme="Famille", duration_minutes=30),
        _context(),
    )
    assert result.level == "A1"
    user = json.loads(llm.calls[0]["user_prompt"])
    assert user["request"]["objective"] is None


def test_prompt_forces_objective_definition_when_absent():
    """The system prompt must always produce an objective: a provided one takes
    priority (CAS A), an absent one is auto-defined (CAS B), both are coherent
    with the theme and level."""
    prompt = CourseGenerationService._build_system_prompt()
    assert "CAS A" in prompt
    assert "CAS B" in prompt
    assert "prioritaire" in prompt or "priorité" in prompt
    assert "thème et au niveau CECRL" in prompt


class RateLimitedLLM(FakeLLMProvider):
    """Raises HTTP 429 (TPM) a configurable number of times, then returns the
    requested course. Never calls a real provider."""

    def __init__(self, course_text: str, n_429: int = 1, *, provider_message: str | None = None, retry_after_seconds: float | None = None) -> None:
        super().__init__(course_text)
        self._remaining_429 = n_429
        self._provider_message = provider_message
        self._retry_after_seconds = retry_after_seconds

    def generate(self, **kwargs):
        self.calls.append({
            "system_prompt": kwargs.get("system_prompt"), "user_prompt": kwargs.get("user_prompt"),
            "temperature": kwargs.get("temperature"), "max_tokens": kwargs.get("max_tokens"),
            "retry_policy": kwargs.get("retry_policy"), "generation_options": kwargs.get("generation_options"),
        })
        if self._remaining_429 > 0:
            self._remaining_429 -= 1
            metadata = {}
            if self._retry_after_seconds is not None:
                metadata["retry_after_seconds"] = self._retry_after_seconds
            raise LLMProviderError(
                "The configured LLM provider did not return a valid response.",
                status_code=429,
                provider_message=self._provider_message or "Rate limit reached for model `openai/gpt-oss-20b`. Please try again in 8.97s",
                category="http_status_error",
                response_metadata=metadata,
            )
        return LLMResult(text=self.response, model="fake-llm")


def _call_service(llm, sleep_record=None, **service_kwargs):
    sleeps: list[float] = [] if sleep_record is None else sleep_record
    service = CourseGenerationService(llm=llm, sleep=sleeps.append, **service_kwargs)
    return service, sleeps


def test_rate_limit_429_retry_then_success():
    llm = RateLimitedLLM(json.dumps(_course(level="A1", duration=30), ensure_ascii=False), n_429=1)
    service, sleeps = _call_service(llm, sleep_record=[])
    result = service.generate(CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30), _context())
    assert result.level == "A1"
    assert len(llm.calls) == 2  # one 429 attempt + one successful attempt
    # delay = provider message delay (8.97) + margin (1.0)
    assert sleeps and abs(sleeps[0] - 9.97) < 0.01
    # exactly one controlled retry, no infinite loop


def test_rate_limit_429_then_429_raises_clean_error():
    llm = RateLimitedLLM(json.dumps(_course(level="A1", duration=30), ensure_ascii=False), n_429=3)
    service, _ = _call_service(llm, sleep_record=[])
    with pytest.raises(CourseRateLimitError, match="temporairement très sollicité"):
        service.generate(CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30), _context())
    # 1 initial attempt + 1 controlled retry -> then exhausted. No infinite loop.
    assert len(llm.calls) == 2


def test_rate_limit_delay_respects_retry_after_header_over_message():
    retry_after = 3.25
    llm = RateLimitedLLM(json.dumps(_course(level="A1", duration=30), ensure_ascii=False), n_429=1, retry_after_seconds=retry_after)
    service, sleeps = _call_service(llm, [])
    service.generate(CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30), _context())
    # Retry-After (3.25) takes priority over the message delay (8.97) + margin
    assert sleeps and abs(sleeps[0] - (retry_after + 1.0)) < 1e-9


def test_rate_limit_success_without_retry():
    llm = FakeLLMProvider(json.dumps(_course(level="A1", duration=30), ensure_ascii=False))
    service, sleeps = _call_service(llm, [])
    service.generate(CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30), _context())
    assert len(llm.calls) == 1
    assert sleeps == []  # no wait on a success path


def test_extract_retry_delay_from_provider_message():
    assert CourseGenerationService._extract_retry_delay("Please try again in 8.97s") == 8.97
    assert CourseGenerationService._extract_retry_delay("Rate limit reached. Retry in 5 seconds.") == 5.0
    assert CourseGenerationService._extract_retry_delay("try again in 2") == 2.0
    assert CourseGenerationService._extract_retry_delay(None) is None
    assert CourseGenerationService._extract_retry_delay("no delay info here") is None


def test_rate_limit_uses_bounded_fallback_when_no_retry_after():
    llm = RateLimitedLLM(json.dumps(_course(level="A1", duration=30), ensure_ascii=False), n_429=1, provider_message="Rate limit reached")
    service, sleeps = _call_service(llm, [])
    service.generate(CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30), _context())
    # no Retry-After, no message delay -> fallback (2.0) + margin (1.0)
    assert sleeps and abs(sleeps[0] - 3.0) < 1e-9


def test_rate_limit_never_loops_past_max_retries():
    # Many consecutive 429s; the service must stop after exactly one retry.
    llm = RateLimitedLLM(json.dumps(_course(level="A1", duration=30), ensure_ascii=False), n_429=100)
    service, sleeps = _call_service(llm, [])
    with pytest.raises(CourseRateLimitError):
        service.generate(CourseGenerateIn(level="A1", theme="Voyage", objective="Interagir", duration_minutes=30), _context())
    assert len(llm.calls) == 2
    assert len(sleeps) == 1


def _genealogy_course():
    """The exact CourseOut shape from the reported bug: B1 / culture marocaine."""
    return _course(level="B1", duration=30, title="Découvrir la culture marocaine")


def _grammar_section():
    return {"title": "الضمائر المتصلة", "body": "Explication du point grammatical.", "examples": [{"title": "Exemple", "body": "ضمائر متصلة"}]}


def test_selects_course_over_nested_grammar_section_when_both_present():
    """TEST 1 — a nested grammar section must never win over the full CourseOut."""
    raw = (
        "Je réponds : "
        + json.dumps(_grammar_section(), ensure_ascii=False)
        + " Ensuite le cours : "
        + json.dumps(_genealogy_course(), ensure_ascii=False)
    )
    payload = CourseGenerationService._json_object(raw)
    assert payload["title"] == "Découvrir la culture marocaine"
    assert payload["level"] == "B1"
    assert "body" not in payload
    assert "الضمائر المتصلة" != payload["title"]


def test_selects_course_root_with_all_nested_sections():
    """TEST 2 — a full CourseOut containing many nested objects returns the root."""
    course = _course()
    course["dialogue"] = {"context": "Context", "lines": ["a: b", "c: d"]}
    raw = "Raisonnement : " + json.dumps(_grammar_section(), ensure_ascii=False) + " Resultat : " + json.dumps(course, ensure_ascii=False)
    payload = CourseGenerationService._json_object(raw)
    assert payload.get("title") == course["title"]
    assert isinstance(payload.get("grammar"), list)
    assert isinstance(payload.get("content"), list)
    assert isinstance(payload.get("dialogue"), dict)


def test_selects_course_among_multiple_valid_json_objects():
    """TEST 3 — when several valid JSON objects are present, pick the CourseOut."""
    interlopers = [
        {"topic": "الضمائر المتصلة", "explanation": "explication"},
        {"hello": "world"},
        {"title": "Un autre fragment", "notes": []},
    ]
    raw = json.dumps(interlopers[0], ensure_ascii=False) + "\n" + json.dumps(interlopers[1], ensure_ascii=False) + "\n" + json.dumps(interlopers[2], ensure_ascii=False) + "\n" + json.dumps(_genealogy_course(), ensure_ascii=False)
    payload = CourseGenerationService._json_object(raw)
    assert payload["title"] == "Découvrir la culture marocaine"
    assert payload["level"] == "B1"


def test_grammar_section_only_response_is_rejected():
    """TEST 4 — a response containing only a CourseSection must NOT be accepted
    as a CourseOut; it triggers the existing error behaviour (Pydantic rejects)."""
    service = CourseGenerationService(llm=FakeLLMProvider(json.dumps(_grammar_section(), ensure_ascii=False)))
    with pytest.raises(CourseGenerationError):
        service.generate(
            CourseGenerateIn(level="B1", theme="la culture marocaine", objective="Découvrir", duration_minutes=30),
            _context(),
        )


def test_extracts_course_root_with_leading_and_trailing_text():
    """TEST 5 — a valid CourseOut wrapped in prose / reasoning fragments."""
    raw = "Raisonnement préalable...\nVeuillez trouver le cours :\n```json\n" + json.dumps(_genealogy_course(), ensure_ascii=False) + "\n```\nParfait, c'est tout."
    payload = CourseGenerationService._json_object(raw)
    assert payload["title"] == "Découvrir la culture marocaine"
    assert payload["level"] == "B1"
    assert payload["duration"] == 30


def test_course_parsing_is_arabic_agnostic():
    """TEST 6 — Arabic content, no dependency on the wording."""
    raw = json.dumps(_course(level="A2", duration=60, title="درس: الأسرة"), ensure_ascii=False)
    payload = CourseGenerationService._json_object(raw)
    assert payload["title"] == "درس: الأسرة"
    assert payload["level"] == "A2"


def test_course_parsing_is_french_agnostic():
    """TEST 7 — French content, no dependency on the wording."""
    raw = json.dumps(_course(level="B1", duration=60, title="Cours : Le voyage"), ensure_ascii=False)
    payload = CourseGenerationService._json_object(raw)
    assert payload["title"] == "Cours : Le voyage"
    assert payload["level"] == "B1"


def test_full_course_roundtrip_validates():
    """TEST 8 — a CourseOut with all current sections passes Pydantic."""
    course = _course(level="B1", duration=60, title="Cours : Le voyage")
    result = CourseGenerationService(llm=FakeLLMProvider(json.dumps(course, ensure_ascii=False))).generate(
        CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
        _context(),
    )
    assert result.level == "B1"
    assert result.grammar and result.content and result.production
    assert result.summary


class _FailingLLM(FakeLLMProvider):
    """Double that raises an LLMProviderError with a configurable category / cause."""

    def __init__(self, error):
        super().__init__()
        self.error = error

    def generate(self, **_kwargs):
        raise self.error


def _assert_logged_provider_failure(caplog, error_type, user_message, secret="sk-NEVER-EXPOSE"):
    records = [r for r in caplog.records if getattr(r, "message", "").startswith("provider_request_failed")]
    assert records, "expected a provider_request_failed log record"
    line = records[0].message
    assert f"error_type={error_type}" in line
    assert f"status_code=" in line
    assert user_message in line
    assert "duration_seconds=" in line
    assert secret not in line


def test_provider_transport_failure_is_logged_without_secrets_and_raises_clean_error(caplog):
    err = LLMProviderError(
        "provider did not return a valid response", status_code=None,
        provider_message="Provider request failed before a response was received.",
        category="transport_error", response_metadata={"error_type": "transport_error"},
    )
    service = CourseGenerationService(llm=_FailingLLM(err))
    with pytest.raises(CourseGenerationError, match="Provider request failed before a response was received"):
        service.generate(
            CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
            _context(),
        )
    _assert_logged_provider_failure(caplog, "transport_error", "Provider request failed before a response was received")


def test_provider_timeout_is_logged_as_read_timeout(caplog):
    err = LLMProviderError(
        "provider timed out", status_code=None, provider_message="Timeout du fournisseur",
        category="read_timeout", response_metadata={"error_type": "read_timeout"},
    )
    service = CourseGenerationService(llm=_FailingLLM(err))
    with pytest.raises(CourseGenerationError, match="Timeout du fournisseur"):
        service.generate(
            CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60),
            _context(),
        )
    _assert_logged_provider_failure(caplog, "read_timeout", "Timeout du fournisseur")


def test_provider_connect_timeout_appends_cause_based_category_when_metadata_absent(caplog):
    err = LLMProviderError("connect timed out", status_code=None, provider_message="Provider request failed before a response was received.", category="connect_timeout")
    service = CourseGenerationService(llm=_FailingLLM(err))
    with pytest.raises(CourseGenerationError, match="Provider request failed before a response was received"):
        service.generate(CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60), _context())
    _assert_logged_provider_failure(caplog, "connect_timeout", "Provider request failed before a response was received")


def test_4xx_and_5xx_provider_errors_are_logged_typed_and_not_retried(caplog):
    for status, category in ((401, "http_status_error"), (403, "http_status_error"), (502, "http_status_error"), (504, "http_status_error")):
        caplog.clear()
        err = LLMProviderError("denied", status_code=status, provider_message=f"Provider returned HTTP {status}.",
                               category=category, response_metadata={"error_type": category, "status_code": status})
        calls = []
        class _ServiceTrackingLLM(_FailingLLM):
            model_id = "openai/gpt-oss-20b"
            def generate(self, **kw):
                calls.append(1)
                raise self.error
        service = CourseGenerationService(llm=_ServiceTrackingLLM(err))
        with pytest.raises(CourseGenerationError, match=f"Provider returned HTTP {status}"):
            service.generate(CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60), _context())
        assert len(calls) == 1, "4xx/5xx must NOT be retried by the course path"
        _assert_logged_provider_failure(caplog, category, f"Provider returned HTTP {status}.")


def test_empty_response_maps_to_request_failed_and_logs_groq_model_without_key(caplog):
    err = LLMProviderError(
        "provider did not return a valid response", status_code=None,
        provider_message="Provider request failed before a response was received.",
        category="transport_error", response_metadata={"error_type": "transport_error"},
    )
    class _ModelLLM(_FailingLLM):
        model_id = "openai/gpt-oss-20b"
    service = CourseGenerationService(llm=_ModelLLM(err))
    with pytest.raises(CourseGenerationError, match="Provider request failed before a response was received"):
        service.generate(CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60), _context())
    line = [r.message for r in caplog.records if getattr(r, "message", "").startswith("provider_request_failed")][0]
    assert "model=openai/gpt-oss-20b" in line
    assert "provider=groq" in line


def test_provider_success_is_not_logged_as_a_failure(caplog):
    service = CourseGenerationService(llm=FakeLLMProvider(json.dumps(_course(), ensure_ascii=False)))
    result = service.generate(CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60), _context())
    assert result.level == "B1"
    assert not any(r.message.startswith("provider_request_failed") for r in caplog.records)


def test_rate_limit_429_still_uses_exactly_one_controlled_retry(caplog):
    class _RateLimited(FakeLLMProvider):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def generate(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise LLMProviderError("rate limited", status_code=429, provider_message="Limit 8000 TPM, Used 6206, Requested 2990")
            return LLMResult(text=json.dumps(_course(), ensure_ascii=False), model="openai/gpt-oss-20b")

    service = CourseGenerationService(llm=_RateLimited(), sleep=lambda _s: None)
    result = service.generate(CourseGenerateIn(level="B1", theme="Voyage", objective="Interagir", duration_minutes=60), _context())
    assert result.level == "B1"
    assert service.llm.calls == 2  # exactly one retry
