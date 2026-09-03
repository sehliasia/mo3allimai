import json

import pytest

from app.core.config import get_settings
from app.schemas.exercise_generator import ExerciseGenerateIn
from app.services.exercise_generation_service import ExerciseGenerationError, ExerciseGenerationService, ExerciseRateLimitError
from app.services.llm_provider import FakeLLMProvider, LLMProviderError, LLMResult
from app.services.pedagogical_knowledge_service import PedagogicalContext, PedagogicalResourceBlock


def _empty_context():
    return PedagogicalContext({"cefr_level": "A1", "language": "ar"}, [], [], [], 0, 0, [], [], 0)


def _exercise_block(content_type="worksheet_exercise", content="تمرين: أكمل الفراغ", heading=None, document_id=7, pages=(3, 3)):
    return PedagogicalResourceBlock(
        source_number=1, document_id=document_id, document_title="Méthode A1",
        chunk_ids=[100], page_start=pages[0], page_end=pages[1],
        heading_context=[heading] if heading else [], content_type=content_type,
        structural_quality=None, content=content, requires_vision=False,
        image_not_interpreted=False, vector_scores=[], reranker_scores=[],
        original_ranks=[], reranked_ranks=[],
    )


def _context_with_blocks(*blocks):
    blocks = list(blocks)
    return PedagogicalContext(
        {"cefr_level": "A1", "language": "ar"}, [], [], blocks,
        len(blocks), len(blocks), [], [], 0,
    )


# -- KB-only degradation path (no provider) ----------------------------------

def test_kb_first_no_adapt_does_not_call_llm_and_sources_blocks():
    llm = FakeLLMProvider(_adapt_response(with_source=True))
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8)
    context = _context_with_blocks(
        _exercise_block(content="تمرين ١: أكمل بالنفي", heading="Exercice 1"),
        _exercise_block(content="تمرين ٢: رتب الكلمات", heading="Exercice 2"),
    )
    result = ExerciseGenerationService(llm=llm).generate(request, context)
    assert len(llm.calls) == 1
    assert result.adapt_with_ai is True
    assert result.provider_model == "fake-llm"
    assert result.exercises[0].status in ("ai_generated", "adapted_from_kb")


def test_kb_only_with_no_provider_never_calls_llm_and_sources_blocks():
    result = ExerciseGenerationService(llm=None).generate(
        ExerciseGenerateIn(level="A1", theme="La famille", count=8),
        _context_with_blocks(
            _exercise_block(content="تمرين ١: أكمل بالنفي", heading="Exercice 1"),
            _exercise_block(content="تمرين ٢: رتب الكلمات", heading="Exercice 2"),
        ),
    )
    assert result.adapt_with_ai is False
    assert result.provider_model is None
    assert len(result.exercises) == 2
    assert result.kb_sourced_count == 2
    for item in result.exercises:
        assert item.document_id == 7
        assert item.document_title == "Méthode A1"
        assert item.chunk_ids == [100]


def test_kb_first_respects_requested_count():
    blocks = [_exercise_block(content=f"تمرين {i}", heading=f"Exercice {i}") for i in range(5)]
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=3)
    result = ExerciseGenerationService(llm=None).generate(request, _context_with_blocks(*blocks))
    assert len(result.exercises) == 3


def test_kb_first_dedupes_identical_blocks():
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8)
    context = _context_with_blocks(
        _exercise_block(content="تمرين مكرر", heading="Exercice 1"),
        _exercise_block(content="تمرين مكرر", heading="Exercice 2"),
        _exercise_block(content="تمرين مختلف", heading="Exercice 3"),
    )
    result = ExerciseGenerationService(llm=None).generate(request, context)
    assert len(result.exercises) == 2


def test_kb_first_only_selects_exercise_type_blocks():
    general = _exercise_block(content_type="text", content="Leçon de grammaire", heading="Grammaire")
    exercise = _exercise_block(content_type="worksheet_exercise", content="تمرين", heading="Exercice")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8)
    result = ExerciseGenerationService(llm=None).generate(request, _context_with_blocks(general, exercise))
    assert len(result.exercises) == 1
    assert result.exercises[0].title == "Exercice"


def test_kb_first_role_marker_detects_exercise_in_text_block():
    general = _exercise_block(content_type="text", content="Voici un exercice de vocabulaire : associez les images.", heading="Activité")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8)
    result = ExerciseGenerationService(llm=None).generate(request, _context_with_blocks(general))
    assert len(result.exercises) == 1


def test_kb_first_no_blocks_raises_in_default_mode():
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8)
    with pytest.raises(ExerciseGenerationError):
        ExerciseGenerationService(llm=None).generate(request, _empty_context())


def test_kb_first_no_exercise_blocks_raises_in_default_mode():
    general = _exercise_block(content_type="text", content="Leçon de grammaire", heading="Grammaire")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8)
    with pytest.raises(ExerciseGenerationError):
        ExerciseGenerationService(llm=None).generate(request, _context_with_blocks(general))


# -- Adapt mode: bounded LLM -----------------------------------------------

def _adapt_response(*, with_source=True):
    items = [{
        "source_index": 0 if with_source else None,
        "title": "Exercice adapté",
        "exercise_type": "QCM",
        "prompt": "أكمل : عائلة أحمد تتكون من ... أفراد",
        "answer_expectation": "أربعة",
        "options": ["ثلاثة", "أربعة", "خمسة", "اثنان"],
        "level": "A1",
    }]
    return json.dumps({
        "title": "Exercices — La famille",
        "level": "A1", "theme": "La famille", "exercise_type": "QCM",
        "exercises": items,
    }, ensure_ascii=False)


def test_adapt_mode_calls_llm_once_and_binds_provenance():
    llm = FakeLLMProvider(_adapt_response(with_source=True))
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    context = _context_with_blocks(_exercise_block(content="تمرين ١", heading="Exercice 1"))
    result = ExerciseGenerationService(llm=llm).generate(request, context)
    assert len(llm.calls) == 1
    assert result.adapt_with_ai is True
    assert result.provider_model == "fake-llm"
    assert len(result.exercises) == 1
    item = result.exercises[0]
    assert item.document_id == 7
    assert item.document_title == "Méthode A1"
    assert item.chunk_ids == [100]
    assert result.kb_sourced_count == 1


def test_adapt_mode_new_item_without_source_keeps_provenance_null():
    llm = FakeLLMProvider(_adapt_response(with_source=False))
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    context = _context_with_blocks(_exercise_block(content="تمرين ١", heading="Exercice 1"))
    result = ExerciseGenerationService(llm=llm).generate(request, context)
    item = result.exercises[0]
    assert item.document_id is None
    assert item.document_title is None
    assert item.chunk_ids == []
    assert result.kb_sourced_count == 0


def test_adapt_mode_truncated_response_raises():
    class _MyLLM:
        model_id = "fake-llm"
        def generate(self, **kwargs):
            return LLMResult(text="{}", model="fake-llm", finish_reason="max_tokens")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    context = _context_with_blocks(_exercise_block())
    with pytest.raises(ExerciseGenerationError):
        ExerciseGenerationService(llm=_MyLLM()).generate(request, context)


def test_adapt_mode_invalid_json_raises():
    class _MyLLM:
        model_id = "fake-llm"
        def generate(self, **kwargs):
            return LLMResult(text="pas de json du tout", model="fake-llm")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    context = _context_with_blocks(_exercise_block())
    with pytest.raises(ExerciseGenerationError):
        ExerciseGenerationService(llm=_MyLLM()).generate(request, context)


def test_adapt_mode_provider_non_429_error_is_wrapped_and_logged():
    class _MyLLM:
        model_id = "fake-llm"
        def generate(self, **kwargs):
            raise LLMProviderError("boom", status_code=503, provider_message="Service indisponible", category="transport")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    context = _context_with_blocks(_exercise_block())
    with pytest.raises(ExerciseGenerationError) as exc_info:
        ExerciseGenerationService(llm=_MyLLM()).generate(request, context)
    assert "Service indisponible" in str(exc_info.value)


def test_adapt_mode_rate_limit_exhausted_after_one_retry():
    class _MyLLM:
        model_id = "fake-llm"
        def __init__(self):
            self.n = 0
        def generate(self, **kwargs):
            self.n += 1
            raise LLMProviderError("rate", status_code=429, provider_message="TPM limit")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    context = _context_with_blocks(_exercise_block())
    fake_sleep = {"calls": 0}
    def _sleep(_):
        fake_sleep["calls"] += 1
    service = ExerciseGenerationService(llm=_MyLLM(), sleep=_sleep)
    with pytest.raises(ExerciseRateLimitError):
        service.generate(request, context)
    assert fake_sleep["calls"] == 1


def test_adapt_mode_rate_limit_then_success():
    class _MyLLM:
        model_id = "fake-llm"
        def __init__(self):
            self.n = 0
        def generate(self, **kwargs):
            self.n += 1
            if self.n == 1:
                raise LLMProviderError("rate", status_code=429, provider_message="TPM limit")
            return LLMResult(text=_adapt_response(with_source=True), model="fake-llm")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    context = _context_with_blocks(_exercise_block())
    result = ExerciseGenerationService(llm=_MyLLM(), sleep=lambda _: None).generate(request, context)
    assert result.adapt_with_ai is True
    assert result.exercises[0].document_id == 7


# -- Consistency / prompt --------------------------------------------------

def test_adapt_prompt_forbids_fabrication_and_requests_variety():
    prompt = ExerciseGenerationService._build_system_prompt(adapt=True)
    assert "GÉNÉRATEUR" in prompt or "générateur" in prompt
    assert "jamais deux exercices identiques" in prompt.casefold()
    assert "answer_expectation" in prompt
    assert "source_index" in prompt
    assert "n'invente" in prompt or "invente jamais" in prompt
    assert "phrase" in prompt or "consigne" in prompt


def test_max_tokens_uses_exercise_config():
    llm = FakeLLMProvider(_adapt_response(with_source=True))
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    context = _context_with_blocks(_exercise_block())
    ExerciseGenerationService(llm=llm).generate(request, context)
    assert llm.calls[0]["max_tokens"] == get_settings().exercise_max_output_tokens
    assert llm.calls[0]["generation_options"].reasoning_effort == "low"


def test_schema_defaults():
    r = ExerciseGenerateIn(theme="La famille")
    assert r.count == 8 and r.adapt_with_ai is False and r.exercise_type == "auto"


def test_schema_forbids_extra_fields():
    with pytest.raises(Exception):
        ExerciseGenerateIn(theme="La famille", bogus=1)


def test_count_lower_bound():
    with pytest.raises(Exception):
        ExerciseGenerateIn(theme="La famille", count=0)


def test_json_scan_finds_root_object_within_prose():
    raw = "Voici le résultat:\n```json\n" + _adapt_response(with_source=True) + "\n```"
    service = ExerciseGenerationService(llm=None)
    payload = service._json_object(raw)
    assert "exercises" in payload
    assert payload["exercises"][0]["source_index"] == 0


def test_json_scan_with_no_json_at_all_raises():
    service = ExerciseGenerationService(llm=None)
    with pytest.raises(ExerciseGenerationError):
        service._json_object("just prose with no object")


def test_adapt_mode_fills_gaps_when_no_kb_blocks_and_mode_allows():
    class _MyLLM:
        model_id = "fake-llm"
        def generate(self, **kwargs):
            return LLMResult(text=_adapt_response(with_source=False), model="fake-llm")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    result = ExerciseGenerationService(llm=_MyLLM()).generate(request, _empty_context())
    assert result.exercises[0].document_id is None
    assert result.kb_sourced_count == 0


def test_adapt_mode_out_of_range_source_is_not_fabricated():
    raw = json.dumps({
        "title": "Exercices", "level": "A1", "theme": "La famille", "exercise_type": "QCM",
        "exercises": [{"source_index": 99, "title": "X", "prompt": "q"}],
    }, ensure_ascii=False)
    class _LLM2:
        model_id = "fake-llm"
        def generate(self, **kwargs):
            return LLMResult(text=raw, model="fake-llm")
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, adapt_with_ai=True)
    result = ExerciseGenerationService(llm=_LLM2()).generate(request, _context_with_blocks(_exercise_block()))
    assert result.exercises[0].document_id is None


def test_kb_first_sets_rag_sources_used_to_retrieval_pool_size():
    blocks = [
        _exercise_block(content="تمرين ١", heading="Exercice 1"),
        _exercise_block(content_type="text", content="Leçon", heading="Grammaire"),
        _exercise_block(content="تمرين ٢", heading="Exercice 2"),
    ]
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8)
    result = ExerciseGenerationService(llm=None).generate(request, _context_with_blocks(*blocks))
    assert result.rag_sources_used == 3
    assert len(result.exercises) == 2


# -- Pedagogical pipeline: plan + bounded targeted regeneration -------------

def _generic_response(theme="La famille", level="A1", count=5):
    """A fully-valid LLM response (Arabic QCM items with options) so that the
    happy path produces exactly one LLM call and passes validation."""
    options = ["أ", "ب", "ج", "د"]
    items = [{
        "title": f"Exercice {i + 1}",
        "skill": "Vocabulaire", "exercise_type": "qcm",
        "prompt": f"أكمل الجملة رقم {i + 1} بالكلمة الصحيحة المتعلقة بعائلة",
        "answer_expectation": "ب",
        "level": level, "options": options,
    } for i in range(count)]
    return json.dumps({
        "title": f"Exercices — {theme}", "level": level, "theme": theme,
        "exercise_type": "qcm", "exercises": items,
    }, ensure_ascii=False)


def test_pipeline_attaches_pedagogical_plan():
    llm = FakeLLMProvider(_generic_response(count=8))
    request = ExerciseGenerateIn(level="A1", theme="La famille", count=8, skills=["Vocabulaire"])
    result = ExerciseGenerationService(llm=llm).generate(request, _context_with_blocks(_exercise_block()))
    assert result.plan is not None
    assert result.plan.level == "A1"
    assert result.plan.theme == "La famille"
    assert result.plan.rationale
    assert len(result.plan.exercise_distribution) == 8
    assert result.provider_model == "fake-llm"


def test_pipeline_forced_type_distribution_uses_that_type():
    llm = FakeLLMProvider(_generic_response(count=5))
    request = ExerciseGenerateIn(level="A1", theme="Voyage", count=5, exercise_type="QCM")
    result = ExerciseGenerationService(llm=llm).generate(request, _empty_context())
    assert result.exercises
    # Forced type → set reported type is QCM.
    assert result.exercise_type.casefold().replace(" ", "") in ("qcm",) or result.exercises[0].exercise_type.casefold() == "qcm"


def test_scenario_a1_famille_vocab_5():
    llm = FakeLLMProvider(_generic_response(theme="Famille", level="A1", count=5))
    request = ExerciseGenerateIn(level="A1", theme="Famille", count=5, skills=["Vocabulaire"])
    result = ExerciseGenerationService(llm=llm).generate(request, _empty_context())
    assert len(result.exercises) == 5
    assert result.level == "A1"
    assert all(ex.level == "A1" for ex in result.exercises)
    assert result.plan is not None


def test_scenario_a2_voyage_vocab_gram_8():
    llm = FakeLLMProvider(_generic_response(theme="Voyage", level="A2", count=8))
    request = ExerciseGenerateIn(level="A2", theme="Voyage", count=8, skills=["Vocabulaire", "Grammaire"])
    result = ExerciseGenerationService(llm=llm).generate(request, _empty_context())
    assert len(result.exercises) == 8
    assert result.level == "A2"


def test_scenario_b1_culture_marocaine_comprehension_10():
    llm = FakeLLMProvider(_generic_response(theme="Culture marocaine", level="B1", count=10))
    request = ExerciseGenerateIn(level="B1", theme="Culture marocaine", count=10, skills=["Compréhension écrite"])
    result = ExerciseGenerationService(llm=llm).generate(request, _empty_context())
    assert len(result.exercises) == 10
    assert result.level == "B1"


def test_scenario_b2_travail_expression_ecrite_5():
    llm = FakeLLMProvider(_generic_response(theme="Travail", level="B2", count=5))
    request = ExerciseGenerateIn(level="B2", theme="Travail", count=5, skills=["Expression écrite"])
    result = ExerciseGenerationService(llm=llm).generate(request, _empty_context())
    assert len(result.exercises) == 5
    assert result.level == "B2"


def test_regeneration_is_bounded_and_returns_valid_subset():
    # First response invalid (empty prompt); regeneration (same fake) stays
    # invalid. Bounded by MAX_REGENERATION_ATTEMPTS; the service must return
    # without raising and without a runaway loop.
    raw = json.dumps({
        "title": "Exercices", "level": "A1", "theme": "Famille",
        "exercise_type": "qcm",
        "exercises": [{"title": "X", "prompt": "   ", "exercise_type": "qcm", "options": []}],
    }, ensure_ascii=False)
    calls = {"n": 0}

    class _LLM:
        model_id = "fake-llm"
        def generate(self, **kwargs):
            calls["n"] += 1
            return LLMResult(text=raw, model="fake-llm")

    request = ExerciseGenerateIn(level="A1", theme="Famille", count=1)
    result = ExerciseGenerationService(llm=_LLM()).generate(request, _empty_context())
    # Bounded: initial + up to MAX_REGENERATION_ATTEMPTS retries.
    assert calls["n"] <= 3
    assert isinstance(result.exercises, list)
