"""Explicit per-exercise AI adaptation: provenance kept, original level kept,
status correct, bounded LLM, Arabic guard."""

import json

import pytest

from app.schemas.exercise_generator import ExerciseAdaptIn, ExerciseItem
from app.services.exercise_adaptation_service import (
    ExerciseAdaptationError,
    ExerciseAdaptationRateLimitError,
    ExerciseAdaptationService,
)
from app.services.llm_provider import LLMProviderError, LLMResult


def _source(**overrides):
    data = dict(
        title="Exercice 4", skill="Vocabulaire", exercise_type="fill_blank",
        prompt="Complète avec le mot correct.\n1. هذا ___ الرجل.",
        context="", answer_expectation="طيب", level="A2", level_source="explicit",
        theme="La famille", status="kb_original", document_title="Miftah 2",
        document_id=9, page_start=35, page_end=35, chunk_ids=[301],
        heading_context=["Exercices A2"],
    )
    data.update(overrides)
    return ExerciseItem(**data)


def _adapt_json(**item):
    payload = {
        "title": "Exercice adapté pour A1",
        "prompt": "هذا رجل طيب.",
        "context": "",
        "answer_expectation": "طيب",
        "difficulty": "easy",
    }
    payload.update(item)
    return json.dumps(payload, ensure_ascii=False)


class _FakeLLM:
    model_id = "fake-llm"

    def __init__(self, responses, errors=None):
        self.responses = list(responses)
        self.errors = list(errors or [])
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return LLMResult(text=self.responses.pop(0), model=self.model_id)


def test_adapt_calls_llm_once_keeps_level_and_provenance():
    llm = _FakeLLM([_adapt_json()])
    result = ExerciseAdaptationService(llm=llm).adapt(ExerciseAdaptIn(
        source=_source(), target_level="A1", language="ar",
    ))
    assert len(llm.calls) == 1
    assert result.status == "adapted_from_kb"
    assert result.level == "A1"          # target level
    assert result.level_source == "generated"
    assert result.original_level == "A2"  # original level is kept, never lost
    assert result.original_document_title == "Miftah 2"
    assert result.original_document_id == 9
    assert result.original_chunk_ids == [301]
    assert result.document_title == "Miftah 2"  # lineage to the KB source
    assert result.document_id == 9
    assert result.prompt == "هذا رجل طيب."
    assert result.difficulty == "easy"


def test_adapt_system_prompt_contains_internal_cefr_rules_not_official_citation():
    from app.services.exercise_adaptation_service import ExerciseAdaptationService
    prompt = ExerciseAdaptationService._build_system_prompt()
    assert "RÈGLES D'ADAPTATION CECRL INTERNES" in prompt
    assert "pas une citation" in prompt
    assert "A1" in prompt and "C2" in prompt


def test_adapt_rejects_source_without_kb_provenance():
    with pytest.raises(ExerciseAdaptationError):
        ExerciseAdaptationService(llm=_FakeLLM([_adapt_json()])).adapt(ExerciseAdaptIn(
            source=_source(document_id=None, status="ai_generated"),
            target_level="A1",
        ))


def test_adapt_arabic_guard_rejects_latin_prompt():
    llm = _FakeLLM([_adapt_json(prompt="Complete the sentence with the correct word.")])
    with pytest.raises(ExerciseAdaptationError):
        ExerciseAdaptationService(llm=llm).adapt(ExerciseAdaptIn(
            source=_source(), target_level="A1", language="ar",
        ))


def test_adapt_empty_prompt_raises():
    llm = _FakeLLM([_adapt_json(prompt="", answer_expectation="x")])
    with pytest.raises(ExerciseAdaptationError):
        ExerciseAdaptationService(llm=llm).adapt(ExerciseAdaptIn(
            source=_source(), target_level="A1", language="fr",
        ))


def test_adapt_retries_single_429_then_succeeds():
    llm = _FakeLLM(
        responses=[_adapt_json()],
        errors=[LLMProviderError("rate limited", status_code=429)],
    )
    result = ExerciseAdaptationService(llm=llm).adapt(ExerciseAdaptIn(
        source=_source(), target_level="A1", language="ar",
    ))
    assert result.status == "adapted_from_kb"
    assert len(llm.calls) == 2


def test_adapt_429_exhausts_retries_with_rate_limit_error():
    llm = _FakeLLM(
        responses=[_adapt_json()],
        errors=[LLMProviderError("429", status_code=429), LLMProviderError("429", status_code=429)],
    )
    with pytest.raises(ExerciseAdaptationRateLimitError):
        ExerciseAdaptationService(llm=llm).adapt(ExerciseAdaptIn(
            source=_source(), target_level="A1", language="ar",
        ))
    assert len(llm.calls) == 2


def test_adapt_non_429_provider_error_wraps():
    llm = _FakeLLM(responses=[], errors=[LLMProviderError("boom", status_code=502)])
    with pytest.raises(ExerciseAdaptationError):
        ExerciseAdaptationService(llm=llm).adapt(ExerciseAdaptIn(
            source=_source(), target_level="A1", language="ar",
        ))


def test_adapt_invalid_json_raises():
    llm = _FakeLLM(["just prose with no object"])
    with pytest.raises(ExerciseAdaptationError):
        ExerciseAdaptationService(llm=llm).adapt(ExerciseAdaptIn(
            source=_source(), target_level="A1", language="ar",
        ))


def test_adapt_requires_llm_provider():
    with pytest.raises(ValueError):
        ExerciseAdaptationService(llm=None)