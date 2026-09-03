"""LLM exercise extraction: semantic detection, multi-chunk reassembly, strict
k-2 provenance, CECR-hard level gate, JSON robustness and dedup. The retrieval
itself is never the LLM's job: these tests exercise only the extraction layer."""

import json

import pytest

from app.core.config import Settings
from app.schemas.exercise_generator import ExerciseSearchIn
from app.services.context_builder import ContextSourceBlock
from app.services.exercise_extraction_service import (
    ExerciseExtractionError,
    ExerciseExtractionService,
)
from app.services.llm_provider import LLMProviderError, LLMResult


def _block(source_number, *, document_id=100, document_title="Manuel X",
           chunk_ids=(41,), page_start=12, page_end=13,
           heading_context=("Niveau A2", "La famille"), content="Activité",
           content_type="worksheet_exercise", structural_quality=None):
    return ContextSourceBlock(
        source_number=source_number, document_id=document_id,
        document_title=document_title, chunk_ids=list(chunk_ids),
        page_start=page_start, page_end=page_end,
        heading_context=list(heading_context), content_type=content_type,
        structural_quality=structural_quality, has_image=False,
        requires_vision=False, image_not_interpreted=False,
        vector_scores=[0.5], reranker_scores=[0.5], original_ranks=[1],
        reranked_ranks=[1], content=content, estimated_token_count=len(content) // 4,
    )


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


def _service(llm):
    return ExerciseExtractionService(llm=llm, settings=Settings())


def _search(query="Exercices A1", level=None, source_ids=None):
    kwargs = dict(query=query, limit=24)
    if level:
        kwargs["level"] = level
    if source_ids:
        kwargs["source_document_ids"] = source_ids
    return ExerciseSearchIn(**kwargs)


def _extraction_json(exercises):
    if exercises is None:
        return json.dumps({"exercises": []})
    return json.dumps({"exercises": exercises}, ensure_ascii=False)


# -- A real exercise is detected, structured and triaged --------------------

def test_extracts_real_exercise_and_keeps_provenance():
    blocks = [
        _block(1, chunk_ids=[41], page_start=12, heading_context=("Niveau A1", "La famille"),
               content="Complète les phrases suivantes.\n1. اسمي ______.\n2. عمري ______ سنوات."),
    ]
    answer = {
        "is_exercise": True, "title": "Se présenter", "instruction": "Complète les phrases suivantes.",
        "exercise_type": "fill_blank",
        "items": [{"number": 1, "content": "اسمي ______."}, {"number": 2, "content": "عمري ______ سنوات."}],
        "expected_answer": None, "level": None, "level_source": "inferred", "theme": "présentation",
        "skill": "production écrite",
        "source": {"document_id": 100, "document_name": "Manuel X", "pages": [12], "chunk_ids": [41]},
    }
    llm = _FakeLLM([_extraction_json([answer])])
    items, calls = _service(llm).extract(blocks, request=_search(level="A1"))
    assert calls == 1
    assert len(items) == 1
    ex = items[0]
    assert ex.status == "kb_original"
    assert ex.document_id == 100
    assert ex.document_title == "Manuel X"
    assert ex.chunk_ids == [41]
    assert ex.page_start == 12
    assert ex.exercise_type == "fill_blank"
    assert "اسمي ______." in ex.prompt


def test_invalid_json_raises_extraction_error():
    llm = _FakeLLM(["this is not json"])
    with pytest.raises(ExerciseExtractionError):
        _service(llm).extract([_block(1, content="Complète.")], request=_search())


def test_valid_json_without_exercise_is_empty_not_error():
    llm = _FakeLLM([_extraction_json(None)])
    items, calls = _service(llm).extract([_block(1, content="Complète.")], request=_search())
    assert items == []
    assert calls == 1


def test_marks_descriptive_passage_as_not_exercise():
    blocks = [_block(1, content="Ce manuel contient plusieurs exercices destinés aux apprenants.")]
    answer = {"is_exercise": False}
    llm = _FakeLLM([_extraction_json([answer])])
    items, _ = _service(llm).extract(blocks, request=_search())
    assert items == []


def test_partially_extracted_exercise_is_captured_without_fabrication():
    # Instruction present, no items and invented provenance ids are dropped.
    blocks = [_block(1, chunk_ids=[41], document_id=100, content="Lis puis réponds. Qui est le père ?")]
    answer = {
        "is_exercise": True, "instruction": "Lis puis réponds.", "exercise_type": "open_question",
        "items": [], "expected_answer": None,
        "source": {"document_id": 9999, "document_name": "Inventé", "pages": [999], "chunk_ids": [99999]},
    }
    llm = _FakeLLM([_extraction_json([answer])])
    items, _ = _service(llm).extract(blocks, request=_search())
    assert len(items) == 1
    assert items[0].document_id == 100  # never the invented 9999
    assert items[0].document_title == "Manuel X"
    assert items[0].chunk_ids == [41]   # falls back to the real block


def test_a2_explicit_source_is_never_relabelled_a1():
    blocks = [_block(1, heading_context=["Niveau A2"], content="Lis puis réponds aux questions.")]
    answer = {
        "is_exercise": True, "instruction": "Lis puis réponds aux questions.",
        "exercise_type": "open_question", "level": None,
        "source": {"document_id": 100, "document_name": "Manuel X", "pages": [12], "chunk_ids": [41]},
    }
    llm = _FakeLLM([_extraction_json([answer])])
    items, _ = _service(llm).extract(blocks, request=_search(level="A1"))
    # The strict gate drops a provably-A2 source from an A1 request rather than
    # silently turning it into A1.
    assert items == []


# -- Multi-chunk reassembly -------------------------------------------------

def test_exercise_spread_across_chunks_is_one_item():
    blocks = [
        _block(1, chunk_ids=[41], page_start=12, content="Observe l'image et lis le dialogue."),
        _block(2, chunk_ids=[42], page_start=12, content="ثم أجب عن الأسئلة التالية.\n1. من هذا؟\n2. أين هو؟"),
        _block(3, chunk_ids=[43], page_start=13, content="3. ماذا يفعل؟"),
    ]
    answer = {
        "is_exercise": True, "instruction": "Observe l'image et lis le dialogue. ثم أجب عن الأسئلة التالية.",
        "exercise_type": "open_question",
        "items": [{"number": 1, "content": "من هذا؟"}, {"number": 2, "content": "أين هو؟"}, {"number": 3, "content": "ماذا يفعل؟"}],
        "source": {"document_id": 100, "document_name": "Manuel X", "pages": [12, 13], "chunk_ids": [41, 42, 43]},
    }
    llm = _FakeLLM([_extraction_json([answer])])
    items, _ = _service(llm).extract(blocks, request=_search())
    assert len(items) == 1             # one exercise, not three
    assert items[0].chunk_ids == [41, 42, 43]
    assert items[0].page_start == 12
    assert "من هذا؟" in items[0].prompt


# -- Deduplication ----------------------------------------------------------

def test_same_exercise_in_two_chunks_deduplicates():
    blocks = [
        _block(1, chunk_ids=[41], content="Complète.\n1. أين ياسين ؟"),
        _block(2, chunk_ids=[51], content="Complète.\n1. أين ياسين ؟"),
    ]
    dup = {
        "is_exercise": True, "instruction": "Complète.", "exercise_type": "fill_blank",
        "items": [{"number": 1, "content": "أين ياسين ؟"}],
        "source": {"document_id": 100, "document_name": "Manuel X", "chunk_ids": [41]},
    }
    llm = _FakeLLM([_extraction_json([dup, dict(dup, source={"document_id": 100, "document_name": "Manuel X", "chunk_ids": [51]})])])
    items, _ = _service(llm).extract(blocks, request=_search())
    assert len(items) == 1


# -- LLM call accounting and retries ----------------------------------------

def test_extraction_uses_llm_only_for_analysis_not_retrieval():
    llm = _FakeLLM([_extraction_json([{"is_exercise": False}])])
    _service(llm).extract([_block(1, content="Complète.")], request=_search())
    assert len(llm.calls) == 1
    response = llm.calls[0]
    # The system prompt is the extraction role, never a retrieval role.
    assert "UNIQUEMENT d'analyser les passages" in response["system_prompt"]
    assert "retrieve" not in response["system_prompt"].lower()


def test_429_retried_once_then_rate_limit_error():
    from app.services.exercise_extraction_service import ExerciseExtractionRateLimitError
    llm = _FakeLLM(
        responses=[_extraction_json([{"is_exercise": False}])],
        errors=[LLMProviderError("429", status_code=429), LLMProviderError("429", status_code=429)],
    )
    with pytest.raises(ExerciseExtractionRateLimitError):
        _service(llm).extract([_block(1, content="Complète.")], request=_search())
    assert len(llm.calls) == 2


def test_extraction_json_parses_md_and_invalid_json_falls_back():
    from app.services.exercise_extraction_service import _parse_extraction_json
    assert _parse_extraction_json("```json\n{\"exercises\": []}\n```") == {"exercises": []}
    with pytest.raises(ExerciseExtractionError):
        _parse_extraction_json("no json at all")