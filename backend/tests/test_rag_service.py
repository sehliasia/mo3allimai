from types import SimpleNamespace

from app.services.context_builder import ContextBuilder
from app.services.llm_provider import FakeLLMProvider
from app.services.rag_prompt_builder import RAGPromptBuilder
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalResult
from app.core.config import get_settings


def _result(*, vision=False):
    return RetrievalResult(
        rank=1, score=0.9, vector_score=0.9, original_rank=1, chunk_id=42, document_id=15,
        document_title="wc-lesson-plans", source_page_start=4, source_page_end=4, content_type="paragraph",
        language="en", cefr_level="A1", structural_quality="structured", has_image=vision, requires_vision=vision,
        heading_context=["Lesson Goals"], content="Identify the countries who qualified for the World Cup.",
    )


class FakeRetrieval:
    def __init__(self, results): self.results, self.calls = results, []
    def search(self, db, query, **kwargs):
        self.calls.append((query, kwargs))
        return SimpleNamespace(results=self.results, model="test-embedding", reranking_applied=kwargs["rerank"], candidate_top_k=20, stale_references_skipped=0)


def _service(results, llm=None):
    return RAGService(retrieval=FakeRetrieval(results), context_builder=ContextBuilder(max_chunks=6, max_tokens=300, neighbor_expansion=False), prompt_builder=RAGPromptBuilder(), llm=llm or FakeLLMProvider("Réponse fondée [S1]."))


def test_rag_uses_grounded_canonical_context_and_trusted_source_metadata_for_french():
    llm = FakeLLMProvider("Réponse fondée [S1].")
    service = _service([_result()], llm)
    response = service.answer_query(None, "Quels sont les objectifs ?", document_ids=[15])
    assert response.answer == "Réponse fondée [S1]."
    assert response.sources[0].document_id == 15 and response.sources[0].page_start == 4 and response.used_chunk_ids == [42]
    assert "Answer in French." in llm.calls[0]["system_prompt"]
    assert "Identify the countries" in llm.calls[0]["user_prompt"]
    assert "QUESTION:\nQuels sont les objectifs ?" in llm.calls[0]["user_prompt"]
    assert "Qdrant" not in llm.calls[0]["user_prompt"]
    assert llm.calls[0]["max_tokens"] == get_settings().rag_llm_max_tokens


def test_rag_detects_arabic_and_english_without_changing_queries_and_honors_reranker_flag():
    llm = FakeLLMProvider()
    service = _service([_result()], llm)
    service.answer_query(None, "ما هي الأهداف؟", use_reranker=False)
    service.answer_query(None, "What are the lesson goals?", use_reranker=True)
    assert "Answer in Arabic." in llm.calls[0]["system_prompt"]
    assert "ما هي الأهداف؟" in llm.calls[0]["user_prompt"]
    assert "Answer in English." in llm.calls[1]["system_prompt"]
    assert service.retrieval.calls[0][1]["rerank"] is False
    assert service.retrieval.calls[1][1]["rerank"] is True


def test_rag_returns_controlled_insufficient_answer_without_calling_llm():
    llm = FakeLLMProvider()
    response = _service([], llm).answer_query(None, "Quelle est la compétence ?")
    assert response.answer == "Les sources disponibles ne permettent pas de répondre précisément à cette question."
    assert response.sources == [] and llm.calls == []


def test_rag_preserves_vision_warning_without_claiming_image_interpretation():
    llm = FakeLLMProvider()
    response = _service([_result(vision=True)], llm).answer_query(None, "What is shown?")
    assert response.has_requires_vision is True
    assert response.warnings
    assert "not interpreted" in llm.calls[0]["system_prompt"]


def test_exact_arabic_query_survives_prompt_building_unchanged():
    query = "ما أهداف درس الدول المتأهلة لكأس العالم؟"
    llm = FakeLLMProvider()
    _service([_result()], llm).answer_query(None, query)
    assert "Answer in Arabic." in llm.calls[0]["system_prompt"]
    assert f"QUESTION:\n{query}" in llm.calls[0]["user_prompt"]
