from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.base import Base
from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.services.document_chunk import DocumentChunk
from app.services.knowledge_ingestion_service import KnowledgeIngestionService
from app.services.retrieval_service import EmptyQueryError, RetrievalFilters, RetrievalService
from app.services.reranker_service import RerankScore


class FakeQueryProvider:
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    dimension = 1024

    def __init__(self, *, dimension=1024):
        self.dimension = dimension
        self.query_calls: list[list[str]] = []
        self.document_calls = 0

    def embed_queries(self, queries):
        self.query_calls.append(list(queries))
        return [[0.5] * self.dimension for _ in queries]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[0.5] * self.dimension for _ in texts]

    def embed_texts(self, texts):
        return self.embed_documents(texts)


class FakeQdrant:
    dimension = 1024
    collection_name = "knowledge-test"

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search_points(self, vector, **kwargs):
        self.calls.append((vector, kwargs))
        return self.hits


class FakeReranker:
    model_id = "Qwen/Qwen3-Reranker-0.6B"

    def __init__(self, scores=None, fail=False):
        self.scores = scores or []
        self.fail = fail
        self.calls = []

    def rerank(self, query, documents, top_k):
        self.calls.append((query, documents, top_k))
        if self.fail:
            raise RuntimeError("offline")
        return [RerankScore(index=index, score=score) for index, score in self.scores[:top_k]]


def _db(tmp_path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'retrieval.db'}")
    Base.metadata.create_all(engine)
    return Session(engine)


def _persist(db, tmp_path):
    document = KnowledgeDocument(
        title="Document canonique",
        original_filename="guide.pdf",
        stored_filename="guide.pdf",
        file_path=str(tmp_path / "never-expose.pdf"),
        mime_type="application/pdf",
        file_size=1,
        uploaded_by=1,
        language="ar",
        cefr_level="A1",
    )
    db.add(document)
    db.commit()
    chunks = [
        DocumentChunk(
            id="first",
            document_id=document.id,
            chunk_index=0,
            text_original="Texte canonique détaillé",
            text_for_embedding="texte vectoriel différent",
            page_start=4,
            page_end=5,
            section=None,
            headings=[],
            content_type="table",
            metadata={"structural_quality": "partially_structured", "has_image": True, "requires_vision": True},
            token_count=5,
        )
    ]
    KnowledgeIngestionService()._replace_persisted_chunks(db, document, chunks, quality_status="complete")
    db.commit()
    return document, db.scalar(select(KnowledgeChunk))


def test_retrieval_uses_query_path_preserves_ranking_and_hydrates_canonical_content(tmp_path):
    db = _db(tmp_path)
    document, chunk = _persist(db, tmp_path)
    hits = [
        SimpleNamespace(score=0.99, payload={"chunk_id": 999999}),
        SimpleNamespace(score=0.87, payload={"chunk_id": chunk.id}),
    ]
    provider = FakeQueryProvider()
    qdrant = FakeQdrant(hits)
    response = RetrievalService(provider=provider, qdrant=qdrant, mode="dense").search(
        db,
        "  ما هي الأهداف؟  ",
        top_k=5,
        filters=RetrievalFilters(document_ids=[document.id], language="ar", cefr_level="A1", content_type="table", requires_vision=True),
    )
    assert provider.query_calls == [["ما هي الأهداف؟"]]
    assert provider.document_calls == 0
    assert qdrant.calls[0][1]["document_ids"] == [document.id]
    assert qdrant.calls[0][1]["language"] == "ar"
    assert qdrant.calls[0][1]["cefr_level"] == "A1"
    assert qdrant.calls[0][1]["content_type"] == "table"
    assert qdrant.calls[0][1]["requires_vision"] is True
    assert response.stale_references_skipped == 1
    assert len(response.results) == 1
    result = response.results[0]
    assert result.rank == 2 and result.score == 0.87
    assert result.content == "Texte canonique détaillé"
    assert result.requires_vision is True and result.has_image is True
    assert not hasattr(result, "vector") and "never-expose" not in result.content


def test_empty_query_and_invalid_embedding_dimension_are_rejected_without_search(tmp_path):
    db = _db(tmp_path)
    qdrant = FakeQdrant([])
    provider = FakeQueryProvider()
    service = RetrievalService(provider=provider, qdrant=qdrant)
    with pytest.raises(EmptyQueryError):
        service.search(db, "   ")
    assert qdrant.calls == []

    wrong_provider = FakeQueryProvider(dimension=8)
    with pytest.raises(ValueError):
        RetrievalService(provider=wrong_provider, qdrant=qdrant)


def test_no_results_returns_an_empty_canonical_response(tmp_path):
    db = _db(tmp_path)
    response = RetrievalService(provider=FakeQueryProvider(), qdrant=FakeQdrant([])).search(db, "CECR")
    assert response.results == []
    assert response.stale_references_skipped == 0


def test_reranking_uses_canonical_multilingual_content_and_preserves_vector_ranks(tmp_path):
    db = _db(tmp_path)
    document, first = _persist(db, tmp_path)
    second = KnowledgeChunk(
        document_id=document.id,
        chunk_index=1,
        content="Instruction française : décrivez les équipes.",
        content_for_embedding="not the canonical source",
        chunk_hash="second-content-hash",
        source_page_start=6,
        source_page_end=6,
        heading_context=["Leçon", "Objectifs"],
        content_type="paragraph",
        chunk_metadata={"requires_vision": False},
        token_count=8,
        quality_status="accepted",
        ingestion_version="test",
        embedding_status="pending",
    )
    db.add(second)
    db.commit()
    reranker = FakeReranker(scores=[(1, 0.99), (0, 0.20)])
    qdrant = FakeQdrant(
        [
            SimpleNamespace(score=0.90, payload={"chunk_id": first.id}),
            SimpleNamespace(score=0.80, payload={"chunk_id": second.id}),
        ]
    )
    response = RetrievalService(
        provider=FakeQueryProvider(), qdrant=qdrant, reranker=reranker, candidate_top_k=20, final_top_k=5
    ).search(db, "ما هي objectifs ?", candidate_top_k=20, top_k=2, rerank=True)
    assert qdrant.calls[0][1]["top_k"] == 20
    assert response.reranking_applied is True and response.candidate_top_k == 20
    assert [result.chunk_id for result in response.results] == [second.id, first.id]
    assert response.results[0].original_rank == 2
    assert response.results[0].vector_score == 0.80
    assert response.results[0].reranked_rank == 1 and response.results[0].reranker_score == 0.99
    assert response.results[1].requires_vision is True
    context = reranker.calls[0][1][1]
    assert "Document: Document canonique" in context
    assert "Section: Leçon > Objectifs" in context
    assert "Content:\nInstruction française" in context and "not the canonical source" not in context


def test_reranker_failure_returns_the_original_vector_order(tmp_path):
    db = _db(tmp_path)
    document, first = _persist(db, tmp_path)
    second = KnowledgeChunk(
        document_id=document.id, chunk_index=1, content="ثانٍ", content_for_embedding="ثانٍ", chunk_hash="second-fallback",
        source_page_start=6, source_page_end=6, heading_context=[], content_type="paragraph", chunk_metadata={},
        token_count=1, quality_status="accepted", ingestion_version="test", embedding_status="pending",
    )
    db.add(second)
    db.commit()
    response = RetrievalService(
        provider=FakeQueryProvider(),
        qdrant=FakeQdrant([
            SimpleNamespace(score=0.90, payload={"chunk_id": first.id}),
            SimpleNamespace(score=0.80, payload={"chunk_id": second.id}),
        ]),
        reranker=FakeReranker(fail=True),
    ).search(db, "requête française", top_k=1, rerank=True)
    assert response.reranking_applied is False
    assert response.reranker_error == "Reranker unavailable; returned vector-ranked results."
    assert [result.chunk_id for result in response.results] == [first.id]
