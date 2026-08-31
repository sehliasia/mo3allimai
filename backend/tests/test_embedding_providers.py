from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.database.base import Base
from app.models.knowledge_document import KnowledgeChunk, KnowledgeChunkEmbeddingStatus, KnowledgeDocument
from app.services.embedding_providers.factory import get_embedding_provider
from app.services.embedding_providers.qwen3_provider import Qwen3EmbeddingProvider
from app.services.embedding_providers.sentence_transformer_provider import (
    EmbeddingDimensionError,
    EmbeddingProviderError,
    SentenceTransformerEmbeddingProvider,
)
from app.services.document_chunk import DocumentChunk
from app.services.knowledge_ingestion_service import KnowledgeIngestionService


class FakeSentenceModel:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.fail = fail

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if self.fail:
            raise self.fail
        return [[float(index + 1)] * kwargs["truncate_dim"] for index, _ in enumerate(texts)]


def test_qwen_documents_remain_unchanged_and_queries_use_instruction():
    SentenceTransformerEmbeddingProvider._models.clear()
    model = FakeSentenceModel()
    provider = Qwen3EmbeddingProvider(
        model_id="qwen-test",
        dimension=1024,
        device="cpu",
        query_instruction="Retrieve multilingual educational passages.",
        model_loader=lambda *_: model,
    )
    document_texts = ["الأسرة باللغة العربية", "Descripteurs du CECR", "Classroom worksheet"]
    provider.embed_documents(document_texts)
    provider.embed_queries(["أنشطة الأسرة"])

    assert model.calls[0][0] == document_texts
    assert model.calls[1][0] == [
        "Instruct: Retrieve multilingual educational passages.\nQuery: أنشطة الأسرة"
    ]
    assert model.calls[0][1]["truncate_dim"] == 1024
    assert model.calls[0][1]["normalize_embeddings"] is True


def test_qwen_loads_once_and_reuses_the_model_for_batches():
    SentenceTransformerEmbeddingProvider._models.clear()
    loader_calls = []
    model = FakeSentenceModel()

    def loader(*args):
        loader_calls.append(args)
        return model

    first = Qwen3EmbeddingProvider(model_id="qwen-test", dimension=1024, device="cpu", batch_size=2, model_loader=loader)
    second = Qwen3EmbeddingProvider(model_id="qwen-test", dimension=1024, device="cpu", batch_size=2, model_loader=loader)
    first.embed_documents(["a", "b"])
    second.embed_documents(["c"])
    assert len(loader_calls) == 1
    assert model.calls[0][1]["batch_size"] == 2


def test_qwen_exposes_load_and_encode_timing_without_reloading_the_cached_model():
    SentenceTransformerEmbeddingProvider._models.clear()
    provider = Qwen3EmbeddingProvider(
        model_id="qwen-diagnostics", dimension=1024, device="cpu", model_loader=lambda *_: FakeSentenceModel(),
    )

    provider.embed_queries(["famille"])
    cold = provider.last_diagnostics()
    provider.embed_queries(["école"])
    warm = provider.last_diagnostics()

    assert cold["embedding_model_cache_hit"] is False
    assert isinstance(cold["embedding_model_load_ms"], int)
    assert cold["embedding_tokenization_ms"] is None
    assert isinstance(cold["embedding_inference_ms"], int)
    assert warm["embedding_model_cache_hit"] is True and warm["embedding_model_load_ms"] == 0


def test_qwen_rejects_wrong_dimension_and_surfaces_batch_failure():
    SentenceTransformerEmbeddingProvider._models.clear()
    wrong_dimension = type("WrongDimension", (), {"encode": lambda self, *args, **kwargs: [[1.0]]})()
    provider = Qwen3EmbeddingProvider(model_id="qwen-wrong", dimension=1024, device="cpu", model_loader=lambda *_: wrong_dimension)
    with pytest.raises(EmbeddingDimensionError):
        provider.embed_documents(["texte"])

    SentenceTransformerEmbeddingProvider._models.clear()
    provider = Qwen3EmbeddingProvider(
        model_id="qwen-oom",
        dimension=1024,
        device="cpu",
        model_loader=lambda *_: FakeSentenceModel(fail=RuntimeError("CUDA out of memory")),
    )
    with pytest.raises(EmbeddingProviderError):
        provider.embed_documents(["texte"])


def test_qwen_factory_is_lazy_and_uses_the_production_1024_configuration():
    provider = get_embedding_provider(
        Settings(
            database_url="sqlite:///test.db",
            jwt_secret_key="test",
            rag_embedding_provider="qwen3",
            rag_embedding_model_id="Qwen/Qwen3-Embedding-0.6B",
            rag_embedding_dimension=1024,
            rag_embedding_device="auto",
            rag_embedding_batch_size=8,
        )
    )
    assert isinstance(provider, Qwen3EmbeddingProvider)
    assert provider.dimension == 1024
    assert provider.device == "auto"
    assert provider._model is None


def test_auto_device_prefers_cuda_only_when_available(monkeypatch):
    import torch

    provider = Qwen3EmbeddingProvider(model_id="qwen-device", dimension=1024, device="auto", model_loader=lambda *_: FakeSentenceModel())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert provider._resolved_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert provider._resolved_device() == "cuda"


def test_provider_failure_does_not_change_embedding_state(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'embedding-provider.db'}")
    Base.metadata.create_all(engine)
    db = Session(engine)
    document = KnowledgeDocument(
        title="Guide",
        original_filename="guide.pdf",
        stored_filename="guide.pdf",
        file_path=str(tmp_path / "guide.pdf"),
        mime_type="application/pdf",
        file_size=1,
        uploaded_by=1,
    )
    db.add(document)
    db.commit()
    KnowledgeIngestionService()._replace_persisted_chunks(
        db,
        document,
        [
            DocumentChunk(
                id="chunk:0",
                document_id=document.id,
                chunk_index=0,
                text_original="محتوى صالح",
                text_for_embedding="محتوى صالح",
                page_start=1,
                page_end=1,
                section=None,
                headings=[],
                content_type="text",
                metadata={"requires_vision": True},
                token_count=3,
            )
        ],
        quality_status="complete",
    )
    db.commit()

    SentenceTransformerEmbeddingProvider._models.clear()
    provider = Qwen3EmbeddingProvider(
        model_id="qwen-failure",
        dimension=1024,
        device="cpu",
        model_loader=lambda *_: FakeSentenceModel(fail=RuntimeError("out of memory")),
    )
    with pytest.raises(EmbeddingProviderError):
        provider.embed_documents(["محتوى صالح"])
    assert db.scalar(select(KnowledgeChunk.embedding_status)) == KnowledgeChunkEmbeddingStatus.pending
