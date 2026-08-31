from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.database.base import Base
from app.models.knowledge_document import KnowledgeChunk, KnowledgeChunkEmbeddingStatus, KnowledgeDocument
from app.services.document_chunk import DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.knowledge_embedding_indexer import KnowledgeEmbeddingIndexer
from app.services.knowledge_ingestion_service import KnowledgeIngestionService
from app.services.qdrant_service import QdrantCollectionCompatibilityError, QdrantService


class NotFound(Exception):
    status_code = 404


class FakeModels:
    class Distance:
        COSINE = "cosine"

    class PayloadSchemaType:
        INTEGER = "integer"
        KEYWORD = "keyword"
        BOOL = "bool"

    class VectorParams:
        def __init__(self, *, size, distance):
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, *, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    class PointIdsList:
        def __init__(self, *, points):
            self.points = points

    class MatchValue:
        def __init__(self, *, value):
            self.value = value

    class MatchAny:
        def __init__(self, *, any):
            self.any = any

    class FieldCondition:
        def __init__(self, *, key, match):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, *, must):
            self.must = must


class FakeQdrantClient:
    def __init__(self, *, size=1024, distance="cosine", exists=False, fail_upsert=False):
        self.exists = exists
        self.size = size
        self.distance = distance
        self.fail_upsert = fail_upsert
        self.points = {}
        self.created = []
        self.payload_indexes = []
        self.upsert_calls = 0
        self.search_kwargs = None

    def get_collection(self, _name):
        if not self.exists:
            raise NotFound("collection not found")
        vector = SimpleNamespace(size=self.size, distance=self.distance)
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vector)))

    def create_collection(self, *, collection_name, vectors_config):
        self.exists = True
        self.size = vectors_config.size
        self.distance = vectors_config.distance
        self.created.append((collection_name, vectors_config.size, vectors_config.distance))

    def create_payload_index(self, **kwargs):
        self.payload_indexes.append(kwargs)

    def upsert(self, *, collection_name, points, wait):
        self.upsert_calls += 1
        if self.fail_upsert:
            raise RuntimeError("offline")
        assert wait is True
        for point in points:
            self.points[str(point.id)] = point

    def delete(self, *, collection_name, points_selector, wait):
        assert wait is True
        for point_id in points_selector.points:
            self.points.pop(str(point_id), None)

    def scroll(self, *, collection_name, scroll_filter, **kwargs):
        document_id = scroll_filter.must[0].match.value
        return ([point for point in self.points.values() if point.payload["document_id"] == document_id], None)

    def retrieve(self, *, collection_name, ids, **kwargs):
        return [self.points[point_id] for point_id in ids if point_id in self.points]

    def query_points(self, **kwargs):
        self.search_kwargs = kwargs
        return SimpleNamespace(points=[])


class FakeEmbeddingProvider:
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    dimension = 1024

    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("out of memory")
        return [[1.0] * self.dimension for _ in texts]

    def embed_queries(self, queries):
        return self.embed_documents(queries)

    def embed_texts(self, texts):
        return self.embed_documents(texts)


def _settings() -> Settings:
    return Settings(
        database_url="sqlite:///test.db",
        jwt_secret_key="test",
        rag_embedding_model_id=FakeEmbeddingProvider.model_id,
        rag_embedding_dimension=1024,
        rag_embedding_batch_size=2,
        qdrant_collection_name="knowledge-test",
    )


def _db(tmp_path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'qdrant-indexing.db'}")
    Base.metadata.create_all(engine)
    return Session(engine)


def _persist(db, tmp_path, chunks):
    document = KnowledgeDocument(
        title="Guide",
        original_filename="guide.pdf",
        stored_filename="guide.pdf",
        file_path=str(tmp_path / "private.pdf"),
        mime_type="application/pdf",
        file_size=1,
        uploaded_by=1,
        language="ar",
        cefr_level="A1",
    )
    db.add(document)
    db.commit()
    KnowledgeIngestionService()._replace_persisted_chunks(db, document, chunks, quality_status="complete")
    db.commit()
    return document


def _chunk(index, text, metadata=None):
    return DocumentChunk(
        id=f"chunk:{index}",
        document_id=1,
        chunk_index=index,
        text_original=text,
        text_for_embedding=text,
        page_start=index + 1,
        page_end=index + 1,
        section=None,
        headings=[],
        content_type="text",
        metadata=metadata or {},
        token_count=3,
    )


def _indexer(client, provider=None):
    settings = _settings()
    return KnowledgeEmbeddingIndexer(
        embedding_service=EmbeddingService(
            model_id=FakeEmbeddingProvider.model_id,
            config_version="test",
            dimension=1024,
            batch_size=2,
        ),
        provider=provider or FakeEmbeddingProvider(),
        qdrant=QdrantService(settings=settings, client=client, models_module=FakeModels),
    )


def test_collection_is_created_with_cosine_1024_and_useful_payload_indexes():
    client = FakeQdrantClient()
    service = QdrantService(settings=_settings(), client=client, models_module=FakeModels)
    service.ensure_collection()
    assert client.created == [("knowledge-test", 1024, "cosine")]
    assert {entry["field_name"] for entry in client.payload_indexes} == {
        "document_id", "language", "cefr_level", "content_type", "structural_quality", "requires_vision"
    }


def test_incompatible_collection_fails_without_recreation():
    client = FakeQdrantClient(size=8, exists=True)
    service = QdrantService(settings=_settings(), client=client, models_module=FakeModels)
    with pytest.raises(QdrantCollectionCompatibilityError):
        service.ensure_collection()
    assert client.created == []


def test_qdrant_search_builds_supported_document_and_metadata_filters():
    client = FakeQdrantClient(exists=True)
    service = QdrantService(settings=_settings(), client=client, models_module=FakeModels)
    assert service.search_points(
        [0.1] * 1024,
        top_k=3,
        document_ids=[3, 15],
        language="ar",
        cefr_level="A1",
        content_type="text",
        requires_vision=False,
    ) == []
    conditions = client.search_kwargs["query_filter"].must
    assert client.search_kwargs["limit"] == 3
    assert isinstance(conditions[0].match, FakeModels.MatchAny)
    assert conditions[0].match.any == [3, 15]
    assert [(item.key, item.match.value) for item in conditions[1:]] == [
        ("language", "ar"), ("cefr_level", "A1"), ("content_type", "text"), ("requires_vision", False)
    ]


def test_successful_indexing_marks_after_upsert_and_preserves_safe_payload(tmp_path):
    db = _db(tmp_path)
    document = _persist(
        db,
        tmp_path,
        [_chunk(0, "نص مفيد", {"requires_vision": True}), _chunk(1, "layout", {"structural_quality": "layout_unreliable"})],
    )
    client = FakeQdrantClient()
    report = _indexer(client).index(db, document_ids=[document.id])
    row = db.scalar(select(KnowledgeChunk).where(KnowledgeChunk.chunk_index == 0))
    excluded = db.scalar(select(KnowledgeChunk).where(KnowledgeChunk.chunk_index == 1))
    point = next(iter(client.points.values()))
    assert report.points_upserted == report.chunks_marked_indexed == 1
    assert row.embedding_status == KnowledgeChunkEmbeddingStatus.indexed
    assert excluded.embedding_status == KnowledgeChunkEmbeddingStatus.pending
    assert point.payload["requires_vision"] is True
    assert "file_path" not in point.payload and "content" not in point.payload


def test_qdrant_failure_never_marks_chunks_indexed(tmp_path):
    db = _db(tmp_path)
    document = _persist(db, tmp_path, [_chunk(0, "texte")])
    report = _indexer(FakeQdrantClient(fail_upsert=True)).index(db, document_ids=[document.id])
    status = db.scalar(select(KnowledgeChunk.embedding_status))
    assert report.chunks_failed == 1
    assert status == KnowledgeChunkEmbeddingStatus.failed


def test_unchanged_skips_while_changed_and_force_reindex_use_deterministic_point_id(tmp_path):
    db = _db(tmp_path)
    document = _persist(db, tmp_path, [_chunk(0, "premier texte")])
    client = FakeQdrantClient()
    indexer = _indexer(client)
    indexer.index(db, document_ids=[document.id])
    first_id = next(iter(client.points))
    skipped = indexer.index(db, document_ids=[document.id])
    assert skipped.points_upserted == 0
    row = db.scalar(select(KnowledgeChunk))
    row.content_for_embedding = "texte modifié"
    db.commit()
    changed = indexer.index(db, document_ids=[document.id])
    assert changed.points_upserted == 1
    forced = indexer.index(db, document_ids=[document.id], force=True)
    assert forced.points_upserted == 1
    assert list(client.points) == [first_id]


def test_document_reconciliation_deletes_only_stale_points(tmp_path):
    db = _db(tmp_path)
    document = _persist(db, tmp_path, [_chunk(0, "texte")])
    client = FakeQdrantClient()
    indexer = _indexer(client)
    indexer.index(db, document_ids=[document.id])
    client.points["stale-point"] = SimpleNamespace(id="stale-point", payload={"document_id": document.id})
    assert indexer.reconcile_document(db, document.id) == 1
    assert "stale-point" not in client.points
