from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.base import Base
from app.core.config import Settings
from app.models.knowledge_document import KnowledgeChunk, KnowledgeChunkEmbeddingStatus, KnowledgeDocument
from app.services.knowledge_sparse_indexer import KnowledgeSparseIndexer
from app.services.qdrant_service import QdrantCollectionCompatibilityError, QdrantService
from app.services.sparse_embedding_service import MultilingualSparseEncoder


class _SchemaModels:
    class Modifier:
        IDF = "idf"

    class SparseVectorConfig:
        def __init__(self, *, modifier): self.modifier = modifier

    class SparseVectorNameConfig:
        def __init__(self, *, sparse): self.sparse = sparse


class _SchemaClient:
    def __init__(self, *, version="1.19.0", modifier=None, confirm=True):
        self.version, self.confirm = version, confirm
        self.sparse = {} if modifier is None else {"lexical_sparse": SimpleNamespace(modifier=modifier)}
        self.create_calls = []
        self.dense = SimpleNamespace(size=1024, distance="cosine")

    def get_collection(self, _name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=self.dense, sparse_vectors=self.sparse)))

    def info(self):
        return SimpleNamespace(version=self.version)

    def create_vector_name(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.confirm:
            self.sparse[kwargs["vector_name"]] = SimpleNamespace(modifier=kwargs["vector_name_config"].sparse.modifier)


def _schema_service(client):
    return QdrantService(
        settings=Settings(database_url="sqlite:///test.db", jwt_secret_key="test", qdrant_collection_name="knowledge-test"),
        client=client, models_module=_SchemaModels,
    )


class _SparseQdrant:
    collection_name = "knowledge-test"
    sparse_vector_name = "lexical_sparse"
    dimension = 1024

    def __init__(self):
        self.configured = False
        self.present: set[str] = set()
        self.updated: list[dict] = []
        self.validate_calls = 0

    def validate_collection(self):
        self.validate_calls += 1

    def sparse_vector_configured(self):
        return self.configured

    def server_version(self):
        return "1.19.0"

    def point_count(self):
        return len(self.present)

    def ensure_sparse_vector(self):
        self.configured = True

    def sparse_vector_present(self, point_id):
        return point_id in self.present

    def sparse_point_state(self, point_id):
        return "sparse_present" if point_id in self.present else "sparse_missing"

    def collection_point_ids(self):
        return set(self.present)

    def update_sparse_vectors(self, points):
        self.updated.extend(points)
        self.present.update(point["id"] for point in points)


def _db(tmp_path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'sparse.db'}")
    Base.metadata.create_all(engine)
    return Session(engine)


def _chunk(db: Session, *, content="نشاط حول أفراد الأسرة", point_id="point-1") -> KnowledgeChunk:
    document = KnowledgeDocument(
        title="Guide", original_filename="guide.pdf", stored_filename="guide.pdf", file_path="private.pdf",
        mime_type="application/pdf", file_size=1, uploaded_by=1,
    )
    db.add(document)
    db.flush()
    chunk = KnowledgeChunk(
        document_id=document.id, chunk_index=0, content=content, content_for_embedding=content, token_count=4,
        content_type="text", quality_status="accepted", heading_context=["La famille"], chunk_metadata={},
        chunk_hash="a" * 64, ingestion_version="test", embedding_status=KnowledgeChunkEmbeddingStatus.indexed,
        vector_point_id=point_id,
    )
    db.add(chunk)
    db.commit()
    return chunk


def test_multilingual_sparse_encoding_preserves_arabic_and_french_lexical_signals():
    encoder = MultilingualSparseEncoder()
    family = encoder.encode("نشاط حول أفراد الأسرة")
    environment = encoder.encode("نشاط حول البيئة")
    hotel = encoder.encode("dialogue \\u00e0 l'h\\u00f4tel")
    food = encoder.encode("exercice alimentation")
    query_family = encoder.encode("الأسرة")
    query_hotel = encoder.encode("الفندق h\\u00f4tel")
    query_food = encoder.encode("exercice alimentation")
    assert family and environment and hotel and food and query_family and query_hotel and query_food
    assert set(family.indices) & set(query_family.indices)
    assert not (set(environment.indices) & set(query_family.indices))
    assert set(hotel.indices) & set(query_hotel.indices)
    assert set(food.indices) == set(query_food.indices)


def test_sparse_encoding_is_stable_and_empty_safe_with_mixed_unicode():
    encoder = MultilingualSparseEncoder()
    text = encoder.indexed_text(heading_context=["Activité", "الأسرة"], content="Dialogue famille / عائلة")
    assert encoder.encode(text) == encoder.encode(text)
    assert encoder.encode("  \n\t") is None
    assert "الأسرة" in text and "famille" in text


def test_sparse_indexer_preflight_and_repeat_indexing_preserve_point_ids_and_dense_state(tmp_path):
    db = _db(tmp_path)
    chunk = _chunk(db)
    qdrant = _SparseQdrant()
    indexer = KnowledgeSparseIndexer(qdrant=qdrant)
    dry = indexer.index(db, document_ids=[chunk.document_id], dry_run=True)
    assert dry.chunks_scanned == dry.chunks_eligible == 1
    assert dry.sparse_schema_action == "would_create" and dry.sparse_configured_after is False
    assert dry.points_updated == 0 and qdrant.updated == [] and qdrant.configured is False
    first = indexer.index(db, document_ids=[chunk.document_id])
    assert first.sparse_configured_after is True and first.points_updated == 1 and qdrant.updated[0]["id"] == "point-1"
    assert chunk.embedding_status == KnowledgeChunkEmbeddingStatus.indexed
    second = indexer.index(db, document_ids=[chunk.document_id])
    assert second.points_updated == 0 and second.skipped >= 1


def test_preflight_mismatch_is_safe(tmp_path):
    db = _db(tmp_path)
    chunk = _chunk(db)
    qdrant = _SparseQdrant()
    def fail():
        raise QdrantCollectionCompatibilityError("dense schema mismatch")
    qdrant.validate_collection = fail
    with pytest.raises(QdrantCollectionCompatibilityError):
        KnowledgeSparseIndexer(qdrant=qdrant).index(db, document_ids=[chunk.document_id], dry_run=True)
    assert qdrant.updated == [] and qdrant.configured is False


def test_qdrant_sparse_configuration_is_explicit_not_part_of_dense_ensure():
    # The existing dense ensure path has no sparse migration call; preserve that
    # contract structurally without requiring a real Qdrant server.
    assert "ensure_sparse_vector" not in QdrantService.ensure_collection.__code__.co_names


def test_supported_server_creates_named_sparse_schema_and_preserves_unnamed_dense():
    client = _SchemaClient()
    service = _schema_service(client)
    service.ensure_sparse_vector()
    assert len(client.create_calls) == 1
    call = client.create_calls[0]
    assert call["vector_name"] == "lexical_sparse" and call["wait"] is True
    assert call["vector_name_config"].sparse.modifier == "idf"
    assert client.dense.size == 1024 and client.dense.distance == "cosine"
    assert service.sparse_vector_configured() is True


def test_schema_call_without_confirmed_refresh_blocks_sparse_indexing():
    client = _SchemaClient(confirm=False)
    with pytest.raises(QdrantCollectionCompatibilityError, match="did not retain"):
        _schema_service(client).ensure_sparse_vector()
    assert len(client.create_calls) == 1


def test_existing_incompatible_or_unsupported_schema_fails_without_creation():
    incompatible = _SchemaClient(modifier="none")
    with pytest.raises(QdrantCollectionCompatibilityError, match="incompatible"):
        _schema_service(incompatible).ensure_sparse_vector()
    assert incompatible.create_calls == []
    unsupported = _SchemaClient(version="1.17.9")
    with pytest.raises(QdrantCollectionCompatibilityError, match="does not support"):
        _schema_service(unsupported).ensure_sparse_vector()
    assert unsupported.create_calls == []


def test_schema_failure_never_allows_point_write(tmp_path):
    db = _db(tmp_path)
    chunk = _chunk(db)
    qdrant = _SparseQdrant()
    def fail_schema():
        raise QdrantCollectionCompatibilityError("schema failed")
    qdrant.ensure_sparse_vector = fail_schema
    with pytest.raises(QdrantCollectionCompatibilityError):
        KnowledgeSparseIndexer(qdrant=qdrant).index(db, document_ids=[chunk.document_id])
    assert qdrant.updated == []


def test_missing_point_and_update_rejection_are_safely_reported(tmp_path):
    db = _db(tmp_path)
    missing = _chunk(db, point_id="missing")
    rejected = _chunk(db, point_id="rejected")
    qdrant = _SparseQdrant()
    def point_state(point_id):
        return "missing" if point_id == "missing" else "sparse_missing"
    def reject(_points):
        raise RuntimeError("update rejected")
    qdrant.sparse_point_state = point_state
    qdrant.update_sparse_vectors = reject
    report = KnowledgeSparseIndexer(qdrant=qdrant).index(db, document_ids=[missing.document_id, rejected.document_id])
    assert report.failed == 2
    by_id = {item["knowledge_chunk_id"]: item for item in report.failures}
    assert by_id[missing.id]["failure_category"] == "missing_qdrant_point"
    assert by_id[missing.id]["qdrant_point_state"] == "missing"
    assert by_id[rejected.id]["failure_category"] == "qdrant_vector_update_rejected"
    assert by_id[rejected.id]["sparse_encoding_succeeded"] is True
    assert "content" not in by_id[missing.id]


def test_coverage_audit_is_read_only_and_explains_extra_noneligible_points(tmp_path):
    db = _db(tmp_path)
    chunk = _chunk(db, point_id="eligible")
    qdrant = _SparseQdrant()
    qdrant.present = {"eligible", "stale"}
    audit = KnowledgeSparseIndexer(qdrant=qdrant).audit_coverage(db, document_ids=[chunk.document_id])
    assert audit.eligible_chunks == audit.eligible_with_sparse == 1
    assert audit.eligible_missing_sparse == audit.eligible_missing_qdrant_point == 0
    assert audit.stale_qdrant_points == 1
    assert audit.missing_sparse_chunks == []
    assert qdrant.updated == []


def test_verifier_uses_the_same_empty_sparse_eligibility_rule_as_indexing(tmp_path):
    db = _db(tmp_path)
    eligible = _chunk(db, point_id="eligible")
    noneligible = _chunk(db, content="dense text", point_id="noneligible")
    noneligible.content = "   "  # Canonical sparse text is empty while dense eligibility remains true.
    db.commit()
    qdrant = _SparseQdrant()
    qdrant.present = {"eligible", "noneligible"}
    indexer = KnowledgeSparseIndexer(qdrant=qdrant)
    preflight = indexer.preflight(db, document_ids=[eligible.document_id, noneligible.document_id])
    audit = indexer.audit_coverage(db, document_ids=[eligible.document_id, noneligible.document_id])
    assert preflight.chunks_eligible == audit.eligible_chunks == 1
    assert audit.eligible_missing_sparse == 0
    assert audit.canonical_noneligible_points == 1
    assert audit.canonical_noneligible_chunks[0]["knowledge_chunk_id"] == noneligible.id
    assert audit.canonical_noneligible_chunks[0]["eligibility_reason"] == "empty_sparse_representation"
