from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.base import Base
from app.models.knowledge_document import KnowledgeChunk, KnowledgeChunkEmbeddingStatus, KnowledgeDocument
from app.services.embedding_service import (
    DeterministicFakeEmbeddingProvider,
    EmbeddingService,
    build_qdrant_payload,
    embedding_input_hash,
    is_chunk_embedding_eligible,
    vector_point_id,
)
from app.services.knowledge_ingestion_service import KnowledgeIngestionService
from app.services.document_chunk import DocumentChunk


def _db(tmp_path: Path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'embedding.db'}")
    Base.metadata.create_all(engine)
    return Session(engine)


def _document(tmp_path: Path) -> KnowledgeDocument:
    return KnowledgeDocument(title="Guide", original_filename="guide.pdf", stored_filename="guide.pdf", file_path=str(tmp_path / "guide.pdf"), mime_type="application/pdf", file_size=1, uploaded_by=1, language="ar", cefr_level="A1")


def _chunk(index: int, text: str, metadata=None) -> DocumentChunk:
    return DocumentChunk(
        id=f"x:{index}", document_id=1, chunk_index=index,
        text_original=text, text_for_embedding=text, page_start=1, page_end=1,
        section=None, headings=[], content_type="text", metadata=metadata or {}, token_count=4,
    )


def _persist(db, document, chunks):
    db.add(document); db.commit()
    KnowledgeIngestionService()._replace_persisted_chunks(db, document, chunks, quality_status="partial")
    db.commit()


def test_embedding_eligibility_covers_unicode_structure_images_and_markers(tmp_path):
    db = _db(tmp_path); document = _document(tmp_path)
    _persist(db, document, [_chunk(0, "Bonjour"), _chunk(1, "العربية et Français"), _chunk(2, "texte", {"structural_quality": "layout_unreliable"}), _chunk(3, "texte", {"requires_vision": True})])
    rows = db.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.chunk_index)).all()
    assert is_chunk_embedding_eligible(rows[0]).eligible
    assert is_chunk_embedding_eligible(rows[1]).eligible
    assert not is_chunk_embedding_eligible(rows[2]).eligible
    assert is_chunk_embedding_eligible(rows[3]).eligible
    rows[0].content_for_embedding = "<!-- image -->"
    assert not is_chunk_embedding_eligible(rows[0]).eligible


def test_embedding_hash_fake_vectors_and_point_ids_are_deterministic(tmp_path):
    first = embedding_input_hash("نص عربي", model_id="model-a", config_version="v1")
    assert first == embedding_input_hash("نص  عربي", model_id="model-a", config_version="v1")
    assert first != embedding_input_hash("نص مختلف", model_id="model-a", config_version="v1")
    assert first != embedding_input_hash("نص عربي", model_id="model-b", config_version="v1")
    provider = DeterministicFakeEmbeddingProvider()
    assert provider.embed_texts(["العربية"])[0] == provider.embed_texts(["العربية"])[0]
    db = _db(tmp_path); document = _document(tmp_path)
    _persist(db, document, [_chunk(0, "duplicate"), _chunk(1, "duplicate")])
    rows = db.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.chunk_index)).all()
    assert vector_point_id(rows[0]) != vector_point_id(rows[1])


def test_embedding_preparation_batches_skips_indexed_and_force_reprocesses(tmp_path):
    db = _db(tmp_path); document = _document(tmp_path)
    _persist(db, document, [_chunk(index, f"texte {index}") for index in range(3)])
    service = EmbeddingService(model_id="fake-v1", config_version="test", batch_size=2)
    batches = list(service.iter_embedding_batches(db))
    assert [len(batch.candidates) for batch in batches] == [2, 1]
    candidates = [candidate for batch in batches for candidate in batch.candidates]
    EmbeddingService.mark_indexed(db, candidates); db.commit()
    skipped_batches = list(service.iter_embedding_batches(db))
    assert sum(len(batch.candidates) for batch in skipped_batches) == 0
    assert sum(batch.skipped for batch in skipped_batches) == 3
    forced_batches = list(service.iter_embedding_batches(db, force=True))
    assert sum(len(batch.candidates) for batch in forced_batches) == 3


def test_embedding_service_hash_config_changes_with_document_dimension():
    first = EmbeddingService(model_id="qwen", config_version="benchmark", dimension=1024)
    second = EmbeddingService(model_id="qwen", config_version="benchmark", dimension=512)
    assert first.config_version != second.config_version
    assert embedding_input_hash("نص", model_id=first.model_id, config_version=first.config_version) != embedding_input_hash(
        "نص", model_id=second.model_id, config_version=second.config_version
    )


def test_embedding_payload_uses_safe_document_metadata_only(tmp_path):
    db = _db(tmp_path); document = _document(tmp_path)
    _persist(db, document, [_chunk(0, "العربية", {"has_image": True, "requires_vision": True, "structural_quality": "partially_structured", "path": "C:/private"})])
    chunk = db.scalar(select(KnowledgeChunk))
    payload = build_qdrant_payload(chunk, document)
    assert payload["document_id"] == document.id and payload["requires_vision"] is True
    assert "path" not in payload and "content" not in payload
