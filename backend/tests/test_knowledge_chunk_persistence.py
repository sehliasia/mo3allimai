from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.routes.admin import knowledge_segments
from app.database.base import Base
from app.models.knowledge_document import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeProcessingJob,
    KnowledgeProcessingJobStatus,
    KnowledgeProcessingJobType,
)
from app.services.document_chunk import DocumentChunk
from app.services.knowledge_ingestion_service import KnowledgeIngestionService
from app.services.legacy_knowledge_chunk_materializer import (
    LegacyKnowledgeChunkMaterializer,
    LegacyMaterializationError,
)


def _session(tmp_path: Path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'chunks.db'}")
    Base.metadata.create_all(engine)
    return Session(engine)


def _document(tmp_path: Path) -> KnowledgeDocument:
    document = KnowledgeDocument(
        title="Guide mixte", original_filename="guide.pdf", stored_filename="guide.pdf",
        file_path=str(tmp_path / "guide.pdf"), mime_type="application/pdf", file_size=12,
        uploaded_by=1,
    )
    return document


def _chunk(index: int, content: str, *, content_type: str = "text") -> DocumentChunk:
    return DocumentChunk(
        id=f"knowledge-document:1:chunk:{index}", document_id=1, chunk_index=index,
        text_original=content,
        text_for_embedding=f"Document: Guide mixte\n\n{content}",
        page_start=index + 1, page_end=index + 1, section="Unité",
        headings=["Unité"], content_type=content_type,
        metadata={"extraction_mode": "native", "image_ids": [], "source": "test", "path": "private-path-must-not-persist"},
        token_count=8,
    )


def test_validated_chunks_are_persisted_unicode_safe_and_replace_prior_generation(tmp_path):
    db = _session(tmp_path)
    document = _document(tmp_path)
    db.add(document)
    db.commit()
    service = KnowledgeIngestionService()

    service._replace_persisted_chunks(db, document, [_chunk(0, "نص عربي مع Français"), _chunk(1, "| A1 | Bonjour |", content_type="table")], quality_status="complete")
    db.commit()
    stored = db.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.chunk_index)).all()
    assert [chunk.content for chunk in stored] == ["نص عربي مع Français", "| A1 | Bonjour |"]
    assert [(chunk.source_page_start, chunk.source_page_end, chunk.token_count) for chunk in stored] == [(1, 1, 8), (2, 2, 8)]
    assert stored[0].chunk_metadata == {"extraction_mode": "native", "image_ids": [], "source": "test"}

    service._replace_persisted_chunks(db, document, [_chunk(0, "Nouvelle génération عربية")], quality_status="partial")
    db.commit()
    stored = db.scalars(select(KnowledgeChunk)).all()
    assert len(stored) == 1
    assert stored[0].content == "Nouvelle génération عربية"
    assert stored[0].quality_status == "partial"


def test_chunk_hash_is_deterministic_for_semantically_identical_content():
    first = _chunk(0, "نص   عربي\nFrançais")
    second = _chunk(0, "نص عربي Français")
    assert KnowledgeIngestionService._chunk_hash(first, "v1") == KnowledgeIngestionService._chunk_hash(second, "v1")
    assert KnowledgeIngestionService._chunk_hash(first, "v1") != KnowledgeIngestionService._chunk_hash(first, "v2")


def test_failed_replacement_keeps_previous_generation(tmp_path):
    db = _session(tmp_path)
    document = _document(tmp_path)
    db.add(document)
    db.commit()
    service = KnowledgeIngestionService()
    service._replace_persisted_chunks(db, document, [_chunk(0, "Version fiable")], quality_status="complete")
    db.commit()

    with pytest.raises(IntegrityError):
        service._replace_persisted_chunks(db, document, [_chunk(0, "un"), _chunk(0, "deux")], quality_status="complete")
    db.rollback()
    assert db.scalar(select(KnowledgeChunk.content).where(KnowledgeChunk.document_id == document.id)) == "Version fiable"


def test_segments_endpoint_reads_persisted_rows_without_debug_artifact(tmp_path):
    db = _session(tmp_path)
    document = _document(tmp_path)
    db.add(document)
    db.commit()
    service = KnowledgeIngestionService()
    service._replace_persisted_chunks(db, document, [_chunk(0, "محتوى محفوظ"), _chunk(1, "Deuxième segment", content_type="table")], quality_status="complete")
    db.commit()

    response = knowledge_segments(document.id, page=1, page_size=20, source_page=2, content_type=None, _=None, db=db)
    assert response["availability"] == "available"
    assert response["total"] == 1
    assert response["items"][0]["content"] == "Deuxième segment"


def test_legacy_materialization_is_noop_when_rows_are_already_persisted(tmp_path):
    db = _session(tmp_path)
    document = _document(tmp_path)
    db.add(document); db.commit()
    KnowledgeIngestionService()._replace_persisted_chunks(db, document, [_chunk(0, "Déjà enregistré")], quality_status="complete")
    db.commit()

    report = LegacyKnowledgeChunkMaterializer().materialize(db, document_id=document.id)
    assert report.source_used == "already_persisted"
    assert report.chunks_persisted == 0


def test_explicit_cache_rebuild_replaces_existing_chunks_without_live_parser(tmp_path):
    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()
    service = KnowledgeIngestionService()
    service._replace_persisted_chunks(db, document, [_chunk(0, "Ancienne génération")], quality_status="complete")
    db.commit()

    class Parser:
        rebuild_calls = 0
        def parse_pdf(self, *_args, **_kwargs): raise AssertionError("live parsing must never run")
        def load_cached_extraction_only(self, *_args, **_kwargs): raise AssertionError("source PDF must not be read")
        def load_cached_extraction_for_rebuild(self, **_kwargs):
            self.rebuild_calls += 1
            return type("Cached", (), {"document": object(), "page_extractions": [], "page_issues": []})()

    class Chunker:
        last_corruptions = []
        def chunk(self, **_kwargs):
            chunk = _chunk(0, "العربية reconstruite")
            return [chunk.__class__(**{**chunk.__dict__, "metadata": {**chunk.metadata, "has_image": True, "structural_quality": "structured"}})]

    parser = Parser()
    report = LegacyKnowledgeChunkMaterializer(parser=parser, chunker_factory=Chunker).materialize(
        db, document_id=document.id, rebuild_from_cache=True
    )
    db.commit()
    stored = db.scalar(select(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id))
    assert parser.rebuild_calls == 1
    assert report.source_used == "extraction_cache_rebuild"
    assert report.old_persisted_chunks == report.chunks_discovered == 1
    assert report.new_persisted_chunks == report.chunks_persisted == 1
    assert stored.content == "العربية reconstruite"
    assert stored.chunk_metadata["has_image"] is True
    assert stored.chunk_metadata["structural_quality"] == "structured"
    assert "<!-- image" not in stored.content_for_embedding.lower()


def test_failed_explicit_cache_rebuild_keeps_existing_chunks(tmp_path):
    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()
    KnowledgeIngestionService()._replace_persisted_chunks(db, document, [_chunk(0, "Version fiable")], quality_status="complete")
    db.commit()

    class Parser:
        def load_cached_extraction_for_rebuild(self, **_kwargs):
            return type("Cached", (), {"document": object(), "page_extractions": [], "page_issues": []})()

    class Chunker:
        last_corruptions = []
        def chunk(self, **_kwargs): return [_chunk(1, "Index invalide")]

    with pytest.raises(LegacyMaterializationError, match="sequential"):
        LegacyKnowledgeChunkMaterializer(parser=Parser(), chunker_factory=Chunker).materialize(
            db, document_id=document.id, rebuild_from_cache=True
        )
    assert db.scalar(select(KnowledgeChunk.content).where(KnowledgeChunk.document_id == document.id)) == "Version fiable"


def test_legacy_debug_materialization_is_unicode_safe_and_segments_use_database(tmp_path, monkeypatch):
    import app.services.legacy_knowledge_chunk_materializer as materializer_module

    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()
    artifact = tmp_path / f"{document.id}.json"
    artifact.write_text('{"document_id": %s, "chunks": [{"chunk_index": 0, "text_original": "العربية et Français", "text_for_embedding": "Document: Guide\\n\\nالعربية et Français", "page_start": 2, "page_end": 2, "headings": ["Unité"], "content_type": "text", "metadata": {"extraction_mode": "native"}, "token_count": 8}]}' % document.id, encoding="utf-8")
    monkeypatch.setattr(materializer_module, "DEBUG_CHUNKS_DIRECTORY", tmp_path)

    report = LegacyKnowledgeChunkMaterializer().materialize(db, document_id=document.id)
    db.commit()
    assert report.source_used == "legacy_debug" and report.ocr_invoked is False and report.docling_invoked is False
    assert db.scalar(select(KnowledgeChunk.content)) == "العربية et Français"
    assert knowledge_segments(document.id, 1, 20, None, None, None, db)["availability"] == "available"


def test_invalid_legacy_debug_artifact_rolls_back_without_rows(tmp_path, monkeypatch):
    import app.services.legacy_knowledge_chunk_materializer as materializer_module

    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()
    (tmp_path / f"{document.id}.json").write_text('{"document_id": %s, "chunks": [{"chunk_index": 1}]}' % document.id, encoding="utf-8")
    monkeypatch.setattr(materializer_module, "DEBUG_CHUNKS_DIRECTORY", tmp_path)
    with pytest.raises(LegacyMaterializationError):
        LegacyKnowledgeChunkMaterializer().materialize(db, document_id=document.id)
    assert db.scalar(select(KnowledgeChunk.id)) is None


def test_cache_only_materialization_never_calls_live_parser(tmp_path):
    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()

    class Parser:
        def parse_pdf(self, *_args, **_kwargs): raise AssertionError("live parser must never run")
        def load_cached_extraction_only(self, *_args, **_kwargs):
            return type("Cached", (), {"document": object(), "page_extractions": [], "page_issues": []})()

    class Chunker:
        last_corruptions: list[object] = []
        def chunk(self, **_kwargs): return [_chunk(0, "نص من cache Français")]

    report = LegacyKnowledgeChunkMaterializer(parser=Parser(), chunker_factory=Chunker).materialize(db, document_id=document.id)
    db.commit()
    assert report.source_used == "extraction_cache"
    assert report.ocr_invoked is False and report.docling_invoked is False
    assert db.scalar(select(KnowledgeChunk.content)) == "نص من cache Français"


def _completed_ingestion_summary(db: Session, document: KnowledgeDocument, *, valid: int, rejected: int) -> None:
    db.add(KnowledgeProcessingJob(
        document_id=document.id,
        job_type=KnowledgeProcessingJobType.ingestion,
        status=KnowledgeProcessingJobStatus.completed,
        stage="completed",
        result_summary={"chunks_valid": valid, "chunks_quarantined_count": rejected},
    ))
    db.commit()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<!-- image -->", True),
        ("<!-- image -->\n<!-- image -->", True),
        ("<!-- image -->\n3\n<!-- image -->\n4", True),
        ("<!-- image -->\n...", True),
        ("<!-- image -->\nمرحبا", False),
        ("<!-- image -->\nTexte réel", False),
        ("Kw A,«aw «.3 34w .3, 3w :33Kw...", False),
    ],
)
def test_image_placeholder_rejection_classifier_is_unicode_safe(text, expected):
    rejection = {"text_preview": repr(text)}
    assert LegacyKnowledgeChunkMaterializer._is_image_placeholder_only(rejection) is expected


def test_cache_materialization_persists_valid_chunks_and_discards_image_placeholder_rejections(tmp_path):
    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()
    _completed_ingestion_summary(db, document, valid=1, rejected=4)

    class Parser:
        def parse_pdf(self, *_args, **_kwargs): raise AssertionError("live parser must never run")
        def load_cached_extraction_only(self, *_args, **_kwargs):
            return type("Cached", (), {"document": object(), "page_extractions": [], "page_issues": []})()

    class Chunker:
        last_corruptions = [
            {"page": 7, "text_preview": repr("<!-- image -->\n<!-- image -->\n<!-- image -->\n<!-- image -->")},
            {"page": 23, "text_preview": repr("<!-- image -->\n<!-- image -->\n<!-- image -->\n<!-- image -->")},
            {"page": 28, "text_preview": repr("<!-- image -->\n<!-- image -->\n<!-- image -->\n3\n<!-- image -->\n4")},
            {"page": 43, "text_preview": repr("<!-- image -->\n<!-- image -->\n<!-- image -->\n<!-- image -->\n<!-- image -->")},
        ]
        def chunk(self, **_kwargs): return [_chunk(0, "نص عربي صالح avec Français")]

    report = LegacyKnowledgeChunkMaterializer(parser=Parser(), chunker_factory=Chunker).materialize(db, document_id=document.id)
    db.commit()
    stored = db.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.chunk_index)).all()
    assert report.chunks_discovered == 5
    assert report.chunks_validated == report.chunks_persisted == 1
    assert report.chunks_rejected == 4
    assert [chunk.content for chunk in stored] == ["نص عربي صالح avec Français"]
    assert "<!-- image" not in stored[0].content_for_embedding.lower()
    assert knowledge_segments(document.id, 1, 20, None, None, None, db)["availability"] == "available"


def test_cache_materialization_keeps_real_corruption_rejected_when_previous_summary_matches(tmp_path):
    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()
    _completed_ingestion_summary(db, document, valid=1, rejected=1)

    class Parser:
        def load_cached_extraction_only(self, *_args, **_kwargs):
            return type("Cached", (), {"document": object(), "page_extractions": [], "page_issues": [], "pages_count": 10})()

    class Chunker:
        last_corruptions = [{"page": 8, "text_preview": repr("Kw A,«aw «.3 34w .3, 3w :33Kw...") }]
        def chunk(self, **_kwargs): return [_chunk(0, "نص عربي صالح")]

    report = LegacyKnowledgeChunkMaterializer(parser=Parser(), chunker_factory=Chunker).materialize(db, document_id=document.id)
    assert report.chunks_rejected == 1
    assert report.rejected_pages == [8]
    assert report.compatibility_source == "previous_ingestion_summary"
    assert db.scalar(select(KnowledgeChunk.content)) == "نص عربي صالح"


def test_cache_materialization_fails_when_rejections_exceed_existing_partial_policy(tmp_path):
    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()
    _completed_ingestion_summary(db, document, valid=1, rejected=1)

    class Parser:
        def load_cached_extraction_only(self, *_args, **_kwargs):
            return type("Cached", (), {"document": object(), "page_extractions": [], "page_issues": [], "pages_count": 10})()

    class Chunker:
        last_corruptions = [
            {"page": 7, "text_preview": repr("<!-- image -->")},
            {"page": 23, "text_preview": repr("<!-- image --> 4")},
            {"page": 24, "text_preview": repr("<!-- image --> 5")},
        ]
        def chunk(self, **_kwargs): return [_chunk(0, "نص عربي صالح")]

    with pytest.raises(LegacyMaterializationError, match="exceed"):
        LegacyKnowledgeChunkMaterializer(parser=Parser(), chunker_factory=Chunker).materialize(db, document_id=document.id)
    assert db.scalar(select(KnowledgeChunk.id)) is None


def test_cache_materialization_uses_existing_partial_threshold_when_summary_is_unavailable(tmp_path):
    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()

    class Parser:
        def load_cached_extraction_only(self, *_args, **_kwargs):
            return type("Cached", (), {"document": object(), "page_extractions": [], "page_issues": [], "pages_count": 10})()

    class Chunker:
        last_corruptions = [{"page": 3, "failure_reasons": ["glyph_noise"], "text_preview": repr("kasratan lamisolated") }]
        def chunk(self, **_kwargs): return [_chunk(0, "Français valide")]

    report = LegacyKnowledgeChunkMaterializer(parser=Parser(), chunker_factory=Chunker).materialize(db, document_id=document.id)
    assert report.partial_materialization is True
    assert report.compatibility_source == "existing_partial_threshold"
    assert db.scalar(select(KnowledgeChunk.content)) == "Français valide"


def test_missing_cache_fails_without_live_processing(tmp_path):
    document = _document(tmp_path)
    db = _session(tmp_path); db.add(document); db.commit()

    class Parser:
        def parse_pdf(self, *_args, **_kwargs): raise AssertionError("live parser must never run")
        def load_cached_extraction_only(self, *_args, **_kwargs): return None

    with pytest.raises(LegacyMaterializationError, match="cache-only materialization"):
        LegacyKnowledgeChunkMaterializer(parser=Parser()).materialize(db, document_id=document.id)
