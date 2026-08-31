"""Private Teacher Library ingestion. It never writes KnowledgeDocument/KnowledgeChunk."""
from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from time import perf_counter

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models.teacher_library import TeacherLibraryChunk, TeacherLibraryDocument
from app.services.embedding_providers import get_embedding_provider
from app.services.qdrant_service import QdrantService
from app.services.sparse_embedding_service import MultilingualSparseEncoder

logger = logging.getLogger(__name__)


class TeacherLibraryIngestionError(RuntimeError):
    """Safe ingestion error with a teacher-facing explanation.

    The chained exception remains available to application logs, but is never
    returned verbatim through the Library API.
    """

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or "L’indexation du document a échoué."


class TeacherLibraryIngestionService:
    """Synchronous V1 ingestion with separate canonical records and Qdrant payloads."""

    def __init__(self, *, qdrant: QdrantService | None = None, provider=None) -> None:
        self.qdrant = qdrant or QdrantService()
        self.provider = provider or get_embedding_provider()
        self.sparse = MultilingualSparseEncoder()

    @staticmethod
    def _extract(path: Path, mime_type: str) -> list[tuple[int | None, str]]:
        if mime_type == "text/plain":
            return [(None, path.read_text(encoding="utf-8", errors="replace"))]
        if mime_type == "application/pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise TeacherLibraryIngestionError("PDF extraction is unavailable.") from exc
            return [(index, page.extract_text() or "") for index, page in enumerate(PdfReader(str(path)).pages, start=1)]
        if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                from docx import Document
            except ImportError as exc:
                raise TeacherLibraryIngestionError("DOCX extraction is unavailable.") from exc
            return [(None, "\n".join(p.text for p in Document(str(path)).paragraphs))]
        raise TeacherLibraryIngestionError("Unsupported document type.")

    @staticmethod
    def _chunks(pages: list[tuple[int | None, str]], *, words_per_chunk: int = 360):
        index = 0
        for page, text in pages:
            words = text.split()
            for start in range(0, len(words), words_per_chunk):
                content = " ".join(words[start:start + words_per_chunk]).strip()
                if content:
                    yield index, page, content
                    index += 1

    @staticmethod
    def _point_id(chunk: TeacherLibraryChunk) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mo3allimai:teacher:{chunk.owner_id}:{chunk.document_id}:{chunk.chunk_hash}:{chunk.chunk_index}"))

    @staticmethod
    def _set_stage(db: Session, document: TeacherLibraryDocument, stage: str) -> None:
        document.status = "processing"
        document.processing_stage = stage
        document.processing_error = None
        db.commit()

    @staticmethod
    def _safe_failure_message(exc: Exception) -> str:
        if isinstance(exc, TeacherLibraryIngestionError):
            return exc.user_message
        name = type(exc).__name__.casefold()
        message = str(exc).casefold()
        if "qdrant" in name or "qdrant" in message or "connection" in name:
            return "Connexion à la base vectorielle impossible."
        if "embedding" in name or "embedding" in message or "vector dimension" in message:
            return "Erreur lors de la génération des embeddings."
        if "pdf" in name or "pypdf" in name or "extract" in message:
            return "Impossible d’extraire le texte du document."
        return "L’indexation du document a échoué. Réessayez plus tard."

    @staticmethod
    def _failure_detail(exc: Exception) -> str:
        """Keep a bounded, actionable diagnostic in the document record."""
        chain = [exc]
        while chain[-1].__cause__ is not None:
            chain.append(chain[-1].__cause__)
        detail = " caused by ".join(
            f"{type(error).__name__}: {error}" for error in chain
        )
        return detail[:1000]

    def ingest(self, db: Session, document: TeacherLibraryDocument, *, storage_root: Path) -> TeacherLibraryDocument:
        started = perf_counter()
        document_id = document.id
        owner_id = document.owner_id
        filename = document.original_filename
        self._set_stage(db, document, "extracting")
        logger.info("[INGESTION] document_id=%s START owner_id=%s filename=%s", document_id, owner_id, filename)
        try:
            path = storage_root / document.storage_key
            logger.info("[INGESTION] document_id=%s stage=file_resolution path=%s exists=%s", document_id, path, path.is_file())
            if not path.is_file():
                raise FileNotFoundError(f"Uploaded document is missing: {path}")

            extraction_started = perf_counter()
            logger.info("[INGESTION] document_id=%s stage=pdf_extraction start", document_id)
            pages = self._extract(path, document.mime_type)
            extraction_seconds = perf_counter() - extraction_started
            logger.info("[INGESTION] document_id=%s stage=pdf_extraction pages=%s chars=%s duration_seconds=%.3f", document_id, len(pages), sum(len(text) for _, text in pages), extraction_seconds)
            self._set_stage(db, document, "chunking")
            chunking_started = perf_counter()
            logger.info("[INGESTION] document_id=%s stage=chunking start", document_id)
            rows: list[TeacherLibraryChunk] = []
            for chunk_index, page, content in self._chunks(pages):
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                rows.append(TeacherLibraryChunk(document_id=document.id, owner_id=document.owner_id, chunk_index=chunk_index, content=content, content_for_embedding=content, token_count=len(content.split()), page_number=page, chunk_hash=digest, chunk_metadata={"source_type": "user_document", "page_number": page}))
            if not rows:
                raise TeacherLibraryIngestionError(
                    "Document contains no extractable text.",
                    user_message="Le document ne contient pas de texte exploitable.",
                )
            postgres_started = perf_counter()
            db.execute(delete(TeacherLibraryChunk).where(TeacherLibraryChunk.document_id == document.id))
            db.add_all(rows)
            db.flush()
            logger.info("[INGESTION] document_id=%s stage=chunking chunks=%s duration_seconds=%.3f", document_id, len(rows), perf_counter() - chunking_started)
            logger.info("[INGESTION] document_id=%s stage=postgres_chunks chunks=%s duration_seconds=%.3f", document_id, len(rows), perf_counter() - postgres_started)
            self._set_stage(db, document, "embedding")
            embedding_started = perf_counter()
            logger.info("[INGESTION] document_id=%s stage=embedding start chunks=%s batch_size=%s model=%s dimension=%s", document_id, len(rows), getattr(self.provider, "batch_size", "provider-default"), self.provider.model_id, self.provider.dimension)

            def report_embedding_progress(completed: int, total: int) -> None:
                logger.info("[INGESTION] document_id=%s stage=embedding progress=%s/%s", document_id, completed, total)

            try:
                vectors = self.provider.embed_documents(
                    [row.content_for_embedding for row in rows],
                    progress_callback=report_embedding_progress,
                )
            except TypeError as exc:
                # Test/deterministic providers predate progress_callback; real
                # production providers always support it.
                if "progress_callback" not in str(exc):
                    raise
                vectors = self.provider.embed_documents([row.content_for_embedding for row in rows])
                report_embedding_progress(len(rows), len(rows))
            logger.info("[EMBEDDING] document_id=%s chunks=%s dimension=%s duration_seconds=%.3f diagnostics=%s", document_id, len(rows), self.provider.dimension, perf_counter() - embedding_started, getattr(self.provider, "last_diagnostics", lambda: {})())
            if len(vectors) != len(rows) or any(len(vector) != self.provider.dimension for vector in vectors):
                raise TeacherLibraryIngestionError(
                    "Embedding provider returned an invalid vector dimension.",
                    user_message="Erreur lors de la génération des embeddings.",
                )
            points = []
            sparse_points = []
            for row, vector in zip(rows, vectors, strict=True):
                row.vector_point_id, row.embedding_model, row.embedding_status = self._point_id(row), self.provider.model_id, "indexed"
                payload = {"source_type": "user_document", "owner_id": row.owner_id, "document_id": row.document_id, "chunk_id": row.id, "page_number": row.page_number, "filename": document.original_filename, "content_type": "text"}
                points.append({"id": row.vector_point_id, "vector": vector, "payload": payload})
                sparse = self.sparse.encode(row.content_for_embedding)
                if sparse is not None:
                    sparse_points.append({"id": row.vector_point_id, "indices": sparse.indices, "values": sparse.values})
            self._set_stage(db, document, "indexing")
            logger.info("[INGESTION] document_id=%s stage=qdrant_dense collection=%s vectors=%s", document_id, self.qdrant.collection_name, len(points))
            qdrant_dense_started = perf_counter()
            self.qdrant.ensure_collection()
            self.qdrant.upsert_points(points)
            logger.info("[INGESTION] document_id=%s stage=qdrant_dense duration_seconds=%.3f", document_id, perf_counter() - qdrant_dense_started)
            logger.info("[INGESTION] document_id=%s stage=qdrant_sparse vectors=%s", document_id, len(sparse_points))
            qdrant_sparse_started = perf_counter()
            self.qdrant.ensure_sparse_vector()
            if sparse_points:
                self.qdrant.update_sparse_vectors(sparse_points)
            logger.info("[INGESTION] document_id=%s stage=qdrant_sparse duration_seconds=%.3f", document_id, perf_counter() - qdrant_sparse_started)
            completion_started = perf_counter()
            document.status, document.processing_stage, document.processing_error = "ready", "completed", None
            db.commit()
            logger.info("[INGESTION] document_id=%s stage=completed postgres_duration_seconds=%.3f", document_id, perf_counter() - completion_started)
            logger.info("[INGESTION] document_id=%s COMPLETED duration_seconds=%.3f", document_id, perf_counter() - started)
            return document
        except Exception as exc:
            db.rollback()
            # Rollback expires ORM state, so re-read before persisting the safe
            # result used by polling clients.
            failed = db.get(TeacherLibraryDocument, document.id)
            if failed is not None:
                failed.status = "failed"
                failed.processing_stage = "failed"
                failed.processing_error = self._failure_detail(exc)
                db.commit()
            logger.exception("[INGESTION] document_id=%s FAILED owner_id=%s filename=%s error_type=%s error=%s duration_seconds=%.3f", document_id, owner_id, filename, type(exc).__name__, exc, perf_counter() - started)
            raise TeacherLibraryIngestionError("Private document ingestion failed.", user_message=self._safe_failure_message(exc)) from exc
