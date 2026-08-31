"""Read-only PostgreSQL/Qdrant embedding-index audit; it never embeds or writes."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.core.config import get_settings
from app.database.session import SessionLocal
from app.models.knowledge_document import KnowledgeChunk, KnowledgeChunkEmbeddingStatus, KnowledgeDocument
from app.services.embedding_service import is_chunk_embedding_eligible
from app.services.qdrant_service import QdrantService, QdrantServiceError


@dataclass(frozen=True)
class DocumentAudit:
    document_id: int
    title: str
    chunks: int
    eligible: int
    indexed: int
    pending: int
    failed: int
    vector_ids: int
    qdrant_points: int | None
    missing_points: int | None
    stale_points: int | None
    status: str


def _status(*, chunks: int, eligible: int, indexed: int, pending: int, failed: int, missing: int | None, stale: int | None, qdrant_error: bool) -> str:
    if chunks == 0:
        return "NO_CHUNKS"
    if qdrant_error:
        return "ERROR"
    if eligible == indexed and pending == 0 and failed == 0 and missing == 0 and stale == 0:
        return "READY"
    if indexed == 0:
        return "NOT_INDEXED"
    return "PARTIAL"


def audit() -> tuple[list[DocumentAudit], Counter[str], Counter[str]]:
    settings = get_settings()
    qdrant = QdrantService(settings=settings)  # Lazy client: no embedding model is loaded.
    records: list[DocumentAudit] = []
    totals: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    with SessionLocal() as db:
        documents = list(db.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.id)))
        for document in documents:
            chunks = list(db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id)))
            status_counts = Counter(getattr(chunk.embedding_status, "value", chunk.embedding_status) for chunk in chunks)
            eligible = sum(is_chunk_embedding_eligible(chunk).eligible for chunk in chunks)
            indexed_chunks = [chunk for chunk in chunks if chunk.embedding_status == KnowledgeChunkEmbeddingStatus.indexed]
            canonical_ids = {chunk.vector_point_id for chunk in chunks if chunk.vector_point_id}
            expected_ids = {chunk.vector_point_id for chunk in indexed_chunks if chunk.vector_point_id}
            qdrant_error = False
            try:
                actual_ids = qdrant.document_point_ids(document.id)
                qdrant_points, missing, stale = len(actual_ids), len(expected_ids - actual_ids), len(actual_ids - canonical_ids)
            except QdrantServiceError:
                qdrant_error = True
                qdrant_points = missing = stale = None
            record = DocumentAudit(
                document.id, document.title, len(chunks), eligible, status_counts["indexed"], status_counts["pending"],
                status_counts["failed"], len(canonical_ids), qdrant_points, missing, stale,
                _status(chunks=len(chunks), eligible=eligible, indexed=status_counts["indexed"], pending=status_counts["pending"], failed=status_counts["failed"], missing=missing, stale=stale, qdrant_error=qdrant_error),
            )
            records.append(record)
            statuses[record.status] += 1
            totals.update({"chunks_total": record.chunks, "chunks_indexed": record.indexed, "chunks_pending": record.pending, "chunks_failed": record.failed})
    return records, totals, statuses


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    records, totals, statuses = audit()
    print(f"ACTIVE EMBEDDING MODEL: {settings.rag_embedding_model_id}")
    print(f"ACTIVE EMBEDDING DIMENSION: {settings.rag_embedding_dimension}")
    print("ID | Document | Chunks | Eligible | Indexed | Pending | Failed | Vector IDs | Qdrant | Missing | Stale | Status")
    for item in records:
        qdrant = item.qdrant_points if item.qdrant_points is not None else "ERROR"
        missing = item.missing_points if item.missing_points is not None else "ERROR"
        stale = item.stale_points if item.stale_points is not None else "ERROR"
        print(f"{item.document_id} | {item.title} | {item.chunks} | {item.eligible} | {item.indexed} | {item.pending} | {item.failed} | {item.vector_ids} | {qdrant} | {missing} | {stale} | {item.status}")
    print("\nSUMMARY")
    print(f"documents_total: {len(records)}")
    print(f"documents_ready: {statuses['READY']}")
    print(f"documents_partial: {statuses['PARTIAL']}")
    print(f"documents_not_indexed: {statuses['NOT_INDEXED']}")
    print(f"documents_without_chunks: {statuses['NO_CHUNKS']}")
    print(f"documents_error: {statuses['ERROR']}")
    for key in ("chunks_total", "chunks_indexed", "chunks_pending", "chunks_failed"):
        print(f"{key}: {totals[key]}")


if __name__ == "__main__":
    main()
