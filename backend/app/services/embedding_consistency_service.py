"""Targeted, canonical PostgreSQL/Qdrant consistency checks for embeddings."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge_document import KnowledgeChunk, KnowledgeChunkEmbeddingStatus, KnowledgeDocument
from app.services.embedding_service import (
    EmbeddingService,
    embedding_input_hash,
    is_chunk_embedding_eligible,
    vector_point_id,
)
from app.services.qdrant_service import QdrantService


@dataclass(frozen=True)
class EmbeddingConsistencyAudit:
    document_id: int
    document_title: str
    chunks_total: int
    chunks_indexed: int
    chunks_pending: int
    chunks_failed: int
    chunks_with_vector_id: int
    qdrant_points: int
    matching_chunk_ids: list[int]
    pg_only_chunk_ids: list[int]
    qdrant_only_point_ids: list[str]
    duplicate_qdrant_chunk_ids: list[int]
    payload_mismatch_point_ids: list[str]
    dimension_mismatch_point_ids: list[str]
    model_mismatch_point_ids: list[str]
    point_id_mismatch_point_ids: list[str]
    model: str
    dimension: int

    @property
    def is_exact_match(self) -> bool:
        return not any((
            self.pg_only_chunk_ids,
            self.qdrant_only_point_ids,
            self.duplicate_qdrant_chunk_ids,
            self.payload_mismatch_point_ids,
            self.dimension_mismatch_point_ids,
            self.model_mismatch_point_ids,
            self.point_id_mismatch_point_ids,
        ))

    def public_dict(self) -> dict[str, object]:
        # Matching IDs can be hundreds long and add no diagnostic value. Keep
        # divergent IDs explicit, while reporting the matched set by count.
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "chunks_total": self.chunks_total,
            "chunks_indexed": self.chunks_indexed,
            "chunks_pending": self.chunks_pending,
            "chunks_failed": self.chunks_failed,
            "chunks_with_vector_id": self.chunks_with_vector_id,
            "qdrant_points": self.qdrant_points,
            "matching_chunk_count": len(self.matching_chunk_ids),
            "pg_only_chunk_ids": self.pg_only_chunk_ids,
            "qdrant_only_point_ids": self.qdrant_only_point_ids,
            "duplicate_qdrant_chunk_ids": self.duplicate_qdrant_chunk_ids,
            "payload_mismatch_point_ids": self.payload_mismatch_point_ids,
            "dimension_mismatch_point_ids": self.dimension_mismatch_point_ids,
            "model_mismatch_point_ids": self.model_mismatch_point_ids,
            "point_id_mismatch_point_ids": self.point_id_mismatch_point_ids,
            "model": self.model,
            "dimension": self.dimension,
        }


@dataclass(frozen=True)
class EmbeddingConsistencyRepair:
    document_id: int
    pg_chunks_reconciled: int
    stale_points_deleted: int
    missing_chunks_indexed: int
    post_repair: EmbeddingConsistencyAudit


class EmbeddingConsistencyService:
    """Repairs only an explicitly scoped document, after a canonical audit."""

    def __init__(self, *, qdrant: QdrantService, embedding_service: EmbeddingService) -> None:
        if qdrant.dimension != embedding_service.dimension:
            raise ValueError("Qdrant and embedding dimensions must match.")
        self.qdrant = qdrant
        self.embedding_service = embedding_service

    @staticmethod
    def _payload(point: Any) -> dict[str, Any]:
        payload = getattr(point, "payload", {}) or {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _vector_dimension(point: Any) -> int | None:
        vector = getattr(point, "vector", None)
        if isinstance(vector, list):
            return len(vector)
        return None

    def audit(self, db: Session, document_id: int) -> EmbeddingConsistencyAudit:
        document = db.get(KnowledgeDocument, document_id)
        if document is None:
            raise ValueError(f"Knowledge document {document_id} does not exist.")
        chunks = list(db.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id).order_by(KnowledgeChunk.id)
        ))
        canonical = {chunk.id: chunk for chunk in chunks}
        points = self.qdrant.document_points(document_id, with_vectors=True)
        by_chunk: dict[int, list[Any]] = {}
        qdrant_only: list[str] = []
        payload_mismatches: list[str] = []
        dimension_mismatches: list[str] = []
        model_mismatches: list[str] = []
        point_id_mismatches: list[str] = []
        for point in points:
            point_id = str(getattr(point, "id", ""))
            payload = self._payload(point)
            try:
                chunk_id = int(payload["chunk_id"])
            except (KeyError, TypeError, ValueError):
                qdrant_only.append(point_id)
                continue
            chunk = canonical.get(chunk_id)
            if chunk is None:
                qdrant_only.append(point_id)
                continue
            by_chunk.setdefault(chunk_id, []).append(point)
            if payload.get("document_id") != document_id:
                payload_mismatches.append(point_id)
            if payload.get("source_page_start") != chunk.source_page_start or payload.get("source_page_end") != chunk.source_page_end:
                payload_mismatches.append(point_id)
            if payload.get("content_type") != chunk.content_type:
                payload_mismatches.append(point_id)
            if self._vector_dimension(point) != self.embedding_service.dimension:
                dimension_mismatches.append(point_id)
            if payload.get("embedding_model") != self.embedding_service.model_id:
                model_mismatches.append(point_id)
            if point_id != vector_point_id(chunk):
                point_id_mismatches.append(point_id)

        duplicates = sorted(chunk_id for chunk_id, items in by_chunk.items() if len(items) > 1)
        matching = sorted(by_chunk)
        pg_only = sorted(set(canonical) - set(by_chunk))
        statuses = Counter(getattr(chunk.embedding_status, "value", chunk.embedding_status) for chunk in chunks)
        return EmbeddingConsistencyAudit(
            document_id=document.id,
            document_title=document.title,
            chunks_total=len(chunks),
            chunks_indexed=statuses[KnowledgeChunkEmbeddingStatus.indexed.value],
            chunks_pending=statuses[KnowledgeChunkEmbeddingStatus.pending.value],
            chunks_failed=statuses[KnowledgeChunkEmbeddingStatus.failed.value],
            chunks_with_vector_id=sum(1 for chunk in chunks if chunk.vector_point_id),
            qdrant_points=len(points),
            matching_chunk_ids=matching,
            pg_only_chunk_ids=pg_only,
            qdrant_only_point_ids=sorted(set(qdrant_only)),
            duplicate_qdrant_chunk_ids=duplicates,
            payload_mismatch_point_ids=sorted(set(payload_mismatches)),
            dimension_mismatch_point_ids=sorted(set(dimension_mismatches)),
            model_mismatch_point_ids=sorted(set(model_mismatches)),
            point_id_mismatch_point_ids=sorted(set(point_id_mismatches)),
            model=self.embedding_service.model_id,
            dimension=self.embedding_service.dimension,
        )

    def repair_canonical_state(self, db: Session, audit: EmbeddingConsistencyAudit) -> int:
        """Mark existing vectors indexed only after their compatibility was proven."""
        if not audit.is_exact_match:
            return 0
        changed = 0
        for chunk in db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(audit.matching_chunk_ids))):
            decision = is_chunk_embedding_eligible(chunk)
            if not decision.eligible:
                continue
            expected_hash = embedding_input_hash(
                chunk.content_for_embedding,
                model_id=self.embedding_service.model_id,
                config_version=self.embedding_service.config_version,
            )
            expected_point_id = vector_point_id(chunk)
            if (
                chunk.embedding_status != KnowledgeChunkEmbeddingStatus.indexed
                or chunk.embedding_model != self.embedding_service.model_id
                or chunk.embedding_input_hash != expected_hash
                or chunk.vector_point_id != expected_point_id
            ):
                chunk.embedding_status = KnowledgeChunkEmbeddingStatus.indexed
                chunk.embedding_model = self.embedding_service.model_id
                chunk.embedding_input_hash = expected_hash
                chunk.vector_point_id = expected_point_id
                chunk.embedding_error = None
                changed += 1
        if changed:
            db.commit()
        return changed
