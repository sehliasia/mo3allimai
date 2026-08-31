"""Read canonical chunks and attach sparse vectors to their existing Qdrant points."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge_document import KnowledgeChunk, KnowledgeChunkEmbeddingStatus
from app.services.embedding_service import is_chunk_embedding_eligible
from app.services.qdrant_service import QdrantCollectionCompatibilityError, QdrantService
from app.services.sparse_embedding_service import MultilingualSparseEncoder


@dataclass
class KnowledgeSparseIndexReport:
    document_ids: list[int]
    chunks_scanned: int = 0
    chunks_eligible: int = 0
    sparse_representations_generated: int = 0
    points_updated: int = 0
    skipped: int = 0
    failed: int = 0
    dense_dimension: int = 0
    collection: str = ""
    sparse_vector_name: str = ""
    sparse_configured_before: bool = False
    sparse_configured_after: bool = False
    sparse_schema_action: str = "none"
    qdrant_server_version: str = ""
    qdrant_points_before: int = 0
    dry_run: bool = False
    failures: list[dict[str, object]] = field(default_factory=list)

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SparseCoverageReport:
    eligible_chunks: int
    eligible_with_sparse: int
    eligible_missing_sparse: int
    eligible_missing_qdrant_point: int
    canonical_noneligible_points: int
    stale_qdrant_points: int
    missing_sparse_chunks: list[dict[str, object]] = field(default_factory=list)
    canonical_noneligible_chunks: list[dict[str, object]] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeSparseIndexer:
    def __init__(self, *, qdrant: QdrantService, encoder: MultilingualSparseEncoder | None = None) -> None:
        self.qdrant = qdrant
        self.encoder = encoder or MultilingualSparseEncoder()

    @staticmethod
    def _failure(chunk: KnowledgeChunk, *, category: str, error: Exception | None, point_state: str, encoding_succeeded: bool) -> dict[str, object]:
        return {
            "knowledge_chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "qdrant_point_id": chunk.vector_point_id,
            "exception_class": type(error).__name__ if error else None,
            "failure_category": category,
            "qdrant_point_state": point_state,
            "sparse_encoding_succeeded": encoding_succeeded,
            "safe_error": (str(error)[:300] if error else None),
        }

    @staticmethod
    def _failure_category(error: Exception) -> str:
        text = str(error).casefold()
        if any(marker in text for marker in ("timeout", "connect", "network", "temporar")):
            return "transient_qdrant_failure"
        return "qdrant_vector_update_rejected"

    def _sparse_eligibility(self, chunk: KnowledgeChunk) -> tuple[bool, str | None, Any | None]:
        """The single canonical sparse-index eligibility decision for index and verify."""
        embedding = is_chunk_embedding_eligible(chunk)
        if not embedding.eligible:
            return False, embedding.reason, None
        if chunk.embedding_status != KnowledgeChunkEmbeddingStatus.indexed:
            return False, "dense_embedding_not_indexed", None
        if not chunk.vector_point_id:
            return False, "missing_dense_point_id", None
        text = self.encoder.indexed_text(heading_context=list(chunk.heading_context or []), content=chunk.content)
        try:
            vector = self.encoder.encode(text)
        except Exception as exc:
            return False, f"sparse_encoding_error:{type(exc).__name__}", None
        if vector is None:
            return False, "empty_sparse_representation", None
        return True, None, vector

    def preflight(self, db: Session, *, document_ids: list[int], chunk_ids: list[int] | None = None) -> KnowledgeSparseIndexReport:
        if not document_ids:
            raise ValueError("Sparse indexing requires an explicit document scope.")
        self.qdrant.validate_collection()
        sparse_before = self.qdrant.sparse_vector_configured()
        report = KnowledgeSparseIndexReport(
            document_ids=sorted(set(document_ids)), dense_dimension=self.qdrant.dimension,
            collection=self.qdrant.collection_name, sparse_vector_name=self.qdrant.sparse_vector_name,
            sparse_configured_before=sparse_before,
            sparse_schema_action="none" if sparse_before else "would_create",
            qdrant_server_version=self.qdrant.server_version(),
            qdrant_points_before=self.qdrant.point_count(),
        )
        query = select(KnowledgeChunk).where(KnowledgeChunk.document_id.in_(report.document_ids))
        if chunk_ids:
            query = query.where(KnowledgeChunk.id.in_(chunk_ids))
        for chunk in db.scalars(query).yield_per(200):
            report.chunks_scanned += 1
            eligible, _, _ = self._sparse_eligibility(chunk)
            if not eligible:
                report.skipped += 1
                continue
            report.chunks_eligible += 1
        return report

    def index(self, db: Session, *, document_ids: list[int], chunk_ids: list[int] | None = None, dry_run: bool = False, force: bool = False) -> KnowledgeSparseIndexReport:
        report = self.preflight(db, document_ids=document_ids, chunk_ids=chunk_ids)
        report.dry_run = dry_run
        if dry_run:
            return report
        # Explicit mutation follows a complete dense/schema preflight.
        self.qdrant.ensure_sparse_vector()
        report.sparse_configured_after = self.qdrant.sparse_vector_configured()
        report.sparse_schema_action = "none" if report.sparse_configured_before else "created"
        query = select(KnowledgeChunk).where(KnowledgeChunk.document_id.in_(report.document_ids))
        if chunk_ids:
            query = query.where(KnowledgeChunk.id.in_(chunk_ids))
        for chunk in db.scalars(query).yield_per(100):
            eligible, reason, vector = self._sparse_eligibility(chunk)
            if not eligible:
                if reason and reason.startswith("sparse_encoding_error:"):
                    report.failed += 1
                    report.failures.append(self._failure(chunk, category="sparse_encoding_failure", error=None, point_state="not_checked", encoding_succeeded=False))
                continue
            report.sparse_representations_generated += 1
            try:
                point_state = self.qdrant.sparse_point_state(chunk.vector_point_id)
                if point_state == "missing":
                    report.failed += 1
                    report.failures.append(self._failure(chunk, category="missing_qdrant_point", error=None, point_state=point_state, encoding_succeeded=True))
                    continue
                if not force and point_state == "sparse_present":
                    report.skipped += 1
                    continue
                self.qdrant.update_sparse_vectors([{"id": chunk.vector_point_id, "indices": vector.indices, "values": vector.values}])
                report.points_updated += 1
            except Exception as exc:
                report.failed += 1
                report.failures.append(self._failure(chunk, category=self._failure_category(exc), error=exc, point_state="inspection_or_update_failed", encoding_succeeded=True))
        return report

    def audit_coverage(self, db: Session, *, document_ids: list[int]) -> SparseCoverageReport:
        """Read-only coverage and identity audit; it never changes Qdrant or PostgreSQL."""
        self.qdrant.validate_collection()
        qdrant_ids = self.qdrant.collection_point_ids()
        canonical_ids: set[str] = set(
            db.scalars(select(KnowledgeChunk.vector_point_id).where(KnowledgeChunk.vector_point_id.is_not(None)))
        )
        eligible: dict[str, KnowledgeChunk] = {}
        noneligible: dict[str, tuple[KnowledgeChunk, str | None]] = {}
        for chunk in db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id.in_(document_ids))).yield_per(200):
            is_eligible, reason, _ = self._sparse_eligibility(chunk)
            if is_eligible:
                eligible[chunk.vector_point_id] = chunk
            elif chunk.vector_point_id:
                noneligible[chunk.vector_point_id] = (chunk, reason)
        present = missing_sparse = missing_point = 0
        missing_chunks: list[dict[str, object]] = []
        for point_id, chunk in eligible.items():
            state = self.qdrant.sparse_point_state(point_id)
            if state == "sparse_present":
                present += 1
            elif state == "missing":
                missing_point += 1
                missing_chunks.append(self._failure(chunk, category="missing_qdrant_point", error=None, point_state=state, encoding_succeeded=True))
            else:
                missing_sparse += 1
                missing_chunks.append(self._failure(chunk, category="missing_sparse_vector", error=None, point_state=state, encoding_succeeded=True))
        noneligible_chunks: list[dict[str, object]] = []
        for point_id, (chunk, reason) in noneligible.items():
            if point_id in qdrant_ids:
                state = self.qdrant.sparse_point_state(point_id)
                noneligible_chunks.append({
                    "knowledge_chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "qdrant_point_id": point_id,
                    "canonical_eligibility": False,
                    "eligibility_reason": reason,
                    "qdrant_point_state": state,
                })
        return SparseCoverageReport(
            eligible_chunks=len(eligible), eligible_with_sparse=present,
            eligible_missing_sparse=missing_sparse, eligible_missing_qdrant_point=missing_point,
            canonical_noneligible_points=len(noneligible_chunks),
            stale_qdrant_points=len(qdrant_ids - canonical_ids),
            missing_sparse_chunks=missing_chunks,
            canonical_noneligible_chunks=noneligible_chunks,
        )
