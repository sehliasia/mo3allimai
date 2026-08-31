"""Production indexing coordinator: PostgreSQL chunks -> embeddings -> Qdrant."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge_document import KnowledgeChunk
from app.services.embedding_service import EmbeddingCandidate, EmbeddingProvider, EmbeddingService
from app.services.qdrant_service import QdrantService


@dataclass
class KnowledgeEmbeddingIndexReport:
    document_ids: list[int]
    chunks_seen: int = 0
    chunks_eligible: int = 0
    chunks_skipped: int = 0
    chunks_embedded: int = 0
    points_upserted: int = 0
    chunks_marked_indexed: int = 0
    chunks_failed: int = 0
    stale_points_deleted: int = 0
    model: str = ""
    dimension: int = 0
    collection: str = ""

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


class KnowledgeEmbeddingIndexer:
    """Coordinates separate systems without pretending they share a transaction."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        provider: EmbeddingProvider,
        qdrant: QdrantService,
    ) -> None:
        if provider.model_id != embedding_service.model_id:
            raise ValueError("Embedding provider model must match EmbeddingService model.")
        if provider.dimension != embedding_service.dimension:
            raise ValueError("Embedding provider dimension must match EmbeddingService dimension.")
        if qdrant.dimension != provider.dimension:
            raise ValueError("Qdrant collection dimension must match the embedding provider dimension.")
        self.embedding_service = embedding_service
        self.provider = provider
        self.qdrant = qdrant

    @staticmethod
    def _points(candidates: list[EmbeddingCandidate], vectors: list[list[float]], model_id: str) -> list[dict[str, object]]:
        if len(vectors) != len(candidates):
            raise ValueError("Embedding provider returned a vector count mismatch.")
        points: list[dict[str, object]] = []
        for candidate, vector in zip(candidates, vectors, strict=True):
            payload = {
                **candidate.payload,
                "chunk_id": candidate.chunk_id,
                "embedding_input_hash": candidate.embedding_input_hash,
                "embedding_model": model_id,
            }
            points.append({"id": candidate.vector_point_id, "vector": vector, "payload": payload})
        return points

    @staticmethod
    def _safe_error(error: Exception) -> str:
        return f"{type(error).__name__}: {str(error)}"[:1000]

    def index(
        self,
        db: Session,
        *,
        document_ids: list[int] | None = None,
        force: bool = False,
        reconcile: bool = False,
    ) -> KnowledgeEmbeddingIndexReport:
        if not document_ids:
            raise ValueError("Indexing requires an explicit document scope.")
        report = KnowledgeEmbeddingIndexReport(
            document_ids=sorted(set(document_ids)),
            model=self.provider.model_id,
            dimension=self.provider.dimension,
            collection=self.qdrant.collection_name,
        )
        self.qdrant.ensure_collection()
        for batch in self.embedding_service.iter_embedding_batches(
            db,
            document_ids=report.document_ids,
            force=force,
        ):
            report.chunks_seen += batch.scanned
            report.chunks_skipped += batch.skipped
            report.chunks_eligible += len(batch.candidates)
            if not batch.candidates:
                continue
            try:
                vectors = self.provider.embed_documents(
                    [candidate.embedding_text for candidate in batch.candidates]
                )
                if any(len(vector) != self.provider.dimension for vector in vectors):
                    raise ValueError("Embedding provider returned an unexpected vector dimension.")
                points = self._points(batch.candidates, vectors, self.provider.model_id)
                self.qdrant.upsert_points(points)
            except Exception as exc:
                # A failed upsert/inference never becomes indexed. It is safe to
                # retry because Qdrant point IDs are deterministic.
                self.embedding_service.mark_failed(db, batch.candidates, self._safe_error(exc))
                db.commit()
                report.chunks_failed += len(batch.candidates)
                continue

            # Qdrant write succeeded with wait=True. This next commit is still a
            # separate transaction; a later retry upserts the same deterministic IDs.
            try:
                self.embedding_service.mark_indexed(db, batch.candidates)
                db.commit()
            except Exception:
                db.rollback()
                raise
            report.chunks_embedded += len(batch.candidates)
            report.points_upserted += len(points)
            report.chunks_marked_indexed += len(batch.candidates)

        if reconcile:
            for document_id in report.document_ids:
                report.stale_points_deleted += self.reconcile_document(db, document_id)
        return report

    def reconcile_document(self, db: Session, document_id: int) -> int:
        """Delete only stale Qdrant points within one explicitly requested document."""
        self.qdrant.ensure_collection()
        canonical_ids = set(
            db.scalars(
                select(KnowledgeChunk.vector_point_id).where(
                    KnowledgeChunk.document_id == document_id,
                    KnowledgeChunk.vector_point_id.is_not(None),
                )
            )
        )
        stale_ids = sorted(self.qdrant.document_point_ids(document_id) - canonical_ids)
        return self.qdrant.delete_points(stale_ids)
