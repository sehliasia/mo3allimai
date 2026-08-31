"""Owner-scoped private hybrid retrieval for TeacherLibraryChunk only."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.teacher_library import TeacherLibraryChunk, TeacherLibraryDocument
from app.services.embedding_providers import get_embedding_provider
from app.services.hybrid_retrieval import fuse_rrf
from app.services.qdrant_service import QdrantService
from app.services.sparse_embedding_service import MultilingualSparseEncoder


class PersonalRetrievalError(RuntimeError): pass
class PersonalDocumentAccessError(PersonalRetrievalError): pass

@dataclass(frozen=True)
class PersonalRetrievalResult:
    chunk_id: int; document_id: int; document_title: str; page_number: int | None; content: str; score: float


class PersonalRetrievalService:
    def __init__(self, *, qdrant: QdrantService | None = None, provider=None) -> None:
        self.qdrant = qdrant or QdrantService(); self.provider = provider or get_embedding_provider(); self.sparse = MultilingualSparseEncoder()

    @staticmethod
    def _authorized(db: Session, *, owner_id: int, document_ids: list[int]) -> list[int]:
        ids = sorted(set(document_ids))
        if not ids: raise PersonalDocumentAccessError("At least one personal document must be selected.")
        docs = db.scalars(select(TeacherLibraryDocument).where(TeacherLibraryDocument.id.in_(ids), TeacherLibraryDocument.owner_id == owner_id)).all()
        if len(docs) != len(ids): raise PersonalDocumentAccessError("A selected document is unavailable.")
        if any(doc.status != "ready" for doc in docs): raise PersonalDocumentAccessError("A selected document is not ready.")
        return ids

    def search(self, db: Session, *, query: str, owner_id: int, document_ids: list[int], top_k: int = 8) -> tuple[list[PersonalRetrievalResult], dict[str, int]]:
        started = perf_counter(); ids = self._authorized(db, owner_id=owner_id, document_ids=document_ids)
        vectors = self.provider.embed_queries([query]); embedding_ms = round((perf_counter() - started) * 1000)
        dense_started = perf_counter(); dense = self.qdrant.search_points(vectors[0], top_k=top_k, document_ids=ids, owner_id=owner_id, source_type="user_document"); dense_ms = round((perf_counter() - dense_started) * 1000)
        sparse_started = perf_counter(); encoded = self.sparse.encode(query); sparse = self.qdrant.search_sparse_points(encoded.indices, encoded.values, top_k=top_k, document_ids=ids, owner_id=owner_id, source_type="user_document") if encoded is not None and self.qdrant.sparse_vector_configured() else []; sparse_ms = round((perf_counter() - sparse_started) * 1000)
        fused = fuse_rrf(dense_hits=dense, sparse_hits=sparse, rrf_k=60, identity=lambda hit: str(getattr(hit, "id", "")))
        chunk_ids = [int((getattr(item.hit, "payload", {}) or {}).get("chunk_id")) for item in fused if str((getattr(item.hit, "payload", {}) or {}).get("chunk_id", "")).isdigit()]
        rows = db.execute(select(TeacherLibraryChunk, TeacherLibraryDocument).join(TeacherLibraryDocument).where(TeacherLibraryChunk.id.in_(chunk_ids), TeacherLibraryChunk.owner_id == owner_id, TeacherLibraryChunk.document_id.in_(ids), TeacherLibraryDocument.status == "ready")).all()
        canonical = {chunk.id: (chunk, document) for chunk, document in rows}
        results = [PersonalRetrievalResult(chunk_id=chunk_id, document_id=canonical[chunk_id][1].id, document_title=canonical[chunk_id][1].original_filename, page_number=canonical[chunk_id][0].page_number, content=canonical[chunk_id][0].content, score=item.rrf_score) for item in fused for chunk_id in [int((getattr(item.hit, "payload", {}) or {}).get("chunk_id", 0))] if chunk_id in canonical][:top_k]
        return results, {"personal_embedding_ms": embedding_ms, "personal_dense_search_ms": dense_ms, "personal_sparse_search_ms": sparse_ms, "personal_total_retrieval_ms": round((perf_counter() - started) * 1000)}
