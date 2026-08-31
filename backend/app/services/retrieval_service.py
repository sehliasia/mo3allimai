"""Canonical PostgreSQL hydration for Qdrant-ranked KnowledgeChunk references."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.services.embedding_service import EmbeddingProvider
from app.services.qdrant_service import QdrantService, QdrantServiceError
from app.services.reranker_service import RerankerProvider, RerankerService, RerankingError
from app.services.sparse_embedding_service import MultilingualSparseEncoder
from app.services.hybrid_retrieval import fuse_rrf
from app.services.pedagogical_retrieval_ranker import PedagogicalRankingRequest, PedagogicalRetrievalRanker
from app.core.config import get_settings
from app.services.retrieval_pipeline import resolve_effective_retrieval_pipeline


class RetrievalError(RuntimeError):
    """A query embedding, vector search, or canonical hydration operation failed."""


class EmptyQueryError(RetrievalError):
    """A manual or future API query contains no searchable text."""


@dataclass(frozen=True)
class RetrievalFilters:
    document_ids: list[int] | None = None
    language: str | None = None
    cefr_level: str | None = None
    content_type: str | None = None
    requires_vision: bool | None = None


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    score: float
    vector_score: float
    original_rank: int
    chunk_id: int
    document_id: int
    document_title: str
    source_page_start: int | None
    source_page_end: int | None
    content_type: str
    language: str | None
    cefr_level: str | None
    structural_quality: str | None
    has_image: bool
    requires_vision: bool
    heading_context: list[str]
    content: str
    reranker_score: float | None = None
    reranked_rank: int | None = None
    appeared_in_dense: bool = True
    dense_rank: int | None = None
    dense_score: float | None = None
    appeared_in_sparse: bool = False
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    fused_rank: int | None = None
    pedagogical_adjustment_total: float = 0.0
    role_adjustment: float = 0.0
    concreteness_adjustment: float = 0.0
    level_adjustment: float = 0.0
    skill_adjustment: float = 0.0
    final_score: float | None = None
    final_rank: int | None = None
    adjustment_reasons: list[str] | None = None
    pedagogical_role: str | None = None
    role_source: str | None = None
    is_concrete_classroom_material: bool | None = None
    concreteness_reasons: list[str] | None = None
    skill_evidence_reason: str | None = None
    composition_category: str | None = None
    composition_rank: int | None = None
    composition_selection_reason: str | None = None
    composition_diversity_override: bool = False
    neighbor_of: int | None = None


@dataclass(frozen=True)
class RetrievalResponse:
    query: str
    model: str
    top_k: int
    results: list[RetrievalResult]
    stale_references_skipped: int
    candidate_top_k: int
    reranking_applied: bool = False
    reranker_model: str | None = None
    reranker_error: str | None = None
    retrieval_mode: str = "dense"
    sparse_fallback_used: bool = False
    dense_candidate_count: int = 0
    sparse_candidate_count: int = 0
    union_candidate_count: int = 0
    pedagogical_desired_in_union: bool | None = None
    pedagogical_union_diagnostics: dict[str, object] | None = None
    composition_candidates: list[RetrievalResult] | None = None
    pipeline_trace: dict[str, object] | None = None
    fallback_reason: str | None = None


class RetrievalService:
    """Searches vectors, then hydrates only canonical persisted rows from PostgreSQL."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider,
        qdrant: QdrantService,
        reranker: RerankerProvider | None = None,
        candidate_top_k: int = 20,
        final_top_k: int = 5, mode: str | None = None, dense_top_k: int | None = None, sparse_top_k: int | None = None, rrf_k: int | None = None,
        pedagogical_ranking_enabled: bool | None = None,
    ) -> None:
        if provider.dimension != qdrant.dimension:
            raise ValueError("Retrieval provider dimension must match Qdrant collection dimension.")
        self.provider = provider
        self.qdrant = qdrant
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k
        self.final_top_k = final_top_k
        settings = get_settings()
        pipeline = resolve_effective_retrieval_pipeline(settings)
        self.mode = mode or pipeline.retrieval_mode
        if self.mode not in {"dense", "hybrid"}:
            raise ValueError("RETRIEVAL_MODE must be dense or hybrid.")
        self.hybrid_dense_top_k = dense_top_k or pipeline.dense_top_k
        self.hybrid_sparse_top_k = sparse_top_k or pipeline.sparse_top_k
        self.rrf_k = rrf_k or pipeline.rrf_k
        self.sparse_encoder = MultilingualSparseEncoder()
        self.pedagogical_ranking_enabled = pipeline.pedagogical_ranking if pedagogical_ranking_enabled is None else pedagogical_ranking_enabled
        self.pedagogical_ranker = PedagogicalRetrievalRanker()

    @staticmethod
    def _payload(hit: Any) -> dict[str, Any]:
        payload = getattr(hit, "payload", {}) or {}
        return payload if isinstance(payload, dict) else {}

    def search(
        self,
        db: Session,
        query: str,
        *,
        top_k: int | None = None,
        candidate_top_k: int | None = None,
        rerank: bool = False,
        filters: RetrievalFilters | None = None,
        pedagogical_request: PedagogicalRankingRequest | None = None,
        composition_pool_size: int | None = None,
    ) -> RetrievalResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise EmptyQueryError("A non-empty query is required.")
        final_top_k = self.final_top_k if top_k is None else top_k
        requested_candidate_top_k = candidate_top_k
        if final_top_k < 1:
            raise RetrievalError("top_k must be at least 1.")
        if requested_candidate_top_k is None:
            requested_candidate_top_k = self.candidate_top_k if rerank else final_top_k
        if requested_candidate_top_k < 1:
            raise RetrievalError("candidate_top_k must be at least 1.")
        if rerank:
            requested_candidate_top_k = max(requested_candidate_top_k, final_top_k)
        filters = filters or RetrievalFilters()
        if self.mode == "hybrid" and not rerank:
            return self._search_hybrid(db, normalized_query, final_top_k, filters, pedagogical_request, composition_pool_size)
        started = perf_counter()
        embedding_started = perf_counter()
        try:
            vectors = self.provider.embed_queries([normalized_query])
        except Exception as exc:
            raise RetrievalError("Query embedding failed.") from exc
        if len(vectors) != 1 or len(vectors[0]) != self.provider.dimension:
            actual = len(vectors[0]) if vectors else 0
            raise RetrievalError(
                f"Query embedding dimension {actual} does not match {self.provider.dimension}."
            )
        embedding_ms = round((perf_counter() - embedding_started) * 1000)
        embedding_diagnostics = getattr(self.provider, "last_diagnostics", lambda: {})()
        dense_started = perf_counter()
        try:
            hits = self.qdrant.search_points(
                vectors[0],
                top_k=requested_candidate_top_k,
                document_ids=filters.document_ids,
                language=filters.language,
                cefr_level=filters.cefr_level,
                content_type=filters.content_type,
                requires_vision=filters.requires_vision,
            )
        except QdrantServiceError as exc:
            raise RetrievalError("Vector search is unavailable.") from exc
        dense_search_ms = round((perf_counter() - dense_started) * 1000)

        chunk_ids: list[int] = []
        for hit in hits:
            try:
                chunk_ids.append(int(self._payload(hit)["chunk_id"]))
            except (KeyError, TypeError, ValueError):
                continue
        try:
            rows = (
                db.execute(
                    select(KnowledgeChunk, KnowledgeDocument)
                    .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
                    .where(KnowledgeChunk.id.in_(set(chunk_ids)))
                ).all()
                if chunk_ids
                else []
            )
        except Exception as exc:
            raise RetrievalError("Canonical chunk hydration failed.") from exc
        canonical = {chunk.id: (chunk, document) for chunk, document in rows}

        results: list[RetrievalResult] = []
        stale_references = 0
        for rank, hit in enumerate(hits, start=1):
            try:
                chunk_id = int(self._payload(hit)["chunk_id"])
            except (KeyError, TypeError, ValueError):
                stale_references += 1
                continue
            row = canonical.get(chunk_id)
            if row is None:
                stale_references += 1
                continue
            chunk, document = row
            metadata = chunk.chunk_metadata or {}
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(getattr(hit, "score", 0.0)),
                    vector_score=float(getattr(hit, "score", 0.0)),
                    original_rank=rank,
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_title=document.title,
                    source_page_start=chunk.source_page_start,
                    source_page_end=chunk.source_page_end,
                    content_type=chunk.content_type,
                    language=document.language or metadata.get("language"),
                    cefr_level=document.cefr_level or metadata.get("level"),
                    structural_quality=metadata.get("structural_quality"),
                    has_image=bool(metadata.get("has_image")),
                    requires_vision=bool(metadata.get("requires_vision")),
                    heading_context=list(chunk.heading_context or []),
                    content=chunk.content,
                )
            )
        reranking_applied = False
        reranker_model: str | None = None
        reranker_error: str | None = None
        if rerank and results:
            if self.reranker is None:
                reranker_error = "Reranker is not configured."
            else:
                reranker_model = self.reranker.model_id
                try:
                    results = RerankerService(self.reranker).rerank(
                        normalized_query, results, top_k=final_top_k
                    )
                    reranking_applied = True
                except RerankingError:
                    reranker_error = "Reranker unavailable; returned vector-ranked results."
                    results = results[:final_top_k]
        else:
            results = results[:final_top_k]
        return RetrievalResponse(
            query=normalized_query,
            model=self.provider.model_id,
            top_k=final_top_k,
            results=results,
            stale_references_skipped=stale_references,
            candidate_top_k=requested_candidate_top_k,
            reranking_applied=reranking_applied,
            reranker_model=reranker_model,
            reranker_error=reranker_error,
            retrieval_mode="dense", dense_candidate_count=len(hits), union_candidate_count=len(hits),
            pipeline_trace={
                "embedding_ms": embedding_ms,
                **embedding_diagnostics,
                "dense_search_ms": dense_search_ms,
                "total_retrieval_ms": round((perf_counter() - started) * 1000),
            },
        )

    def _identity(self, hit: Any) -> str:
        return str(getattr(hit, "id", None) or self._payload(hit).get("chunk_id", ""))

    def _search_hybrid(self, db: Session, query: str, final_top_k: int, filters: RetrievalFilters, pedagogical_request: PedagogicalRankingRequest | None, composition_pool_size: int | None) -> RetrievalResponse:
        started = perf_counter()
        dense_hits: list[Any] = []
        sparse_hits: list[Any] = []
        dense_error = sparse_error = None
        embedding_started = perf_counter()
        dense_search_ms = sparse_search_ms = rrf_ms = h4_ms = 0
        try:
            vectors = self.provider.embed_queries([query])
            if len(vectors) != 1 or len(vectors[0]) != self.provider.dimension:
                raise RetrievalError("Query embedding dimension is invalid.")
        except Exception as exc:
            dense_error = exc
        embedding_ms = round((perf_counter() - embedding_started) * 1000)
        embedding_diagnostics = getattr(self.provider, "last_diagnostics", lambda: {})()
        if not dense_error:
            try:
                dense_started = perf_counter()
                dense_hits = self.qdrant.search_points(vectors[0], top_k=self.hybrid_dense_top_k, document_ids=filters.document_ids, language=filters.language, cefr_level=filters.cefr_level, content_type=filters.content_type, requires_vision=filters.requires_vision)
                dense_search_ms = round((perf_counter() - dense_started) * 1000)
            except Exception as exc:
                dense_error = exc
        sparse_started = perf_counter()
        try:
            vector = self.sparse_encoder.encode(query)
            if vector is not None:
                sparse_hits = self.qdrant.search_sparse_points(vector.indices, vector.values, top_k=self.hybrid_sparse_top_k, document_ids=filters.document_ids, language=filters.language, cefr_level=filters.cefr_level, content_type=filters.content_type, requires_vision=filters.requires_vision)
                sparse_search_ms = round((perf_counter() - sparse_started) * 1000)
        except Exception as exc:
            sparse_error = exc
        if dense_error and not sparse_hits:
            raise RetrievalError("Vector search is unavailable.") from dense_error
        rrf_started = perf_counter()
        fused = fuse_rrf(dense_hits=dense_hits, sparse_hits=sparse_hits, rrf_k=self.rrf_k, identity=self._identity)
        rrf_ms = round((perf_counter() - rrf_started) * 1000)
        chunk_ids = [int(self._payload(item.hit).get("chunk_id")) for item in fused if str(self._payload(item.hit).get("chunk_id", "")).isdigit()]
        try:
            rows = db.execute(select(KnowledgeChunk, KnowledgeDocument).join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id).where(KnowledgeChunk.id.in_(set(chunk_ids)))).all() if chunk_ids else []
        except Exception as exc:
            raise RetrievalError("Canonical chunk hydration failed.") from exc
        canonical = {chunk.id: (chunk, document) for chunk, document in rows}
        results: list[RetrievalResult] = []
        stale = 0
        for fused_rank, item in enumerate(fused, start=1):
            raw_id = self._payload(item.hit).get("chunk_id")
            try: chunk_id = int(raw_id)
            except (TypeError, ValueError): stale += 1; continue
            row = canonical.get(chunk_id)
            if row is None: stale += 1; continue
            chunk, document = row; metadata = chunk.chunk_metadata or {}
            results.append(RetrievalResult(rank=fused_rank, score=item.rrf_score, vector_score=item.dense_score or 0.0, original_rank=item.dense_rank or item.sparse_rank or fused_rank, chunk_id=chunk.id, document_id=document.id, document_title=document.title, source_page_start=chunk.source_page_start, source_page_end=chunk.source_page_end, content_type=chunk.content_type, language=document.language or metadata.get("language"), cefr_level=document.cefr_level or metadata.get("level"), structural_quality=metadata.get("structural_quality"), has_image=bool(metadata.get("has_image")), requires_vision=bool(metadata.get("requires_vision")), heading_context=list(chunk.heading_context or []), content=chunk.content, appeared_in_dense=item.dense_rank is not None, dense_rank=item.dense_rank, dense_score=item.dense_score, appeared_in_sparse=item.sparse_rank is not None, sparse_rank=item.sparse_rank, sparse_score=item.sparse_score, rrf_score=item.rrf_score, fused_rank=fused_rank))
        desired_in_union = None
        union_diagnostics = None
        ranking_fallback = False
        if self.pedagogical_ranking_enabled and pedagogical_request is not None:
            h4_started = perf_counter()
            pre_h4_results = list(results)
            try:
                desired_in_union = self.pedagogical_ranker.desired_candidate_exists(results, pedagogical_request)
                results = self.pedagogical_ranker.rank(results, pedagogical_request)
                union_diagnostics = self.pedagogical_ranker.union_diagnostics(results, pedagogical_request)
            except Exception:
                results = pre_h4_results
                ranking_fallback = True
            h4_ms = round((perf_counter() - h4_started) * 1000)
        composition_candidates = results[:composition_pool_size] if composition_pool_size else None
        fallback_reason = (
            "pedagogical_ranking_fallback" if ranking_fallback
            else "sparse_unavailable_dense_fallback" if sparse_error and dense_hits
            else "dense_unavailable_sparse_fallback" if dense_error and sparse_hits
            else None
        )
        return RetrievalResponse(query=query, model=self.provider.model_id, top_k=final_top_k, results=results[:final_top_k], stale_references_skipped=stale, candidate_top_k=max(self.hybrid_dense_top_k, self.hybrid_sparse_top_k), retrieval_mode="hybrid", sparse_fallback_used=bool(sparse_error), dense_candidate_count=len(dense_hits), sparse_candidate_count=len(sparse_hits), union_candidate_count=len(fused), pedagogical_desired_in_union=desired_in_union, pedagogical_union_diagnostics=union_diagnostics, composition_candidates=composition_candidates, fallback_reason=fallback_reason, pipeline_trace={"embedding_ms": embedding_ms, **embedding_diagnostics, "dense_search_ms": dense_search_ms, "sparse_search_ms": sparse_search_ms, "rrf_ms": rrf_ms, "h4_ranking_ms": h4_ms, "total_retrieval_ms": round((perf_counter() - started) * 1000), "fallback_used": bool(fallback_reason)})
