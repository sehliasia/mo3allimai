"""Manual vector-search smoke test; it neither indexes nor processes PDFs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.session import SessionLocal
from app.services.embedding_providers import get_embedding_provider
from app.services.qdrant_service import QdrantService
from app.services.reranker_providers import get_reranker_provider
from app.services.retrieval_service import RetrievalError, RetrievalFilters, RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual KnowledgeChunk semantic search; no indexing occurs.")
    parser.add_argument("query", help="Text to search for")
    parser.add_argument("--top-k", type=int, default=None, help="Final result count (default: configured final Top K)")
    parser.add_argument("--candidate-top-k", type=int, default=None, help="Qdrant candidate count before reranking")
    parser.add_argument("--rerank", action="store_true", help="Apply the optional Qwen3 cross-encoder reranker")
    parser.add_argument("--document-id", type=int, action="append", dest="document_ids")
    parser.add_argument("--language")
    parser.add_argument("--cefr-level")
    parser.add_argument("--content-type")
    parser.add_argument("--requires-vision", choices=("true", "false"))
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    filters = RetrievalFilters(
        document_ids=args.document_ids,
        language=args.language,
        cefr_level=args.cefr_level,
        content_type=args.content_type,
        requires_vision=None if args.requires_vision is None else args.requires_vision == "true",
    )
    try:
        with SessionLocal() as db:
            response = RetrievalService(
                provider=get_embedding_provider(),
                qdrant=QdrantService(),
                reranker=get_reranker_provider() if args.rerank else None,
            ).search(
                db,
                args.query,
                top_k=args.top_k,
                candidate_top_k=args.candidate_top_k,
                rerank=args.rerank,
                filters=filters,
            )
    except (RetrievalError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"QUERY: {response.query}\nMODEL: {response.model}\nFINAL TOP K: {response.top_k}"
        f"\nCANDIDATE TOP K: {response.candidate_top_k}\nRERANKING APPLIED: {response.reranking_applied}"
    )
    if response.reranker_model:
        print(f"RERANKER: {response.reranker_model}")
    if response.reranker_error:
        print(f"RERANKER FALLBACK: {response.reranker_error}")
    for result in response.results:
        print(
            f"\n#{result.rank}\noriginal_rank: {result.original_rank}\nvector_score: {result.vector_score:.6f}"
            f"\nreranked_rank: {result.reranked_rank}\nreranker_score: {result.reranker_score}"
            f"\ndocument: {result.document_title} "
            f"(id={result.document_id})\npage: {result.source_page_start}-{result.source_page_end}"
            f"\ncontent_type: {result.content_type}\nlanguage: {result.language}"
            f"\nrequires_vision: {result.requires_vision}\ncontent: {result.content[:700]}"
        )
    if response.stale_references_skipped:
        print(f"\nstale_references_skipped: {response.stale_references_skipped}")
    if not response.results:
        print("\nNo canonical results found.")


if __name__ == "__main__":
    main()
