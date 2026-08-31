"""Read-only diagnostic for canonical RAG context construction; no LLM is called."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.database.session import SessionLocal
from app.services.context_builder import ContextBuilder
from app.services.embedding_providers import get_embedding_provider
from app.services.qdrant_service import QdrantService
from app.services.reranker_providers import get_reranker_provider
from app.services.retrieval_service import RetrievalError, RetrievalFilters, RetrievalService


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Inspect retrieved KnowledgeChunk context; no LLM is called.")
    parser.add_argument("query")
    parser.add_argument("--document-id", type=int, action="append", dest="document_ids")
    parser.add_argument("--top-k", type=int, default=settings.rag_retrieval_top_k)
    parser.add_argument("--candidate-top-k", type=int, default=None)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=settings.rag_context_max_chunks)
    parser.add_argument("--max-tokens", type=int, default=settings.rag_context_max_tokens)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        with SessionLocal() as db:
            response = RetrievalService(
                provider=get_embedding_provider(), qdrant=QdrantService(),
                reranker=get_reranker_provider() if args.rerank else None,
            ).search(
                db, args.query, top_k=args.top_k, candidate_top_k=args.candidate_top_k,
                rerank=args.rerank, filters=RetrievalFilters(document_ids=args.document_ids),
            )
            context = ContextBuilder(max_chunks=args.max_chunks, max_tokens=args.max_tokens).build(
                args.query, response.results, db=db
            )
    except (RetrievalError, ValueError) as exc:
        parser.error(str(exc))
    print("RETRIEVED CHUNKS:")
    for result in response.results:
        print(f"- chunk={result.chunk_id} rank={result.rank} page={result.source_page_start}-{result.source_page_end}")
    print(f"\nINCLUDED: {context.included_chunk_ids}\nEXCLUDED: {context.excluded_chunk_ids}")
    print(f"ESTIMATED TOKENS: {context.estimated_token_count}\n\n{context.context_text or 'No eligible context.'}")
    if context.warnings:
        print("\nWARNINGS:\n" + "\n".join(f"- {warning}" for warning in context.warnings))


if __name__ == "__main__":
    main()
