"""Manual grounded Knowledge Base question CLI; it does not expose prompts or vectors."""

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
from app.services.llm_providers import get_llm_provider
from app.services.qdrant_service import QdrantService
from app.services.rag_prompt_builder import RAGPromptBuilder
from app.services.rag_service import RAGService, RAGServiceError
from app.services.reranker_providers import get_reranker_provider
from app.services.retrieval_service import RetrievalError, RetrievalService


def _diagnostic(error: Exception) -> str:
    """Safe local-only diagnostic: never stringify requests, headers, or settings."""
    cause = error.__cause__
    status = getattr(error, "status_code", None) or getattr(cause, "status_code", None)
    provider_message = getattr(error, "provider_message", None) or getattr(cause, "provider_message", None)
    return "\n".join([
        f"ERROR TYPE: {type(cause).__name__ if cause else type(error).__name__}",
        f"HTTP STATUS: {status if status is not None else 'unavailable'}",
        f"PROVIDER MESSAGE: {provider_message or 'unavailable'}",
        f"CAUSE: {type(cause).__name__ if cause else 'unavailable'}",
    ])


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask the grounded Knowledge Base; no hidden prompts are displayed.")
    parser.add_argument("query")
    parser.add_argument("--document-id", type=int, action="append", dest="document_ids")
    parser.add_argument("--language")
    parser.add_argument("--top-k", type=int, default=settings.rag_retrieval_top_k)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        with SessionLocal() as db:
            service = RAGService(
                retrieval=RetrievalService(provider=get_embedding_provider(), qdrant=QdrantService(), reranker=get_reranker_provider() if args.rerank else None),
                context_builder=ContextBuilder(max_chunks=settings.rag_context_max_chunks, max_tokens=settings.rag_context_max_tokens),
                prompt_builder=RAGPromptBuilder(), llm=get_llm_provider(settings), settings=settings,
            )
            response = service.answer_query(db, args.query, document_ids=args.document_ids, language=args.language, top_k=args.top_k, use_reranker=args.rerank)
    except RAGServiceError as exc:
        print(_diagnostic(exc), file=sys.stderr)
        raise SystemExit(2)
    except (RetrievalError, ValueError) as exc:
        parser.error(str(exc))
    print(f"QUESTION\n{args.query}\n\nANSWER\n{response.answer}")
    print(f"\nFINISH REASON: {response.finish_reason or 'unavailable'}")
    if response.output_token_count is not None:
        print(f"OUTPUT TOKENS: {response.output_token_count}")
    if (response.finish_reason or "").casefold() in {"length", "max_tokens", "max_output_tokens"}:
        print("WARNING: Response was truncated because the configured output token limit was reached.")
    print("\nSOURCES")
    for source in response.sources:
        pages = source.page_start if source.page_start == source.page_end else f"{source.page_start}—{source.page_end}"
        print(f"- {source.document_title} (id={source.document_id}), pages {pages}, chunks {source.chunk_ids}")
    if response.warnings:
        print("\nWARNINGS\n" + "\n".join(f"- {warning}" for warning in response.warnings))


if __name__ == "__main__":
    main()
