"""Read-only Phase 6C diagnostic: assemble pedagogical knowledge without an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.database.session import SessionLocal
from app.services.cefr_knowledge_service import CEFRKnowledgeService
from app.services.context_builder import ContextBuilder
from app.services.embedding_providers import get_embedding_provider
from app.services.pedagogical_knowledge_service import (
    PedagogicalKnowledgeRequest,
    PedagogicalKnowledgeService,
    PedagogicalKnowledgeValidationError,
)
from app.services.qdrant_service import QdrantService
from app.services.reranker_providers import get_reranker_provider
from app.services.retrieval_service import RetrievalError, RetrievalService


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build pedagogical context only; no LLM or writes.")
    parser.add_argument("--level", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--objective")
    parser.add_argument("--language")
    parser.add_argument("--skill", action="append", default=[])
    parser.add_argument("--competency", action="append", default=[])
    parser.add_argument("--activity-type")
    parser.add_argument("--document-id", type=int, action="append", dest="document_ids", default=[])
    parser.add_argument("--source-language", help="Strict Qdrant source-language metadata filter")
    parser.add_argument("--source-cefr-level", help="Strict Qdrant source-CEFR metadata filter")
    parser.add_argument("--top-k", type=int, default=settings.rag_retrieval_top_k)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--no-cefr", action="store_true")
    parser.add_argument("--no-resources", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    request = PedagogicalKnowledgeRequest(
        cefr_level=args.level, topic=args.topic, objective=args.objective, language=args.language,
        skills=tuple(args.skill), competencies=tuple(args.competency), activity_type=args.activity_type,
        source_document_ids=tuple(args.document_ids), source_language=args.source_language,
        source_cefr_level=args.source_cefr_level, retrieval_top_k=args.top_k,
        include_cefr=not args.no_cefr, include_resources=not args.no_resources,
        rerank=True if args.rerank else None,
    )
    try:
        retrieval = None
        context_builder = None
        if request.include_resources:
            retrieval = RetrievalService(
                provider=get_embedding_provider(), qdrant=QdrantService(),
                reranker=get_reranker_provider() if request.rerank else None,
            )
            context_builder = ContextBuilder(
                max_chunks=settings.rag_context_max_chunks, max_tokens=settings.rag_context_max_tokens,
            )
        service = PedagogicalKnowledgeService(
            cefr=CEFRKnowledgeService(), retrieval=retrieval, context_builder=context_builder,
        )
        with SessionLocal() as db:
            result = service.build_context(db, request)
    except (PedagogicalKnowledgeValidationError, RetrievalError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
