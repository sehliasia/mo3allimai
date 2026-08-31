"""Manual Phase 6D CLI. It builds context then validates JSON-only LLM output."""

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
from app.services.llm_providers import get_llm_provider
from app.services.pedagogical_knowledge_service import PedagogicalKnowledgeRequest, PedagogicalKnowledgeService
from app.services.qdrant_service import QdrantService
from app.services.retrieval_service import RetrievalService
from app.services.structured_generation_service import StructuredGenerationService


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Validate generic structured pedagogy JSON; no generator business logic.")
    parser.add_argument("--level", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--skill", action="append", default=[])
    parser.add_argument("--language", required=True)
    parser.add_argument("--top-k", type=int, default=settings.rag_retrieval_top_k)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with SessionLocal() as db:
        context = PedagogicalKnowledgeService(
            cefr=CEFRKnowledgeService(),
            retrieval=RetrievalService(provider=get_embedding_provider(), qdrant=QdrantService()),
            context_builder=ContextBuilder(max_chunks=settings.rag_context_max_chunks, max_tokens=settings.rag_context_max_tokens),
        ).build_context(db, PedagogicalKnowledgeRequest(
            cefr_level=args.level, topic=args.topic, skills=tuple(args.skill), language=args.language,
            retrieval_top_k=args.top_k,
        ))
        result = StructuredGenerationService(llm=get_llm_provider(settings), settings=settings).generate(context)
    print(json.dumps({"generation": {"provider_model": result.provider_model, "finish_reason": result.finish_reason, "parse_succeeded": result.parse_succeeded}, "validation": asdict(result.validation), "sources": [asdict(source) for source in context.sources], "diagnostics": {"retrieved_count": context.retrieved_count, "selected_count": context.selected_count}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
