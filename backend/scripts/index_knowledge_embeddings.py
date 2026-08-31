"""Explicitly index canonical PostgreSQL KnowledgeChunks into Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.database.session import SessionLocal
from app.models.knowledge_document import KnowledgeDocument
from app.services.embedding_providers import get_embedding_provider
from app.services.embedding_service import EmbeddingService
from app.services.knowledge_embedding_indexer import KnowledgeEmbeddingIndexer
from app.services.qdrant_service import QdrantService


def main() -> None:
    parser = argparse.ArgumentParser(description="Index knowledge chunks into Qdrant; no PDF processing occurs.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--document-id", action="append", type=int, dest="document_ids")
    scope.add_argument("--all", action="store_true", help="Explicitly index all canonical documents.")
    parser.add_argument("--force", action="store_true", help="Re-embed even unchanged indexed chunks.")
    parser.add_argument("--reconcile", action="store_true", help="Delete stale vectors for the selected documents only.")
    args = parser.parse_args()

    with SessionLocal() as db:
        document_ids = args.document_ids
        if args.all:
            document_ids = list(db.scalars(select(KnowledgeDocument.id).order_by(KnowledgeDocument.id)))
        provider = get_embedding_provider()
        indexer = KnowledgeEmbeddingIndexer(
            embedding_service=EmbeddingService(),
            provider=provider,
            qdrant=QdrantService(),
        )
        report = indexer.index(db, document_ids=document_ids, force=args.force, reconcile=args.reconcile)
    print(json.dumps(report.public_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
