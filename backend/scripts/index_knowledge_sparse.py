"""Explicit sparse-vector indexing for already dense-indexed KnowledgeChunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.database.session import SessionLocal
from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.services.knowledge_sparse_indexer import KnowledgeSparseIndexer
from app.services.qdrant_service import QdrantService, QdrantServiceError


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach lexical sparse vectors to existing dense Qdrant points; no embeddings or LLM calls.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--document-id", action="append", type=int, dest="document_ids")
    scope.add_argument("--all", action="store_true")
    scope.add_argument("--chunk-id", action="append", type=int, dest="chunk_ids", help="Retry or inspect only these canonical KnowledgeChunk IDs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count only; never changes Qdrant.")
    parser.add_argument("--verify", action="store_true", help="Read-only sparse coverage and stale-point audit.")
    parser.add_argument("--force", action="store_true", help="Reattach sparse vectors even when a point already has one.")
    args = parser.parse_args()
    with SessionLocal() as db:
        if args.chunk_ids:
            document_ids = list(db.scalars(
                select(KnowledgeChunk.document_id).where(KnowledgeChunk.id.in_(args.chunk_ids)).distinct()
            ))
        else:
            document_ids = args.document_ids or list(db.scalars(select(KnowledgeDocument.id).order_by(KnowledgeDocument.id)))
        try:
            indexer = KnowledgeSparseIndexer(qdrant=QdrantService())
            report = indexer.audit_coverage(db, document_ids=document_ids).public_dict() if args.verify else indexer.index(
                db, document_ids=document_ids, chunk_ids=args.chunk_ids, dry_run=args.dry_run, force=args.force,
            ).public_dict()
        except (QdrantServiceError, ValueError) as exc:
            parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
