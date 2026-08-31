"""Read-only PostgreSQL/Qdrant consistency audit with an explicit scoped repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.session import SessionLocal
from app.services.embedding_consistency_service import EmbeddingConsistencyService
from app.services.embedding_providers import get_embedding_provider
from app.services.embedding_service import EmbeddingService
from app.services.knowledge_embedding_indexer import KnowledgeEmbeddingIndexer
from app.services.qdrant_service import QdrantService


def _service() -> EmbeddingConsistencyService:
    return EmbeddingConsistencyService(qdrant=QdrantService(), embedding_service=EmbeddingService())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical KnowledgeChunk/Qdrant consistency.")
    parser.add_argument("--document-id", action="append", type=int, required=True, dest="document_ids")
    parser.add_argument("--repair", action="store_true", help="Explicitly repair only the selected document IDs.")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    service = _service()
    output: list[dict[str, object]] = []
    with SessionLocal() as db:
        for document_id in sorted(set(args.document_ids)):
            before = service.audit(db, document_id)
            entry: dict[str, object] = {"before": before.public_dict(), "repair_applied": False}
            if args.repair:
                # Orphans are never canonical. Deleting only their explicit point
                # IDs is idempotent and cannot affect another document.
                if before.qdrant_only_point_ids:
                    service.qdrant.delete_points(before.qdrant_only_point_ids)
                after_cleanup = service.audit(db, document_id)
                reconciled = service.repair_canonical_state(db, after_cleanup)
                # If canonical chunks are missing from Qdrant, reuse the existing
                # bounded indexer. It marks PostgreSQL indexed only after upsert.
                indexed = 0
                if after_cleanup.pg_only_chunk_ids:
                    provider = get_embedding_provider()
                    report = KnowledgeEmbeddingIndexer(
                        embedding_service=EmbeddingService(), provider=provider, qdrant=service.qdrant,
                    ).index(db, document_ids=[document_id])
                    indexed = report.chunks_marked_indexed
                entry.update({
                    "repair_applied": True,
                    "stale_points_deleted": len(before.qdrant_only_point_ids),
                    "pg_chunks_reconciled": reconciled,
                    "missing_chunks_indexed": indexed,
                    "after": service.audit(db, document_id).public_dict(),
                })
            output.append(entry)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
