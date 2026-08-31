"""Explicit selected-document materialization; it never starts PDF extraction or OCR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database.session import SessionLocal
from app.services.legacy_knowledge_chunk_materializer import (
    LegacyKnowledgeChunkMaterializer,
    LegacyMaterializationError,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize one historical Knowledge Base document from trusted local artifacts only.")
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument(
        "--rebuild-from-cache",
        action="store_true",
        help="Explicitly replace this document's persisted chunks using only its existing extraction cache.",
    )
    arguments = parser.parse_args()
    with SessionLocal() as db:
        try:
            report = LegacyKnowledgeChunkMaterializer().materialize(
                db,
                document_id=arguments.document_id,
                rebuild_from_cache=arguments.rebuild_from_cache,
            )
            db.commit()
        except Exception as error:
            db.rollback()
            message = str(error) if isinstance(error, LegacyMaterializationError) else "Materialization failed safely; no chunks were written."
            raise SystemExit(message) from error
    print(json.dumps(report.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
