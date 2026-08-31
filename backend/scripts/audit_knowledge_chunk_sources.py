"""Read-only report of selected-document materialization sources; never processes PDFs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database.session import SessionLocal
from app.services.legacy_knowledge_chunk_materializer import LegacyKnowledgeChunkMaterializer


def main() -> None:
    with SessionLocal() as db:
        report = [item.__dict__ for item in LegacyKnowledgeChunkMaterializer().audit(db)]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
