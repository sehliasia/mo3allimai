"""Read-only audit of deterministic CEFR knowledge and its provenance."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import ProgrammingError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.session import SessionLocal
from app.models.cefr_knowledge import CEFRDescriptor, CEFRDescriptorSource, CEFRLevel, CEFRScale
from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.services.cefr_parser import classify_no_descriptor, is_leading_fragment, validate_available_descriptor


def main() -> None:
    try:
        with SessionLocal() as db:
            levels = list(db.scalars(select(CEFRLevel).order_by(CEFRLevel.sort_order)))
            scales = list(db.scalars(select(CEFRScale)))
            descriptors = list(db.scalars(select(CEFRDescriptor)))
            sources = list(db.scalars(select(CEFRDescriptorSource)))
            referenced = {source.descriptor_id for source in sources}
            invalid_document_sources = sum(db.get(KnowledgeDocument, source.document_id) is None for source in sources)
            invalid_chunk_sources = sum(db.get(KnowledgeChunk, source.chunk_id) is None for source in sources)
            duplicate_groups = Counter((item.level_id, item.scale_id, item.descriptor_hash) for item in descriptors)
            duplicate_source_groups = Counter((item.descriptor_id, item.chunk_id) for item in sources)
            available = [item for item in descriptors if item.status == "AVAILABLE"]
            available_embedded_serialized_cell = sum(validate_available_descriptor(item.descriptor_text or "", source_chunk_ids=[1]) == "EMBEDDED_SERIALIZED_CELL" for item in available)
            available_tail_contamination = sum(validate_available_descriptor(item.descriptor_text or "", source_chunk_ids=[1]) == "TAIL_CONTAMINATION" for item in available)
            available_leading_fragment = sum(is_leading_fragment(item.descriptor_text or "") for item in available)
            available_no_descriptor_misclassified = sum(classify_no_descriptor(item.descriptor_text or "") is not None for item in available)
            invalid_reference_level = sum(item.reference_level_id is not None and item.reference_level is None for item in descriptors)
            source_document_titles = sorted({document.title for source in sources if (document := db.get(KnowledgeDocument, source.document_id)) is not None})

            print(f"levels: {len(levels)}")
            print(f"scales: {len(scales)}")
            print(f"descriptors: {len(descriptors)}")
            print(f"available: {len(available)}")
            print(f"no_descriptor_available: {sum(item.status == 'NO_DESCRIPTOR_AVAILABLE' for item in descriptors)}")
            print(f"descriptors_without_sources: {sum(item.id not in referenced for item in descriptors)}")
            print(f"invalid_document_sources: {invalid_document_sources}")
            print(f"invalid_chunk_sources: {invalid_chunk_sources}")
            print(f"duplicate_descriptor_identities: {sum(count > 1 for count in duplicate_groups.values())}")
            print(f"available_embedded_serialized_cell: {available_embedded_serialized_cell}")
            print(f"available_tail_contamination: {available_tail_contamination}")
            print(f"available_leading_fragment: {available_leading_fragment}")
            print(f"available_no_descriptor_misclassified: {available_no_descriptor_misclassified}")
            print(f"invalid_reference_level: {invalid_reference_level}")
            print(f"orphan_descriptor_sources: {invalid_document_sources + invalid_chunk_sources}")
            print(f"duplicate_descriptor_source_links: {sum(count > 1 for count in duplicate_source_groups.values())}")
            print(f"source_documents: {', '.join(source_document_titles) or '(none)'}")
            for level in levels:
                print(f"{level.code}: {sum(item.level_id == level.id for item in descriptors)}")
    except ProgrammingError:
        print("CEFR structured tables are unavailable. Apply migration 20260824_0013 before auditing.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
