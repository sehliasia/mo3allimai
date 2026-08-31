"""Read-only cache diagnostic for one worksheet page; never opens a PDF or OCR."""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.document_chunker import ChunkMetadataBuilder
from app.services.document_parser_service import DocumentParserService
from app.services.worksheet_structure import WorksheetStructureBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one cached worksheet page without OCR or PDF parsing.")
    parser.add_argument("--document-id", required=True, type=int)
    parser.add_argument("--page", required=True, type=int)
    args = parser.parse_args()
    parsed = DocumentParserService().load_cached_extraction_for_rebuild(document_id=args.document_id)
    if parsed is None:
        raise SystemExit("No extraction cache is available for this document.")
    page = next((value for value in parsed.page_extractions if value.page_number == args.page), None)
    if page is None:
        raise SystemExit("The requested page is not in the extraction cache.")
    builder = WorksheetStructureBuilder()
    blocks = builder.blocks(page.document, args.page)
    sections = builder.sections(page.document, args.page, has_image=bool(page.images))
    print(json.dumps({
        "document_id": args.document_id,
        "page": args.page,
        "ocr_invoked": False,
        "docling_conversion_invoked": False,
        "raw_ocr_units": [{"text": block.text, "bbox": block.bbox, "label": block.label} for block in blocks],
        "structural_quality": [section.structural_quality for section in sections],
        "worksheet_sections": [
            {"exercise_number": section.number, "text": section.text, "has_image": section.has_image,
             "requires_vision": section.requires_vision,
             "text_for_embedding": ChunkMetadataBuilder.contextual_text("cached worksheet", [], section.text)}
            for section in sections
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
