"""Dry-run first importer for deterministic CEFR structures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.session import SessionLocal
from app.services.cefr_import_service import CEFRImportService


def main() -> None:
    parser = argparse.ArgumentParser(description="Import structured CEFR data from canonical chunks only.")
    parser.add_argument("--document-id", type=int, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Inspect only; this is the default.")
    mode.add_argument("--apply", action="store_true", help="Persist only after reviewing a dry run.")
    parser.add_argument("--replace-document", action="store_true", help="Atomically replace only this document's existing CEFR projection; requires --apply.")
    parser.add_argument("--show-parsed", type=int, default=0, metavar="N", help="Show up to N parsed examples with provenance.")
    parser.add_argument("--show-rejected", type=int, default=0, metavar="N", help="Show up to N rejected examples with reasons.")
    parser.add_argument("--show-integrity-failures", type=int, default=0, metavar="N", help="Show up to N records blocked by the final AVAILABLE integrity gate.")
    args = parser.parse_args()
    if args.replace_document and not args.apply:
        parser.error("--replace-document requires explicit --apply")

    with SessionLocal() as db:
        service = CEFRImportService()
        report = (
            service.replace_document(db, document_id=args.document_id, dry_run=False)
            if args.replace_document
            else service.import_document(db, document_id=args.document_id, dry_run=not args.apply)
        )
        if args.apply:
            db.commit()
    print(json.dumps(report.json_summary(show_parsed=max(args.show_parsed, 0), show_rejected=max(args.show_rejected, 0), show_integrity_failures=max(args.show_integrity_failures, 0)), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
