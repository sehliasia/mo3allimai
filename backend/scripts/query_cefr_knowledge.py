from __future__ import annotations
import argparse, sys
from pathlib import Path
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database.session import SessionLocal
from app.models.cefr_knowledge import CEFRScale
from app.models.knowledge_document import KnowledgeDocument
from app.services.cefr_knowledge_service import CEFRKnowledgeService
def main():
    parser=argparse.ArgumentParser(description="Query deterministic structured CEFR knowledge."); parser.add_argument("--level", required=True); parser.add_argument("--scale"); args=parser.parse_args(); service=CEFRKnowledgeService()
    with SessionLocal() as db:
        descriptors=service.get_descriptors(db, level_code=args.level, scale_name=args.scale)
        for descriptor in descriptors:
            scale=db.get(CEFRScale, descriptor.scale_id); reference = descriptor.reference_level.code if descriptor.reference_level else None; print(f"LEVEL: {args.level}\nSCALE: {scale.name}\nSTATUS: {descriptor.status}\nREFERENCE LEVEL: {reference or '-'}\nDESCRIPTOR:\n{descriptor.descriptor_text or 'No descriptor available.'}")
            for source in service.get_descriptor_sources(db, descriptor.id):
                document=db.get(KnowledgeDocument, source.document_id); print(f"SOURCE: {document.title}\npage {source.page_start}-{source.page_end}\nchunk {source.chunk_id}")
if __name__ == "__main__": main()
