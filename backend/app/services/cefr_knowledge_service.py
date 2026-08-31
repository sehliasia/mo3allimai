"""Deterministic SQL lookups over structured CEFR knowledge."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.cefr_knowledge import CEFRDescriptor, CEFRDescriptorSource, CEFRLevel, CEFRScale
from app.services.cefr_parser import normalize_text


class CEFRKnowledgeService:
    def get_level(self, db: Session, code: str): return db.scalar(select(CEFRLevel).where(CEFRLevel.code == code.upper()))
    def list_levels(self, db: Session): return list(db.scalars(select(CEFRLevel).order_by(CEFRLevel.sort_order)))
    def list_scales(self, db: Session, level_code: str | None = None):
        query = select(CEFRScale).order_by(CEFRScale.name)
        if level_code:
            query = query.join(CEFRDescriptor).join(CEFRLevel, CEFRDescriptor.level_id == CEFRLevel.id).where(CEFRLevel.code == level_code.upper()).distinct()
        return list(db.scalars(query))
    def get_descriptors(self, db: Session, *, level_code: str, scale_id: int | None = None, scale_name: str | None = None):
        query = select(CEFRDescriptor).join(CEFRLevel, CEFRDescriptor.level_id == CEFRLevel.id).where(CEFRLevel.code == level_code.upper())
        if scale_id is not None: query = query.where(CEFRDescriptor.scale_id == scale_id)
        if scale_name is not None: query = query.join(CEFRScale).where(CEFRScale.normalized_name == normalize_text(scale_name).casefold())
        return list(db.scalars(query.order_by(CEFRDescriptor.id)))
    def get_descriptor_sources(self, db: Session, descriptor_id: int):
        return list(db.scalars(select(CEFRDescriptorSource).where(CEFRDescriptorSource.descriptor_id == descriptor_id).order_by(CEFRDescriptorSource.source_order)))
