"""Idempotent, auditable CEFR import from canonical PostgreSQL chunks only."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.cefr_knowledge import CEFRDescriptor, CEFRDescriptorSource, CEFRLevel, CEFRScale
from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.services.cefr_parser import CEFRParser, ParsedCEFRDescriptor, RejectedCEFRCandidate, classify_no_descriptor, is_valid_long_scale, normalize_text, validate_available_descriptor, validate_scale_name

CORE_LEVELS = (
    ("PRE-A1", "Pre-A1", 0, True), ("A1", "A1", 10, True), ("A2", "A2", 20, True),
    ("B1", "B1", 30, True), ("B2", "B2", 40, True), ("C1", "C1", 50, True), ("C2", "C2", 60, True),
)
_ROW_PREFIX = re.compile(r"^(?:PR(?:E|É)[- ]?A1|A1|A2\+?|B1\+?|B2\+?|C1|C2)\s*(?:[,;])", re.IGNORECASE)


@dataclass
class CEFRImportReport:
    document_id: int
    document_title: str
    dry_run: bool
    chunks_scanned: int = 0
    pages_scanned: int = 0
    candidate_rows_detected: int = 0
    parsed_available: int = 0
    no_descriptor_available: int = 0
    duplicates: int = 0
    ambiguous: int = 0
    truncated: int = 0
    unsupported_structure: int = 0
    embedded_next_row_marker: int = 0
    mid_sentence_truncated: int = 0
    no_descriptor_with_reference: int = 0
    narrative_false_positive: int = 0
    unsupported_non_candidate_chunks: int = 0
    available_integrity_failures: int = 0
    available_suspicious_scales: int = 0
    available_no_descriptor_misclassified: int = 0
    available_mid_sentence: int = 0
    available_narrative: int = 0
    valid_long_scales_recovered: int = 0
    incomplete_scale_fragments_rejected: int = 0
    tail_contamination_rejected: int = 0
    available_tail_contamination: int = 0
    leading_fragment_rejected: int = 0
    leading_fragment_reconstructed: int = 0
    available_leading_fragment: int = 0
    embedded_serialized_cell_rejected: int = 0
    available_embedded_serialized_cell: int = 0
    quality_rejected: int = 0
    persisted: int = 0
    source_links_seen: int = 0
    source_links_unique: int = 0
    source_links_deduplicated: int = 0
    source_links_existing: int = 0
    source_links_inserted: int = 0
    levels_discovered: set[str] = field(default_factory=set)
    scales_discovered: set[str] = field(default_factory=set)
    suspicious_scale_candidates: set[str] = field(default_factory=set)
    descriptors_by_level: Counter[str] = field(default_factory=Counter)
    parsed_examples: list[dict] = field(default_factory=list)
    rejected_examples: list[dict] = field(default_factory=list)
    no_descriptor_examples: list[dict] = field(default_factory=list)
    integrity_failure_examples: list[dict] = field(default_factory=list)

    @property
    def parsed(self) -> int:
        """Compatibility alias for the previous dry-run report field."""
        return self.parsed_available

    @property
    def descriptors_without_source(self) -> int:
        return 0

    def json_summary(self, *, show_parsed: int = 0, show_rejected: int = 0, show_integrity_failures: int = 0) -> dict:
        summary = asdict(self)
        summary["levels_discovered"] = sorted(self.levels_discovered)
        summary["scales_discovered"] = sorted(self.scales_discovered)
        summary["unique_levels"] = sorted(self.levels_discovered)
        summary["unique_scales"] = sorted(self.scales_discovered)
        summary["suspicious_scale_candidates"] = sorted(self.suspicious_scale_candidates)
        summary["descriptors_by_level"] = dict(sorted(self.descriptors_by_level.items()))
        summary["parsed"] = self.parsed
        summary["descriptors_without_source"] = self.descriptors_without_source
        summary.pop("parsed_examples", None)
        summary.pop("rejected_examples", None)
        summary.pop("no_descriptor_examples", None)
        summary.pop("integrity_failure_examples", None)
        if show_parsed:
            summary["parsed_examples"] = self.parsed_examples[:show_parsed]
            summary["no_descriptor_examples"] = self.no_descriptor_examples[:show_parsed]
        if show_rejected:
            summary["rejected_examples"] = self.rejected_examples[:show_rejected]
        if show_integrity_failures:
            summary["integrity_failure_examples"] = self.integrity_failure_examples[:show_integrity_failures]
        return summary


class CEFRImportService:
    def __init__(self, parser: CEFRParser | None = None) -> None:
        self.parser = parser or CEFRParser()

    @staticmethod
    def seed_levels(db: Session) -> dict[str, CEFRLevel]:
        existing = {level.code: level for level in db.scalars(select(CEFRLevel)).all()}
        for code, label, order, core in CORE_LEVELS:
            if code not in existing:
                existing[code] = CEFRLevel(code=code, label=label, sort_order=order, is_core_reference_level=core)
                db.add(existing[code])
        db.flush()
        return existing

    @staticmethod
    def _level(db: Session, levels: dict[str, CEFRLevel], code: str) -> CEFRLevel:
        if code not in levels:
            base = code.rstrip("+")
            base_order = levels.get(base).sort_order if base in levels else 1000
            levels[code] = CEFRLevel(code=code, label=code, sort_order=base_order + 1, is_core_reference_level=False)
            db.add(levels[code])
            db.flush()
        return levels[code]

    @staticmethod
    def _scale(db: Session, name: str) -> CEFRScale:
        normalized = normalize_text(name).casefold()
        scale = db.scalar(select(CEFRScale).where(CEFRScale.normalized_name == normalized))
        if scale is None:
            scale = CEFRScale(name=name, normalized_name=normalized)
            db.add(scale)
            db.flush()
        return scale

    @staticmethod
    def _same_structure(current: KnowledgeChunk, following: KnowledgeChunk) -> bool:
        return (
            current.source_page_start == following.source_page_start
            and current.source_page_end == following.source_page_end
            and current.heading_context == following.heading_context
            and (following.chunk_metadata or {}).get("structural_quality") != "layout_unreliable"
        )

    @classmethod
    def _compatible_descriptor_continuation(cls, current: KnowledgeChunk, following: KnowledgeChunk) -> bool:
        if not cls._same_structure(current, following):
            return False
        first, second = (current.content or "").strip(), (following.content or "").strip()
        if not (first and second and _ROW_PREFIX.match(first) and "=" in first and not _ROW_PREFIX.match(second) and not first.endswith((".", "!", "?", ";", ":"))):
            return False
        # A continuation can complete an existing descriptor, never supply its
        # missing beginning after a bare ``=``.
        descriptor_beginning = first.split("=", 1)[1].strip()
        return bool(descriptor_beginning and validate_available_descriptor(descriptor_beginning, source_chunk_ids=[current.id], reject_incomplete_tail=False) is None)

    @classmethod
    def _compatible_scale_continuation(cls, current: KnowledgeChunk, following: KnowledgeChunk) -> bool:
        """Join only a visibly split `LEVEL, Scale = descriptor` header."""
        if not cls._same_structure(current, following):
            return False
        first, second = (current.content or "").strip(), (following.content or "").strip()
        return bool(first and second and _ROW_PREFIX.match(first) and "=" not in first and "=" in second and not _ROW_PREFIX.match(second) and first[-1:].isalnum() and second[:1].islower())

    def _compatible_same_row_fragment_continuation(self, current: KnowledgeChunk, following: KnowledgeChunk) -> bool:
        """Join only a repeated explicit row whose second value is a fragment."""
        if not self._same_structure(current, following):
            return False
        combined = self.parser.parse(
            f"{(current.content or '').rstrip()} {(following.content or '').lstrip()}",
            source_chunk_ids=[current.id, following.id],
        )
        return len(combined.records) == 1 and combined.records[0].reconstructed_from_fragments

    @staticmethod
    def _default_scale(chunk: KnowledgeChunk) -> str | None:
        headings = [item for item in (chunk.heading_context or []) if item]
        if not headings:
            return None
        candidate = headings[-1]
        normalized = candidate.upper().replace(" ", "-").replace("PRÉ", "PRE").rstrip("+")
        return None if normalized in {"PRE-A1", "A1", "A2", "B1", "B2", "C1", "C2"} else candidate

    @staticmethod
    def _source_metadata(chunk: KnowledgeChunk, source_chunks: list[KnowledgeChunk]) -> dict:
        return {
            "document_id": chunk.document_id,
            "page_start": chunk.source_page_start,
            "page_end": chunk.source_page_end,
            "chunk_ids": [source.id for source in source_chunks],
        }

    def _record_rejection(self, report: CEFRImportReport, candidate: RejectedCEFRCandidate, source: dict) -> None:
        reason = candidate.reason
        if reason == "TRUNCATED_DESCRIPTOR":
            report.truncated += 1
        elif reason == "INCOMPLETE_SCALE_FRAGMENT":
            report.incomplete_scale_fragments_rejected += 1
            report.ambiguous += 1
        elif reason == "UNSUPPORTED_STRUCTURE":
            report.unsupported_structure += 1
        elif reason == "EMBEDDED_NEXT_ROW_MARKER":
            report.embedded_next_row_marker += 1
            report.ambiguous += 1
        elif reason == "EMBEDDED_SERIALIZED_CELL":
            report.embedded_serialized_cell_rejected += 1
            report.ambiguous += 1
        elif reason == "MID_SENTENCE_TRUNCATED":
            report.mid_sentence_truncated += 1
            report.truncated += 1
        elif reason == "NARRATIVE_FALSE_POSITIVE":
            report.narrative_false_positive += 1
            report.ambiguous += 1
        elif reason == "TAIL_CONTAMINATION":
            # This is intentionally distinct from an integrity failure in an
            # AVAILABLE record: the parser rejected it before acceptance.
            report.tail_contamination_rejected += 1
            report.ambiguous += 1
        elif reason == "AVAILABLE_LEADING_FRAGMENT":
            report.leading_fragment_rejected += 1
            report.truncated += 1
        else:
            report.ambiguous += 1
            if candidate.scale_name:
                report.suspicious_scale_candidates.add(candidate.scale_name)
        report.rejected_examples.append({
            **source,
            "reason": reason,
            "level": candidate.level_code,
            "scale": candidate.scale_name,
            "text_preview": candidate.text_preview,
        })

    def _records(self, chunks: list[KnowledgeChunk], report: CEFRImportReport):
        scanned_pages: set[int] = set()
        index = 0
        while index < len(chunks):
            chunk = chunks[index]
            source_chunks = [chunk]
            report.chunks_scanned += 1
            if chunk.source_page_start is not None:
                scanned_pages.add(chunk.source_page_start)
            if (chunk.chunk_metadata or {}).get("structural_quality") == "layout_unreliable" or not (chunk.content or "").strip():
                report.quality_rejected += 1
                index += 1
                continue

            text = chunk.content
            if index + 1 < len(chunks):
                following = chunks[index + 1]
                if self._compatible_descriptor_continuation(chunk, following) or self._compatible_scale_continuation(chunk, following) or self._compatible_same_row_fragment_continuation(chunk, following):
                    text = f"{text.rstrip()} {following.content.lstrip()}"
                    source_chunks.append(following)
                    report.chunks_scanned += 1
                    if following.source_page_start is not None:
                        scanned_pages.add(following.source_page_start)
                    index += 1

            result = self.parser.parse(text, source_chunk_ids=[source.id for source in source_chunks], default_scale=self._default_scale(chunk))
            if not result.records and not result.rejected:
                report.unsupported_non_candidate_chunks += 1
            report.candidate_rows_detected += result.candidate_rows_detected
            source = self._source_metadata(chunk, source_chunks)
            for rejected in result.rejected:
                self._record_rejection(report, rejected, source)
            yield chunk, result.records, source_chunks
            index += 1
        report.pages_scanned = len(scanned_pages)

    def _record_parsed(self, report: CEFRImportReport, record: ParsedCEFRDescriptor, source: dict) -> bool:
        diagnostic = {
            **source,
            "level": record.level_code,
            "scale": record.scale_name,
            "status": record.status,
            "descriptor_preview": (record.descriptor_text or "Pas de descripteur disponible.")[:240],
        }
        if record.reference_level:
            diagnostic["reference_level"] = record.reference_level
        if record.reconstructed_from_fragments:
            diagnostic["reconstructed_from_fragments"] = True
        integrity_reason = None
        if record.status == "AVAILABLE":
            integrity_reason = validate_available_descriptor(record.descriptor_text or "", source_chunk_ids=record.source_chunk_ids)
            integrity_reason = integrity_reason or validate_scale_name(record.scale_name)
            if integrity_reason:
                report.available_integrity_failures += 1
                report.integrity_failure_examples.append({**diagnostic, "reason": integrity_reason})
                if integrity_reason == "CONTAMINATED_SCALE":
                    report.available_suspicious_scales += 1
                elif integrity_reason == "NO_DESCRIPTOR_MISCLASSIFIED":
                    report.available_no_descriptor_misclassified += 1
                elif integrity_reason == "MID_SENTENCE_TRUNCATED":
                    report.available_mid_sentence += 1
                elif integrity_reason == "NARRATIVE_FALSE_POSITIVE":
                    report.available_narrative += 1
                elif integrity_reason == "AVAILABLE_LEADING_FRAGMENT":
                    report.available_leading_fragment += 1
                elif integrity_reason == "EMBEDDED_SERIALIZED_CELL":
                    report.available_embedded_serialized_cell += 1
                return False
        elif classify_no_descriptor(record.descriptor_text or "Pas de descripteur disponible.") is None and record.status == "NO_DESCRIPTOR_AVAILABLE":
            report.available_no_descriptor_misclassified += 1
            return False
        report.levels_discovered.add(record.level_code)
        report.scales_discovered.add(record.scale_name)
        if is_valid_long_scale(record.scale_name):
            report.valid_long_scales_recovered += 1
        if record.reconstructed_from_fragments:
            report.leading_fragment_reconstructed += 1
        report.descriptors_by_level[record.level_code] += 1
        if record.status == "NO_DESCRIPTOR_AVAILABLE":
            report.no_descriptor_available += 1
            if record.reference_level:
                report.no_descriptor_with_reference += 1
            report.no_descriptor_examples.append(diagnostic)
        else:
            report.parsed_available += 1
            report.parsed_examples.append(diagnostic)
        return True

    def _collect_document(self, db: Session, *, document_id: int, dry_run: bool):
        document = db.get(KnowledgeDocument, document_id)
        if document is None:
            raise ValueError("Knowledge document was not found.")
        chunks = list(db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id).order_by(KnowledgeChunk.chunk_index)))
        report = CEFRImportReport(document_id, document.title, dry_run)
        parsed: list[tuple[KnowledgeChunk, ParsedCEFRDescriptor, list[KnowledgeChunk]]] = []
        for chunk, records, source_chunks in self._records(chunks, report):
            source = self._source_metadata(chunk, source_chunks)
            for record in records:
                if self._record_parsed(report, record, source):
                    parsed.append((chunk, record, source_chunks))
        return document, report, parsed

    def _persist_parsed(self, db: Session, *, document_id: int, report: CEFRImportReport, parsed: list[tuple[KnowledgeChunk, ParsedCEFRDescriptor, list[KnowledgeChunk]]]) -> None:
        levels = self.seed_levels(db)
        pending_source_links: dict[tuple[int, int], KnowledgeChunk] = {}
        for chunk, record, source_chunks in parsed:
            level = self._level(db, levels, record.level_code)
            scale = self._scale(db, record.scale_name)
            reference_level = self._level(db, levels, record.reference_level) if record.reference_level else None
            descriptor = db.scalar(select(CEFRDescriptor).where(
                CEFRDescriptor.level_id == level.id,
                CEFRDescriptor.scale_id == scale.id,
                CEFRDescriptor.descriptor_hash == record.descriptor_hash,
            ))
            if descriptor is None:
                descriptor = CEFRDescriptor(
                    level_id=level.id,
                    reference_level_id=reference_level.id if reference_level else None,
                    scale_id=scale.id,
                    descriptor_text=record.descriptor_text,
                    normalized_text=record.normalized_text,
                    descriptor_hash=record.descriptor_hash,
                    status=record.status,
                )
                db.add(descriptor)
                db.flush()
                report.persisted += 1
            else:
                report.duplicates += 1
                if descriptor.reference_level_id is None and reference_level is not None:
                    descriptor.reference_level_id = reference_level.id
            for source_chunk in source_chunks:
                if source_chunk.id not in record.source_chunk_ids:
                    continue
                report.source_links_seen += 1
                source_key = (descriptor.id, source_chunk.id)
                prior = pending_source_links.get(source_key)
                if prior is None:
                    pending_source_links[source_key] = source_chunk
                else:
                    report.source_links_deduplicated += 1
                    if source_chunk.chunk_index < prior.chunk_index:
                        pending_source_links[source_key] = source_chunk

        report.source_links_unique = len(pending_source_links)
        existing_links = set(db.execute(
            select(CEFRDescriptorSource.descriptor_id, CEFRDescriptorSource.chunk_id)
            .where(CEFRDescriptorSource.document_id == document_id)
        ).all())
        for (descriptor_id, chunk_id), source_chunk in sorted(
            pending_source_links.items(), key=lambda item: (item[1].chunk_index, item[0][0], item[0][1])
        ):
            if (descriptor_id, chunk_id) in existing_links:
                report.source_links_existing += 1
                continue
            db.add(CEFRDescriptorSource(
                descriptor_id=descriptor_id,
                document_id=document_id,
                chunk_id=chunk_id,
                page_start=source_chunk.source_page_start,
                page_end=source_chunk.source_page_end,
                source_order=source_chunk.chunk_index,
            ))
            report.source_links_inserted += 1
        db.flush()

    @staticmethod
    def _ensure_replacement_is_safe(report: CEFRImportReport) -> None:
        unsafe = (
            report.available_integrity_failures,
            report.available_embedded_serialized_cell,
            report.available_tail_contamination,
            report.available_leading_fragment,
            report.available_mid_sentence,
            report.descriptors_without_source,
        )
        if any(unsafe):
            raise ValueError("CEFR replacement refused because final AVAILABLE integrity checks failed.")

    def import_document(self, db: Session, *, document_id: int, dry_run: bool = True) -> CEFRImportReport:
        _, report, parsed = self._collect_document(db, document_id=document_id, dry_run=dry_run)

        if dry_run:
            return report

        with db.begin_nested():
            self._persist_parsed(db, document_id=document_id, report=report, parsed=parsed)
        return report

    def replace_document(self, db: Session, *, document_id: int, dry_run: bool = True) -> CEFRImportReport:
        """Atomically replace only this document's structured CEFR projection."""
        _, report, parsed = self._collect_document(db, document_id=document_id, dry_run=dry_run)
        self._ensure_replacement_is_safe(report)
        if dry_run:
            return report
        with db.begin_nested():
            affected_descriptor_ids = set(db.scalars(
                select(CEFRDescriptorSource.descriptor_id).where(CEFRDescriptorSource.document_id == document_id)
            ))
            affected_scale_ids = set(db.scalars(
                select(CEFRDescriptor.scale_id).where(CEFRDescriptor.id.in_(affected_descriptor_ids))
            )) if affected_descriptor_ids else set()
            for source in db.scalars(select(CEFRDescriptorSource).where(CEFRDescriptorSource.document_id == document_id)):
                db.delete(source)
            db.flush()
            for descriptor_id in affected_descriptor_ids:
                if db.scalar(select(CEFRDescriptorSource.id).where(CEFRDescriptorSource.descriptor_id == descriptor_id)) is None:
                    descriptor = db.get(CEFRDescriptor, descriptor_id)
                    if descriptor is not None:
                        db.delete(descriptor)
            db.flush()
            for scale_id in affected_scale_ids:
                if db.scalar(select(CEFRDescriptor.id).where(CEFRDescriptor.scale_id == scale_id)) is None:
                    scale = db.get(CEFRScale, scale_id)
                    if scale is not None:
                        db.delete(scale)
            db.flush()
            self._persist_parsed(db, document_id=document_id, report=report, parsed=parsed)
        return report
