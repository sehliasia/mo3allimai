"""Bounded, native-only preflight analysis for Knowledge Base PDFs."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.knowledge_document import KnowledgeDocument
from app.services.document_parser_service import DocumentParserService, DocumentParsingError, NativePreflight

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreflightReport:
    document_id: int
    pages_total: int
    native_good_page_numbers: list[int]
    native_borderline_page_numbers: list[int]
    native_bad_page_numbers: list[int]
    analysis_failed_page_numbers: list[int]
    ocr_candidate_pages: list[int]
    ocr_required_page_ratio: float
    recommended_strategy: str | None
    estimated_complexity: str | None
    native_analysis_duration_ms: int
    preflight_status: str
    preflight_cache_hit: bool
    detected_picture_count: int | None = None
    detected_table_count: int | None = None
    failure_debug: dict[str, Any] = field(default_factory=dict)
    lifecycle_diagnostics: dict[str, Any] = field(default_factory=dict)
    page_decisions: list[dict[str, Any]] = field(default_factory=list)

    def to_response(self, *, export_debug: bool = False) -> dict[str, Any]:
        response: dict[str, Any] = {
            "document_id": self.document_id, "pages_total": self.pages_total,
            "pages_analyzed": len(self.native_good_page_numbers) + len(self.native_borderline_page_numbers) + len(self.native_bad_page_numbers),
            "analysis_failed_pages": len(self.analysis_failed_page_numbers), "analysis_failed_page_numbers": self.analysis_failed_page_numbers,
            "native_good_pages": len(self.native_good_page_numbers), "native_borderline_pages": len(self.native_borderline_page_numbers), "native_bad_pages": len(self.native_bad_page_numbers),
            "native_good_page_numbers": self.native_good_page_numbers, "native_borderline_page_numbers": self.native_borderline_page_numbers, "native_bad_page_numbers": self.native_bad_page_numbers,
            "ocr_candidate_pages": self.ocr_candidate_pages, "ocr_required_page_ratio": self.ocr_required_page_ratio,
            "recommended_strategy": self.recommended_strategy, "estimated_complexity": self.estimated_complexity,
            "native_analysis_duration_ms": self.native_analysis_duration_ms, "preflight_status": self.preflight_status,
            "preflight_cache_hit": self.preflight_cache_hit,
        }
        if self.detected_picture_count is not None: response["detected_picture_count"] = self.detected_picture_count
        if self.detected_table_count is not None: response["detected_table_count"] = self.detected_table_count
        if export_debug:
            response["debug"] = {"analysis_only": True, "ocr_executed": False, "chunking_executed": False, "page_decisions": self.page_decisions, **self.lifecycle_diagnostics, **self.failure_debug}
        return response


class KnowledgePreflightService:
    """Batch native conversion only. Technical failures are never OCR candidates."""

    def __init__(self, parser: DocumentParserService | None = None):
        self.parser = parser or DocumentParserService()
        self.settings = get_settings()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""): digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _page_count(path: Path) -> int:
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(str(path))
        try: return len(document)
        finally: document.close()

    def _strategy(self, ratio: float) -> str:
        if ratio <= self.settings.rag_preflight_native_only_max_bad_ratio: return "native_only"
        if ratio >= self.settings.rag_preflight_ocr_heavy_ratio: return "ocr_heavy"
        return "native_with_targeted_ocr"

    @staticmethod
    def _complexity(pages_total: int, ocr_ratio: float) -> str:
        if pages_total >= 150 or ocr_ratio >= 0.60: return "high"
        if pages_total >= 40 or ocr_ratio > 0.02: return "medium"
        return "low"

    def _ranges(self, pages_total: int) -> list[tuple[int, int]]:
        size = max(1, self.settings.rag_preflight_batch_size)
        return [(start, min(pages_total, start + size - 1)) for start in range(1, pages_total + 1, size)]

    def _analyze_range(self, path: Path, page_range: tuple[int, int], *, depth: int = 0) -> tuple[list[NativePreflight], list[int], list[dict[str, str]]]:
        try:
            return [self.parser.preflight_pdf(path, page_range=page_range)], [], []
        except DocumentParsingError as error:
            start, end = page_range
            length = end - start + 1
            if length > max(1, self.settings.rag_preflight_min_batch_size):
                midpoint = start + (length // 2) - 1
                left = self._analyze_range(path, (start, midpoint), depth=depth + 1)
                right = self._analyze_range(path, (midpoint + 1, end), depth=depth + 1)
                return left[0] + right[0], left[1] + right[1], left[2] + right[2]
            cause = error.__cause__ or error
            diagnostic = {"failure_stage": "native_docling_conversion", "exception_type": type(cause).__name__, "sanitized_message": str(cause)[:500] or "Native Docling analysis failed.", "document_converter_started": "true", "docling_document_returned": "false", "page_range": f"{start}-{end}", "retry_depth": str(depth)}
            logger.exception("knowledge_preflight_batch_failed page_range=%s-%s diagnostic=%s", start, end, diagnostic)
            return [], list(range(start, end + 1)), [diagnostic]

    def _from_stored(self, document: KnowledgeDocument) -> PreflightReport:
        details = document.preflight_page_details or {}
        return PreflightReport(document.id, document.preflight_pages_total or 0, list(details.get("native_good_page_numbers", [])), list(details.get("native_borderline_page_numbers", [])), list(details.get("native_bad_page_numbers", [])), list(details.get("analysis_failed_page_numbers", [])), list(details.get("ocr_candidate_pages", [])), document.preflight_ocr_required_page_ratio or 0.0, document.preflight_recommended_strategy, document.preflight_estimated_complexity, 0, document.preflight_status or "failed", True, details.get("detected_picture_count"), details.get("detected_table_count"), page_decisions=list(details.get("page_decisions", [])))

    def analyze(self, db: Session, document: KnowledgeDocument) -> PreflightReport:
        source = Path(document.file_path)
        try:
            source_exists = source.is_file()
            source_sha256 = self._file_sha256(source)
            pages_total = self._page_count(source)
        except OSError:
            return self._persist_failure(db, document, 0, {"failure_stage": "source_file", "exception_type": "OSError", "sanitized_message": "The source PDF is unavailable.", "source_file_exists": False, "page_count_helper_called": False, "pdfium_page_count": None, "docling_started": False, "docling_document_returned": False})
        except Exception as error:
            return self._persist_failure(db, document, 0, {"failure_stage": "pdf_page_count", "exception_type": type(error).__name__, "sanitized_message": str(error)[:500] or "The PDF page count could not be read.", "source_file_exists": source_exists, "page_count_helper_called": True, "pdfium_page_count": None, "docling_started": False, "docling_document_returned": False})
        if document.preflight_status in {"complete", "partial"} and document.preflight_source_sha256 == source_sha256 and document.preflight_analysis_version == self.settings.rag_preflight_analysis_version:
            return self._from_stored(document)
        batches: list[NativePreflight] = []; failed_pages: list[int] = []; diagnostics: list[dict[str, str]] = []
        session = getattr(self.parser, "native_preflight_session", None)
        with (session() if callable(session) else nullcontext({"document_converter_instances_created": 0, "conversion_calls": 0, "page_ranges_processed": []})) as lifecycle:
            for page_range in self._ranges(pages_total):
                successful, failed, failures = self._analyze_range(source, page_range)
                batches.extend(successful); failed_pages.extend(failed); diagnostics.extend(failures)
        good = sorted(page for batch in batches for page in batch.native_good_page_numbers)
        borderline = sorted(page for batch in batches for page in batch.native_borderline_page_numbers)
        bad = sorted(page for batch in batches for page in batch.native_bad_page_numbers)
        failed_pages = sorted(set(failed_pages)); pages_analyzed = len(good) + len(borderline) + len(bad)
        page_decisions = [decision for batch in batches for decision in batch.page_decisions]
        candidates = sorted(decision["page"] for decision in page_decisions if decision["ocr_candidate"]) if page_decisions else borderline + bad
        ratio = len(candidates) / max(1, pages_analyzed)
        status = "complete" if not failed_pages else "partial" if pages_analyzed else "failed"
        logger.info("knowledge_preflight_lifecycle document_id=%s diagnostics=%s", document.id, lifecycle)
        report = PreflightReport(document.id, pages_total, good, borderline, bad, failed_pages, candidates, ratio, self._strategy(ratio) if pages_analyzed else None, self._complexity(pages_total, ratio) if pages_analyzed else None, sum(batch.native_analysis_duration_ms for batch in batches), status, False, sum((batch.detected_picture_count or 0) for batch in batches), sum((batch.detected_table_count or 0) for batch in batches), {"failures": diagnostics}, dict(lifecycle), page_decisions)
        self._persist_result(db, document, source_sha256, report)
        return report

    def _persist_result(self, db: Session, document: KnowledgeDocument, source_sha256: str, report: PreflightReport) -> None:
        pages_analyzed = len(report.native_good_page_numbers) + len(report.native_borderline_page_numbers) + len(report.native_bad_page_numbers)
        document.preflight_status = report.preflight_status; document.preflight_analyzed_at = datetime.now(timezone.utc)
        document.preflight_source_sha256 = source_sha256 if report.preflight_status != "failed" else None
        document.preflight_analysis_version = self.settings.rag_preflight_analysis_version if report.preflight_status != "failed" else None
        document.preflight_pages_total = report.pages_total; document.preflight_pages_analyzed = pages_analyzed; document.preflight_analysis_failed_pages = len(report.analysis_failed_page_numbers)
        document.preflight_native_good_pages = len(report.native_good_page_numbers); document.preflight_native_borderline_pages = len(report.native_borderline_page_numbers); document.preflight_native_bad_pages = len(report.native_bad_page_numbers)
        document.preflight_ocr_candidate_page_count = len(report.ocr_candidate_pages); document.preflight_ocr_required_page_ratio = report.ocr_required_page_ratio
        document.preflight_recommended_strategy = report.recommended_strategy; document.preflight_estimated_complexity = report.estimated_complexity
        document.preflight_page_details = {"native_good_page_numbers": report.native_good_page_numbers, "native_borderline_page_numbers": report.native_borderline_page_numbers, "native_bad_page_numbers": report.native_bad_page_numbers, "analysis_failed_page_numbers": report.analysis_failed_page_numbers, "ocr_candidate_pages": report.ocr_candidate_pages, "detected_picture_count": report.detected_picture_count, "detected_table_count": report.detected_table_count, "page_decisions": report.page_decisions}
        db.commit()

    def _persist_failure(self, db: Session, document: KnowledgeDocument, pages_total: int, diagnostic: dict[str, Any]) -> PreflightReport:
        logger.warning("knowledge_preflight_failed document_id=%s diagnostic=%s", document.id, diagnostic)
        report = PreflightReport(document.id, pages_total, [], [], [], [], [], 0.0, None, None, 0, "failed", False, failure_debug=diagnostic)
        self._persist_result(db, document, "", report)
        return report
