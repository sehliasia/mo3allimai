"""Controlled, synchronous parsing preview for one knowledge document."""

from __future__ import annotations

import json
import hashlib
import logging
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument, KnowledgeDocumentStatus
from app.services.document_chunk import DocumentChunk
from app.services.document_chunker import DocumentChunker, DocumentChunkingError
from app.services.document_parser_service import DocumentParserService, DocumentParsingError, ModelResourceUnavailableError, PageExtractionIssue

logger = logging.getLogger(__name__)
DEBUG_CHUNKS_DIRECTORY = Path(__file__).resolve().parents[3] / "debug" / "chunks"


class KnowledgeIngestionError(RuntimeError):
    def __init__(self, message: str, *, result_summary: dict[str, Any] | None = None):
        super().__init__(message)
        self.result_summary = result_summary


class FailedPageRatioError(DocumentChunkingError):
    def __init__(self, result_summary: dict[str, Any]):
        super().__init__("Failed-page ratio exceeds the configured partial-ingestion threshold.")
        self.result_summary = result_summary


@dataclass(frozen=True)
class ParsePreview:
    document_id: int
    pages: int
    items: int
    chunks_count: int
    statistics: dict[str, int | float | list[int]]
    chunks: list[DocumentChunk]
    parsing_duration_ms: int
    chunking_duration_ms: int
    page_extractions: list[Any]
    quality_status: str
    warnings: list[dict[str, Any]]
    timings_ms: dict[str, float]
    cache_hit: bool
    ocr_strategy: str
    ocr_required_page_ratio: float
    cache_key: str | None
    cache_miss_reason: str | None
    cache_write_success: bool
    page_issues: list[PageExtractionIssue] = field(default_factory=list)

    def to_response(self, *, include_debug: bool = False) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pages": self.pages,
            "items": self.items,
            "chunks_count": self.chunks_count,
            "statistics": self.statistics,
            "parsing_duration_ms": self.parsing_duration_ms,
            "chunking_duration_ms": self.chunking_duration_ms,
            "quality_status": self.quality_status,
            "warnings": self.warnings,
            "timings_ms": self.timings_ms,
            "cache_hit": self.cache_hit,
            "ocr_strategy": self.ocr_strategy,
            "ocr_required_page_ratio": self.ocr_required_page_ratio,
            "cache_key": self.cache_key,
            "cache_miss_reason": self.cache_miss_reason,
            "cache_write_success": self.cache_write_success,
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section,
                    "headings": chunk.headings,
                    "content_type": chunk.content_type,
                    "token_count": chunk.token_count,
                    "text_preview": chunk.text_original[:700],
                    "image_ids": chunk.metadata.get("image_ids", []),
                    **({"text_for_embedding": chunk.text_for_embedding} if include_debug else {}),
                }
                for chunk in self.chunks
            ],
            "extraction": {
                "native_pages_count": sum(page.extraction_mode == "native" for page in self.page_extractions),
                "ocr_pages_count": sum(page.extraction_mode == "full_page_ocr" for page in self.page_extractions),
                "failed_pages_count": sum(issue.disposition == "quarantined" for issue in self.page_issues),
                "skipped_low_information_pages_count": sum(issue.disposition == "skipped_low_information" for issue in self.page_issues),
                "per_page_quality": [
                    {"page": page.page_number, "extraction_mode": page.extraction_mode, "quality_score": page.quality.quality_score, "languages": list(page.quality.languages_detected)}
                    for page in self.page_extractions
                ],
                "page_issues": [
                    {
                        "page": issue.page_number,
                        "disposition": issue.disposition,
                        "extraction_mode_attempted": issue.extraction_mode_attempted,
                        "native_quality_score": issue.native_quality.quality_score,
                        "ocr_quality_score": issue.ocr_quality.quality_score if issue.ocr_quality else None,
                        "failure_reasons": list(issue.failure_reasons),
                    }
                    for issue in self.page_issues
                ],
            },
            "images": [
                {"image_id": image.image_id, "page": image.page, "bbox": image.bbox, "caption": image.caption, "nearby_text": image.nearby_text, "image_role": image.image_role, "associated_chunk_ids": image.associated_chunk_ids}
                for page in self.page_extractions for image in page.images
            ],
        }


class KnowledgeIngestionService:
    """Runs Docling preview only. It never persists vectors or marks a document indexed."""

    def __init__(self, parser: DocumentParserService | None = None):
        self.parser = parser or DocumentParserService()
        self.settings = get_settings()

    def parse_preview(self, db: Session, document: KnowledgeDocument, *, export_debug: bool = False, force_reprocess: bool = False) -> ParsePreview:
        start = time.perf_counter()
        try:
            parsed = self.parser.parse_pdf(
                Path(document.file_path),
                document_id=document.id,
                force_reprocess=force_reprocess,
                strict_cache_write=export_debug,
            )
            parsed_at = time.perf_counter()
            page_extractions = list(parsed.page_extractions)
            chunker = self._new_chunker()
            chunks = chunker.chunk(
                document=parsed.document,
                document_id=document.id,
                title=document.title,
                original_filename=document.original_filename,
                source=document.source,
                extraction_mode=parsed.extraction_mode,
                page_extractions=page_extractions,
            )
            association_started = time.perf_counter()
            chunks, page_extractions = self._associate_images(chunks, page_extractions)
            image_association_ms = round((time.perf_counter() - association_started) * 1000, 2)
            parser_quarantined_pages = {
                issue.page_number for issue in parsed.page_issues if issue.disposition == "quarantined"
            }
            parser_skipped_pages = {
                issue.page_number for issue in parsed.page_issues if issue.disposition == "skipped_low_information"
            }
            warnings: list[dict[str, Any]] = [
                {
                    "page": issue.page_number,
                    "reason": "ocr_candidate_failed" if issue.disposition == "quarantined" else "blank_or_decorative_page",
                    "disposition": issue.disposition,
                    "extraction_mode_attempted": issue.extraction_mode_attempted,
                    "native_quality_score": issue.native_quality.quality_score,
                    "ocr_quality_score": issue.ocr_quality.quality_score if issue.ocr_quality else None,
                    "failure_reasons": list(issue.failure_reasons),
                }
                for issue in parsed.page_issues
            ]
            repaired_pages: set[int] = set()
            for corruption in list(chunker.last_corruptions):
                page_number = corruption["page"]
                page = next((candidate for candidate in page_extractions if candidate.page_number == page_number), None)
                if page is None or page.extraction_mode != "native" or page_number in repaired_pages:
                    continue
                repaired_pages.add(page_number)
                logger.warning("late_page_repair_started document_id=%s page=%s reason=final_chunk_corruption", document.id, page_number)
                repaired = self.parser.repair_page_with_ocr(
                    Path(document.file_path), document_id=document.id, page_number=page_number, native_quality=page.quality,
                    native_corruption_reasons=tuple(corruption.get("failure_reasons", [])),
                )
                if repaired is None:
                    warnings.append({
                        "page": page_number,
                        "reason": "unrecoverable_extraction_corruption",
                        "native_quality_score": page.quality.quality_score,
                        "ocr_quality_score": None,
                    })
                    continue
                page_extractions[page_extractions.index(page)] = repaired
                logger.info("late_page_repair_completed document_id=%s page=%s mode=full_page_ocr quality_score=%.2f", document.id, page_number, repaired.quality.quality_score)

            if repaired_pages - {warning["page"] for warning in warnings}:
                # Regenerate IDs and replace every native chunk of successfully repaired pages.
                chunker = self._new_chunker()
                chunks = chunker.chunk(
                    document=parsed.document, document_id=document.id, title=document.title,
                    original_filename=document.original_filename, source=document.source,
                    extraction_mode=parsed.extraction_mode, page_extractions=page_extractions,
                )
                association_started = time.perf_counter()
                chunks, page_extractions = self._associate_images(chunks, page_extractions)
                image_association_ms = round((time.perf_counter() - association_started) * 1000, 2)
            quarantined_pages = parser_quarantined_pages | {
                event["page"] for event in chunker.last_corruptions if event["page"] is not None
            }
            for event in chunker.last_corruptions:
                if not any(warning["page"] == event["page"] for warning in warnings):
                    warnings.append({
                        "page": event["page"],
                        "reason": "unrecoverable_extraction_corruption",
                        "quality_score": event["quality_score"],
                        "failure_reasons": event["failure_reasons"],
                    })
            failed_ratio = len(quarantined_pages) / max(1, parsed.pages_count)
            if len(chunks) < self.settings.rag_min_valid_chunks:
                raise DocumentChunkingError("Document has no minimum usable semantic content.")
            if failed_ratio > self.settings.rag_max_failed_page_ratio or (quarantined_pages and not self.settings.rag_allow_partial_ingestion):
                raise FailedPageRatioError({
                    "quality_status": "failed",
                    "pages_total": parsed.pages_count,
                    "pages_quarantined_count": len(quarantined_pages),
                    "quarantined_page_numbers": sorted(quarantined_pages),
                    "failed_page_ratio": round(failed_ratio, 6),
                    "max_failed_page_ratio": self.settings.rag_max_failed_page_ratio,
                    "failure_reason": "failed_page_ratio_exceeded",
                    "chunks_valid": len(chunks),
                    "chunks_quarantined_count": chunker.last_metrics["chunks_quarantined"],
                    "ocr_failures_count": len(parser_quarantined_pages),
                    "warnings_count": len(warnings),
                })
            quality_status = "partial" if quarantined_pages else "complete"
            chunked_at = time.perf_counter()
            token_counts = [chunk.token_count for chunk in chunks]
            late_repair_rejected = len({warning["page"] for warning in warnings if warning["page"] in repaired_pages})
            stats: dict[str, int | float | list[int]] = {
                "text_chunks_count": sum(chunk.content_type == "text" for chunk in chunks),
                "table_chunks_count": sum(chunk.content_type == "table" for chunk in chunks),
                "min_tokens": min(token_counts),
                "max_tokens": max(token_counts),
                "average_tokens": round(statistics.fmean(token_counts), 2),
                "median_tokens": round(statistics.median(token_counts), 2),
                "chunks_kept": chunker.last_metrics["chunks_kept"],
                "chunks_skipped_low_information": chunker.last_metrics["chunks_skipped_low_information"],
                "chunks_failed_quality": chunker.last_metrics["chunks_failed_quality"],
                "chunks_quarantined": chunker.last_metrics["chunks_quarantined"],
                "pages_total": parsed.pages_count,
                "pages_native": sum(page.extraction_mode == "native" for page in page_extractions),
                "pages_ocr": sum(page.extraction_mode == "full_page_ocr" for page in page_extractions),
                "pages_late_repaired": len(repaired_pages - {warning["page"] for warning in warnings}),
                "pages_quarantined": len(quarantined_pages),
                "quarantined_page_numbers": sorted(quarantined_pages),
                "pages_skipped_low_information": len(parser_skipped_pages),
                "skipped_low_information_page_numbers": sorted(parser_skipped_pages),
                "ocr_failures": len(parser_quarantined_pages),
                "late_repairs_attempted": len(repaired_pages),
                "late_repairs_accepted": len(repaired_pages - {warning["page"] for warning in warnings}),
                "late_repairs_rejected": late_repair_rejected,
                "warnings_count": len(warnings),
            }
            preview = ParsePreview(
                document_id=document.id,
                pages=parsed.pages_count,
                items=parsed.items_count,
                chunks_count=len(chunks),
                statistics=stats,
                chunks=self._sample_chunks(chunks),
                parsing_duration_ms=round((parsed_at - start) * 1000),
                chunking_duration_ms=round((chunked_at - parsed_at) * 1000),
                page_extractions=page_extractions,
                quality_status=quality_status,
                warnings=warnings,
                timings_ms={**parsed.timings_ms, "image_association": image_association_ms},
                cache_hit=parsed.cache_hit,
                ocr_strategy=parsed.ocr_strategy,
                ocr_required_page_ratio=parsed.ocr_required_page_ratio,
                cache_key=parsed.cache_key,
                cache_miss_reason=parsed.cache_miss_reason,
                cache_write_success=parsed.cache_write_success,
                page_issues=parsed.page_issues,
            )
            if export_debug or self.settings.rag_debug_export_chunks:
                self._export_debug(document.id, chunks, page_extractions, parsed.page_issues)
            # This is deliberately the final side effect after all extraction,
            # validation, association and optional diagnostics have completed.
            self._replace_persisted_chunks(db, document, chunks, quality_status=quality_status)
            # Preparation does not imply retrieval readiness: keep the public status pending.
            document.status = KnowledgeDocumentStatus.pending
            db.commit()
            logger.info(
                "knowledge_preview_completed document_id=%s filename=%s pages=%s items=%s chunks=%s tables=%s parsing_ms=%s chunking_ms=%s min_tokens=%s max_tokens=%s average_tokens=%s",
                document.id, document.original_filename, preview.pages, preview.items, preview.chunks_count,
                stats["table_chunks_count"], preview.parsing_duration_ms, preview.chunking_duration_ms,
                stats["min_tokens"], stats["max_tokens"], stats["average_tokens"],
            )
            return preview
        except ModelResourceUnavailableError as error:
            document.status = KnowledgeDocumentStatus.failed
            db.commit()
            logger.exception("knowledge_preview_model_resource_unavailable document_id=%s filename=%s", document.id, document.original_filename)
            raise KnowledgeIngestionError("Model resource unavailable. Please retry when the Docling model resource is available.") from error
        except (DocumentParsingError, DocumentChunkingError, OSError, ValueError) as error:
            document.status = KnowledgeDocumentStatus.failed
            db.commit()
            logger.exception("knowledge_preview_failed document_id=%s filename=%s error_type=%s", document.id, document.original_filename, type(error).__name__)
            raise KnowledgeIngestionError(
                "Document parsing could not be completed.",
                result_summary=getattr(error, "result_summary", None),
            ) from error

    def _new_chunker(self) -> DocumentChunker:
        return DocumentChunker(
            max_tokens=self.settings.rag_chunk_max_tokens,
            tokenizer_name=self.settings.rag_chunk_tokenizer,
        )

    def _replace_persisted_chunks(
        self,
        db: Session,
        document: KnowledgeDocument,
        chunks: list[DocumentChunk],
        *,
        quality_status: str,
    ) -> None:
        """Atomically swap a complete, validated generation into the relational source of truth."""
        # Lightweight parser unit tests intentionally use a commit-only fake DB.
        # Production calls always use a SQLAlchemy Session and materialize every final chunk.
        if not hasattr(db, "execute") or not hasattr(db, "add_all"):
            return

        ingestion_version = self.settings.rag_extraction_pipeline_version
        rows = [
            KnowledgeChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                content=chunk.text_original,
                content_for_embedding=chunk.text_for_embedding,
                token_count=chunk.token_count,
                content_type=chunk.content_type,
                source_page_start=chunk.page_start,
                source_page_end=chunk.page_end,
                extraction_mode=chunk.metadata.get("extraction_mode"),
                quality_status=quality_status,
                heading_context=list(chunk.headings),
                chunk_metadata=self._safe_chunk_metadata(chunk.metadata),
                chunk_hash=self._chunk_hash(chunk, ingestion_version),
                ingestion_version=ingestion_version,
            )
            for chunk in chunks
        ]
        # All parsing, repair, chunking and quality decisions completed above. The
        # savepoint means a persistence error never commits a partially replaced set.
        with db.begin_nested():
            db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id))
            db.add_all(rows)
            db.flush()
        logger.info(
            "knowledge_chunks_persisted document_id=%s chunks=%s ingestion_version=%s",
            document.id,
            len(rows),
            ingestion_version,
        )

    @staticmethod
    def _safe_chunk_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Persist retrieval-relevant structure only; never leak debug filesystem data."""
        allowed_keys = {
            "document_id", "original_filename", "source", "page_start", "page_end",
            "page_number", "headings", "content_type", "language", "document_type",
            "level", "skill", "theme", "extraction_mode", "extraction_modes_used",
            "quality_score", "languages", "image_ids",
            "has_image", "requires_vision", "structural_quality",
        }
        return {key: value for key, value in metadata.items() if key in allowed_keys}

    @staticmethod
    def _chunk_hash(chunk: DocumentChunk, ingestion_version: str) -> str:
        normalized_content = " ".join(chunk.text_original.split())
        semantic_identity = {
            "content": normalized_content,
            "pages": [chunk.page_start, chunk.page_end],
            "content_type": chunk.content_type,
            "headings": chunk.headings,
            "ingestion_version": ingestion_version,
        }
        payload = json.dumps(semantic_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _associate_images(chunks: list[DocumentChunk], pages: list[Any]) -> tuple[list[DocumentChunk], list[Any]]:
        """Associate only a confident same-page pedagogical chunk, never every page chunk."""
        image_to_chunks: dict[str, list[str]] = {}
        for page in pages:
            page_chunks = [chunk for chunk in chunks if chunk.page_start == page.page_number or chunk.page_end == page.page_number]
            for image in page.images:
                context = image.caption or image.nearby_text or ""
                context_terms = {term for term in context.lower().split() if len(term) >= 2}
                matches = [chunk for chunk in page_chunks if context_terms and len(context_terms.intersection({term for term in chunk.text_original.lower().split() if len(term) >= 2})) >= 2]
                if len(matches) == 1:
                    image_to_chunks[image.image_id] = [matches[0].id]
                elif len(page_chunks) == 1:
                    image_to_chunks[image.image_id] = [page_chunks[0].id]
        updated_chunks = []
        for chunk in chunks:
            image_ids = [image_id for image_id, chunk_ids in image_to_chunks.items() if chunk.id in chunk_ids]
            updated_chunks.append(replace(chunk, metadata={**chunk.metadata, "image_ids": image_ids}))
        updated_pages = []
        for page in pages:
            images = [replace(image, associated_chunk_ids=image_to_chunks.get(image.image_id, [])) for image in page.images]
            updated_pages.append(replace(page, images=images))
        return updated_chunks, updated_pages

    @staticmethod
    def _sample_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        indexes = set(range(min(5, len(chunks))))
        middle = len(chunks) // 2
        indexes.update(range(max(0, middle - 2), min(len(chunks), middle + 3)))
        indexes.update(range(max(0, len(chunks) - 5), len(chunks)))
        indexes.update(chunk.chunk_index for chunk in chunks if chunk.content_type == "table")
        return [chunk for chunk in chunks if chunk.chunk_index in indexes][:20]

    @staticmethod
    def _export_debug(document_id: int, chunks: list[DocumentChunk], page_extractions: list[Any], page_issues: list[PageExtractionIssue]) -> None:
        DEBUG_CHUNKS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output = DEBUG_CHUNKS_DIRECTORY / f"{document_id}.json"
        payload = {
            "document_id": document_id,
            "pages_count": len(page_extractions),
            "native_pages_count": sum(page.extraction_mode == "native" for page in page_extractions),
            "ocr_pages_count": sum(page.extraction_mode == "full_page_ocr" for page in page_extractions),
            "failed_pages_count": sum(issue.disposition == "quarantined" for issue in page_issues),
            "page_issues": [
                {
                    "page": issue.page_number,
                    "disposition": issue.disposition,
                    "extraction_mode_attempted": issue.extraction_mode_attempted,
                    "native_quality_score": issue.native_quality.quality_score,
                    "ocr_quality_score": issue.ocr_quality.quality_score if issue.ocr_quality else None,
                    "failure_reasons": list(issue.failure_reasons),
                }
                for issue in page_issues
            ],
            "detected_languages": sorted({language for page in page_extractions for language in page.quality.languages_detected}),
            "per_page_quality": [
                {"page": page.page_number, "extraction_mode": page.extraction_mode, "quality_score": page.quality.quality_score, "languages": list(page.quality.languages_detected)}
                for page in page_extractions
            ],
            "chunks": [chunk.to_dict() for chunk in chunks],
            "images": [
                {
                    "image_id": image.image_id, "page": image.page, "bbox": image.bbox,
                    "path": image.path, "caption": image.caption, "nearby_text": image.nearby_text,
                    "image_role": image.image_role, "associated_chunk_ids": image.associated_chunk_ids,
                }
                for page in page_extractions for image in page.images
            ],
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
