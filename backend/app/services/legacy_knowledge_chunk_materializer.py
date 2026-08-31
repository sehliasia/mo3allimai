"""Explicit, offline-only materialization of selected historical Knowledge Base chunks."""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.knowledge_document import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeProcessingJob,
    KnowledgeProcessingJobStatus,
    KnowledgeProcessingJobType,
)
from app.services.document_chunk import DocumentChunk
from app.services.document_chunker import DocumentChunker
from app.services.document_parser_service import DocumentParserService
from app.services.knowledge_ingestion_service import DEBUG_CHUNKS_DIRECTORY, KnowledgeIngestionService

logger = logging.getLogger(__name__)


class LegacyMaterializationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LegacyChunkAudit:
    document_id: int
    filename: str
    status: str
    persisted_chunk_count: int
    legacy_debug_available: bool
    extraction_cache_available: bool
    safest_materialization_method: str


@dataclass(frozen=True)
class LegacyMaterializationReport:
    document_id: int
    source_used: str
    chunks_discovered: int
    chunks_validated: int
    chunks_persisted: int
    chunks_rejected: int
    pages_from_cache: int = 0
    ocr_invoked: bool = False
    docling_invoked: bool = False
    old_persisted_chunks: int = 0
    new_persisted_chunks: int = 0
    old_ingestion_version: str | None = None
    new_ingestion_version: str | None = None
    rejected_pages: list[int] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    partial_materialization: bool = False
    compatibility_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LegacyKnowledgeChunkMaterializer:
    """Never calls parse_pdf: it accepts only an existing DB/debug/cache artifact."""

    def __init__(
        self,
        *,
        parser: DocumentParserService | None = None,
        chunker_factory: Callable[[], DocumentChunker] | None = None,
    ):
        self.settings = get_settings()
        self.parser = parser or DocumentParserService()
        self._chunker_factory = chunker_factory or (lambda: DocumentChunker(
            max_tokens=self.settings.rag_chunk_max_tokens,
            tokenizer_name=self.settings.rag_chunk_tokenizer,
            local_files_only=True,
        ))

    def audit(self, db: Session) -> list[LegacyChunkAudit]:
        documents = db.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.id)).all()
        return [self._audit_document(db, document) for document in documents]

    def materialize(
        self,
        db: Session,
        *,
        document_id: int,
        rebuild_from_cache: bool = False,
    ) -> LegacyMaterializationReport:
        document = db.get(KnowledgeDocument, document_id)
        if document is None:
            raise LegacyMaterializationError("Knowledge document not found.")
        persisted_count = self._persisted_count(db, document.id)
        old_ingestion_version = self._persisted_ingestion_version(db, document.id)
        if persisted_count and not rebuild_from_cache:
            return LegacyMaterializationReport(document.id, "already_persisted", persisted_count, persisted_count, 0, 0)

        artifact = DEBUG_CHUNKS_DIRECTORY / f"{document.id}.json"
        pages_from_cache = 0
        chunks_rejected = 0
        chunks_discovered = 0
        rejected: list[object] = []
        compatibility_source: str | None = None
        if artifact.is_file() and not rebuild_from_cache:
            chunks = self._load_debug_chunks(document.id, artifact)
            chunks_discovered = len(chunks)
            source_used = "legacy_debug"
        else:
            source_used = "extraction_cache_rebuild" if rebuild_from_cache else "extraction_cache"
            logger.info("materialization_started document_id=%s source=%s", document.id, source_used)
            parsed = (
                self.parser.load_cached_extraction_for_rebuild(document_id=document.id)
                if rebuild_from_cache
                else self.parser.load_cached_extraction_only(Path(document.file_path), document_id=document.id)
            )
            if parsed is None:
                raise LegacyMaterializationError("No matching extraction cache is available for cache-only materialization.")
            pages_from_cache = len(getattr(parsed, "page_extractions", []))
            logger.info("cached_pages_loaded document_id=%s pages=%s", document.id, pages_from_cache)
            chunks, rejected, compatibility_source = self._chunks_from_cache(
                db, document, parsed, allow_valid_count_change=rebuild_from_cache
            )
            chunks_rejected = len(rejected)
            chunks_discovered = len(chunks) + chunks_rejected

        logger.info(
            "chunks_generated document_id=%s discovered=%s valid=%s rejected=%s",
            document.id,
            chunks_discovered,
            len(chunks),
            chunks_rejected,
        )
        self._validate_chunks(document.id, chunks)
        logger.info("chunks_validated document_id=%s count=%s", document.id, len(chunks))
        quality_status = "legacy_imported"
        if source_used.startswith("extraction_cache"):
            quality_status = "partial" if getattr(parsed, "page_issues", ()) or rejected else "complete"
        KnowledgeIngestionService()._replace_persisted_chunks(db, document, chunks, quality_status=quality_status)
        logger.info("chunks_persisted document_id=%s count=%s", document.id, len(chunks))
        logger.info("materialization_completed document_id=%s source=%s", document.id, source_used)
        return LegacyMaterializationReport(
            document.id,
            source_used,
            chunks_discovered,
            len(chunks),
            len(chunks),
            chunks_rejected,
            pages_from_cache,
            old_persisted_chunks=persisted_count,
            new_persisted_chunks=len(chunks),
            old_ingestion_version=old_ingestion_version,
            new_ingestion_version=self.settings.rag_extraction_pipeline_version,
            rejected_pages=sorted({item.get("page") for item in rejected if isinstance(item, dict) and isinstance(item.get("page"), int)}),
            rejection_reasons=sorted({reason for item in rejected if isinstance(item, dict) for reason in item.get("failure_reasons", []) if isinstance(reason, str)}),
            partial_materialization=bool(rejected),
            compatibility_source=compatibility_source,
        )

    def _audit_document(self, db: Session, document: KnowledgeDocument) -> LegacyChunkAudit:
        count = self._persisted_count(db, document.id)
        debug_available = (DEBUG_CHUNKS_DIRECTORY / f"{document.id}.json").is_file()
        cache_available = any((directory / "manifest.json").is_file() for directory in self._cache_directories(document.id))
        method = "already_persisted" if count else "legacy_debug" if debug_available else "extraction_cache" if cache_available else "unavailable_without_reprocessing"
        status = document.status.value if hasattr(document.status, "value") else str(document.status)
        return LegacyChunkAudit(document.id, document.original_filename, status, count, debug_available, cache_available, method)

    @staticmethod
    def _persisted_count(db: Session, document_id: int) -> int:
        return int(db.scalar(select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id)) or 0)

    @staticmethod
    def _persisted_ingestion_version(db: Session, document_id: int) -> str | None:
        return db.scalar(
            select(KnowledgeChunk.ingestion_version)
            .where(KnowledgeChunk.document_id == document_id)
            .order_by(KnowledgeChunk.id.desc())
            .limit(1)
        )

    @staticmethod
    def _cache_directories(document_id: int) -> tuple[Path, Path]:
        return (DocumentParserService._cache_directory(document_id), DocumentParserService._legacy_cache_directory(document_id))

    def _chunks_from_cache(
        self,
        db: Session,
        document: KnowledgeDocument,
        parsed: object,
        *,
        allow_valid_count_change: bool = False,
    ) -> tuple[list[DocumentChunk], list[object], str | None]:
        # This intentionally uses cached DoclingDocument JSON only. No converter,
        # PDF reader, EasyOCR, model download, or network fallback is available.
        chunker = self._chunker_factory()
        chunks = chunker.chunk(
            document=getattr(parsed, "document"), document_id=document.id, title=document.title,
            original_filename=document.original_filename, source=document.source,
            extraction_mode=getattr(parsed, "extraction_mode", "page_adaptive"),
            page_extractions=list(getattr(parsed, "page_extractions")),
        )
        rejected = list(getattr(chunker, "last_corruptions", []))
        compatibility_source = self._validate_cache_rejections(
            db, document, chunks, rejected,
            allow_valid_count_change=allow_valid_count_change,
            pages_total=int(getattr(parsed, "pages_count", 0) or 0),
        )
        chunks, _ = KnowledgeIngestionService._associate_images(chunks, list(getattr(parsed, "page_extractions")))
        return chunks, rejected, compatibility_source

    @staticmethod
    def _rejection_text(rejection: object) -> str:
        if not isinstance(rejection, dict):
            return ""
        preview = rejection.get("text_preview", "")
        if not isinstance(preview, str):
            return ""
        # The chunker records repr(fragment[:500]) for safe logging. Parse this
        # Python string literal so escaped newlines are not mistaken for letter
        # "n" content during the Unicode-aware placeholder check below.
        try:
            value = ast.literal_eval(preview)
        except (SyntaxError, ValueError):
            value = None
        if isinstance(value, str):
            return value
        return preview

    @classmethod
    def _is_image_placeholder_only(cls, rejection: object) -> bool:
        text = cls._rejection_text(rejection)
        if not re.search(r"<!--\s*image\s*-->", text, flags=re.IGNORECASE):
            return False
        without_markers = re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)
        # A page/item number, punctuation, and whitespace add no embeddable
        # semantic content. Any Unicode letter — Arabic, Latin, or otherwise —
        # makes this a real textual rejection that must remain unsafe.
        if any(character.isalpha() for character in without_markers):
            return False
        numeric_tokens = re.findall(r"\d+", without_markers)
        return len(numeric_tokens) <= 4 and all(len(token) <= 3 for token in numeric_tokens)

    def _validate_cache_rejections(
        self,
        db: Session,
        document: KnowledgeDocument,
        chunks: list[DocumentChunk],
        rejected: list[object],
        *,
        allow_valid_count_change: bool,
        pages_total: int,
    ) -> str | None:
        if not rejected:
            return None

        summary = self._last_successful_ingestion_summary(db, document.id)
        if not self._summary_matches_cache(
            summary,
            valid_count=len(chunks),
            rejected_count=len(rejected),
            allow_valid_count_change=allow_valid_count_change,
        ):
            if not self._within_existing_partial_policy(rejected, pages_total):
                raise LegacyMaterializationError(
                    "Cached extraction rejected chunks exceed the successful-ingestion compatibility or partial-ingestion policy."
                )
            compatibility_source = "existing_partial_threshold"
        else:
            compatibility_source = "previous_ingestion_summary"

        for rejection in rejected:
            details = rejection if isinstance(rejection, dict) else {}
            logger.info(
                "materialization_chunk_discarded document_id=%s page=%s reason=%s placeholder_only=%s",
                document.id,
                details.get("page"),
                ",".join(details.get("failure_reasons", [])) if isinstance(details.get("failure_reasons"), list) else "rejected_quality",
                self._is_image_placeholder_only(rejection),
            )
        return compatibility_source

    def _within_existing_partial_policy(self, rejected: list[object], pages_total: int) -> bool:
        pages = {item.get("page") for item in rejected if isinstance(item, dict) and isinstance(item.get("page"), int)}
        return bool(
            self.settings.rag_allow_partial_ingestion
            and pages_total > 0
            and len(pages) == len({item.get("page") for item in rejected if isinstance(item, dict)})
            and len(pages) / pages_total <= self.settings.rag_max_failed_page_ratio
        )

    @staticmethod
    def _last_successful_ingestion_summary(db: Session, document_id: int) -> dict[str, object] | None:
        job = db.scalar(
            select(KnowledgeProcessingJob)
            .where(
                KnowledgeProcessingJob.document_id == document_id,
                KnowledgeProcessingJob.job_type == KnowledgeProcessingJobType.ingestion,
                KnowledgeProcessingJob.status == KnowledgeProcessingJobStatus.completed,
            )
            .order_by(KnowledgeProcessingJob.completed_at.desc(), KnowledgeProcessingJob.id.desc())
            .limit(1)
        )
        return job.result_summary if job and isinstance(job.result_summary, dict) else None

    @staticmethod
    def _summary_matches_cache(
        summary: dict[str, object] | None,
        *,
        valid_count: int,
        rejected_count: int,
        allow_valid_count_change: bool,
    ) -> bool:
        if summary is None:
            return False
        expected_valid = summary.get("chunks_valid")
        expected_rejected = summary.get("chunks_quarantined_count")
        return (
            isinstance(expected_valid, int)
            and not isinstance(expected_valid, bool)
            and isinstance(expected_rejected, int)
            and not isinstance(expected_rejected, bool)
            and expected_rejected == rejected_count
            and (allow_valid_count_change or expected_valid == valid_count)
        )

    @staticmethod
    def _load_debug_chunks(document_id: int, artifact: Path) -> list[DocumentChunk]:
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LegacyMaterializationError("Trusted debug artifact is unavailable or malformed.") from error
        raw_chunks = payload.get("chunks")
        if payload.get("document_id") != document_id or not isinstance(raw_chunks, list) or not raw_chunks:
            raise LegacyMaterializationError("Debug artifact document id or chunk schema is invalid.")
        chunks: list[DocumentChunk] = []
        for index, raw in enumerate(raw_chunks):
            if not isinstance(raw, dict):
                raise LegacyMaterializationError("Debug artifact contains an invalid chunk.")
            chunks.append(DocumentChunk(
                id=f"knowledge-document:{document_id}:chunk:{index}", document_id=document_id,
                chunk_index=raw.get("chunk_index"), text_original=raw.get("text_original"),
                text_for_embedding=raw.get("text_for_embedding"),
                page_start=raw.get("page_start"), page_end=raw.get("page_end"),
                section=raw.get("section"), headings=raw.get("headings"),
                content_type=raw.get("content_type"), metadata=raw.get("metadata"),
                token_count=raw.get("token_count"),
            ))
        return chunks

    def _validate_chunks(self, document_id: int, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise LegacyMaterializationError("No validated chunks are available for materialization.")
        for index, chunk in enumerate(chunks):
            if chunk.document_id != document_id or chunk.chunk_index != index:
                raise LegacyMaterializationError("Chunk indexes must be unique and sequential for the selected document.")
            if not isinstance(chunk.text_original, str) or not chunk.text_original.strip():
                raise LegacyMaterializationError("Chunk content is missing.")
            if not isinstance(chunk.text_for_embedding, str) or not chunk.text_for_embedding.strip() or "<!-- image" in chunk.text_for_embedding.lower():
                raise LegacyMaterializationError("Chunk embedding content is invalid.")
            if not isinstance(chunk.token_count, int) or isinstance(chunk.token_count, bool) or not 0 < chunk.token_count <= self.settings.rag_chunk_max_tokens:
                raise LegacyMaterializationError("Chunk token count is invalid.")
            for value in (chunk.page_start, chunk.page_end):
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                    raise LegacyMaterializationError("Chunk source page is invalid.")
            if chunk.page_start is not None and chunk.page_end is not None and chunk.page_end < chunk.page_start:
                raise LegacyMaterializationError("Chunk source page range is invalid.")
            if not isinstance(chunk.headings, list) or not all(isinstance(heading, str) for heading in chunk.headings):
                raise LegacyMaterializationError("Chunk headings are invalid.")
            if not isinstance(chunk.metadata, dict) or not isinstance(chunk.content_type, str) or not chunk.content_type:
                raise LegacyMaterializationError("Chunk metadata is invalid.")
