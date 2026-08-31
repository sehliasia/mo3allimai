"""Structure-preserving Docling HybridChunker adapter with safe overflow handling."""

from collections.abc import Callable, Iterable
from dataclasses import replace
import logging
import re
from typing import Any

from app.services.document_chunk import DocumentChunk
from app.services.document_cleaner import DocumentCleaner
from app.services.document_parser_service import ExtractionQualityValidator
from app.services.worksheet_structure import WorksheetStructureBuilder

logger = logging.getLogger(__name__)


def create_traverse_pictures_provider() -> Any:
    """Build the official Docling provider lazily so normal PDFs keep light imports."""
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import MarkdownParams

    class TraversePicturesProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc: Any) -> Any:
            return ChunkingDocSerializer(doc=doc, params=MarkdownParams(traverse_pictures=True))

    return TraversePicturesProvider()


class DocumentChunkingError(RuntimeError):
    pass


class ChunkQualityCategory:
    VALID_SEMANTIC = "valid_semantic"
    LOW_INFORMATION = "low_information"
    CORRUPTED = "corrupted"


class ChunkMetadataBuilder:
    @staticmethod
    def pages_from_doc_items(doc_items: Iterable[Any]) -> tuple[int | None, int | None]:
        pages: list[int] = []
        for item in doc_items:
            for provenance in getattr(item, "prov", []) or []:
                page_no = getattr(provenance, "page_no", None)
                if isinstance(page_no, int): pages.append(page_no)
        return (min(pages), max(pages)) if pages else (None, None)

    @staticmethod
    def content_type(doc_items: Iterable[Any], text: str) -> str:
        labels = " ".join(str(getattr(item, "label", "")).lower() for item in doc_items)
        if "table" in labels: return "table"
        if "list" in labels: return "list"
        if "heading" in labels or "title" in labels: return "heading"
        return "mixed" if len(labels.split()) > 1 else "text"

    @staticmethod
    def contextual_text(title: str, headings: list[str], text: str) -> str:
        context = [f"Document: {title}"]
        if headings: context.append(f"Section: {' > '.join(headings)}")
        embedding_text = re.sub(r"<!--\s*image\s*-->", "", text).strip()
        return "\n".join([*context, "", embedding_text]).strip()


class DocumentChunker:
    """Use Docling first, then safely split rare contextualization overflows."""

    _paragraphs = re.compile(r"\n\s*\n+")
    _sentences = re.compile(r"(?<=[.!?…؟])\s+")
    _URL_OR_EMAIL = re.compile(r"(?:https?://|www\.)[^\s<>()]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)
    _REFERENCE_TOKEN = re.compile(r"\b(?:[A-C][1-2]\+?|CM\d+|\d{1,4}(?:er|e|ème)?|MENE\d+[A-Z]?)\b", re.IGNORECASE)

    def __init__(
        self,
        max_tokens: int,
        tokenizer_name: str,
        cleaner: DocumentCleaner | None = None,
        *,
        local_files_only: bool = False,
    ):
        if max_tokens < 4: raise ValueError("RAG_CHUNK_MAX_TOKENS must be at least 4.")
        self.max_tokens = max_tokens
        self.tokenizer_name = tokenizer_name
        self.cleaner = cleaner or DocumentCleaner()
        self.local_files_only = local_files_only
        self._tokenizer: Any | None = None

    def chunk(
        self,
        *,
        document: Any,
        document_id: int,
        title: str,
        original_filename: str,
        source: str | None,
        extraction_mode: str = "pdf_text",
        page_extractions: list[Any] | None = None,
        serialized_text_override: str | None = None,
        serialized_page_number: int | None = None,
        _collecting_metrics: bool = False,
    ) -> list[DocumentChunk]:
        if not _collecting_metrics:
            self.last_metrics = {
                "chunks_kept": 0,
                "chunks_skipped_low_information": 0,
                "chunks_failed_quality": 0,
                "chunks_quarantined": 0,
            }
            self.last_corruptions: list[dict[str, Any]] = []
        if page_extractions is not None:
            return self._chunk_page_extractions(
                page_extractions=page_extractions,
                document_id=document_id,
                title=title,
                original_filename=original_filename,
                source=source,
            )
        try:
            from docling.chunking import HybridChunker
        except ImportError as error:
            raise DocumentChunkingError("Docling chunking dependencies are not installed.") from error
        try:
            tokenizer = self._get_tokenizer()
            diagnostics = self._document_diagnostics(document)
            traverse_pictures = self._should_traverse_pictures(extraction_mode)
            logger.info(
                "knowledge_chunk_input document_id=%s extraction_mode=%s texts=%s pictures=%s tables=%s "
                "text_without_pictures_length=%s text_with_pictures_length=%s traverse_pictures=%s",
                document_id,
                extraction_mode,
                diagnostics["texts"],
                diagnostics["pictures"],
                diagnostics["tables"],
                diagnostics["text_without_pictures_length"],
                diagnostics["text_with_pictures_length"],
                traverse_pictures,
            )
            chunker_options: dict[str, Any] = {
                "tokenizer": tokenizer,
                "merge_peers": True,
                "repeat_table_header": True,
            }
            if traverse_pictures:
                # Full-page OCR TextItems may live under PictureItems. Keep the default
                # serializer for all normal PDFs, but traverse those OCR pictures.
                chunker_options["serializer_provider"] = create_traverse_pictures_provider()
            chunker = HybridChunker(**chunker_options)
            native_chunks = list(chunker.chunk(dl_doc=document))
            if serialized_text_override and serialized_text_override.strip():
                # Kept only as a diagnostic for legacy callers. Current OCR pages are
                # normalized structurally before HybridChunker runs, never flattened.
                raw_serialized = "\n".join(str(getattr(chunk, "text", "")) for chunk in native_chunks)
                logger.info(
                    "rtl_chunk_serialization document_id=%s page=%s raw_hybrid_serialized=%r normalized_serialized=%r",
                    document_id, serialized_page_number, raw_serialized[:1200], serialized_text_override[:1200],
                )
        except DocumentChunkingError:
            raise
        except Exception as error:
            raise DocumentChunkingError("Docling could not create structured chunks.") from error

        chunks: list[DocumentChunk] = []
        for native_index, native_chunk in enumerate(native_chunks):
            text_original = self.cleaner.clean(str(getattr(native_chunk, "text", "")))
            metadata = getattr(native_chunk, "meta", None)
            headings = [str(value).strip() for value in (getattr(metadata, "headings", []) or []) if str(value).strip()]
            doc_items = list(getattr(metadata, "doc_items", []) or [])
            page_start, page_end = ChunkMetadataBuilder.pages_from_doc_items(doc_items)
            content_type = ChunkMetadataBuilder.content_type(doc_items, text_original)
            if not text_original:
                self.last_metrics["chunks_skipped_low_information"] += 1
                logger.info(
                    "chunk_skipped_low_information document_id=%s page=%s content_type=%s reason=empty",
                    document_id, page_start, content_type,
                )
                continue
            try:
                fragments = self._split_to_budget(text_original, lambda value: ChunkMetadataBuilder.contextual_text(title, headings, value), lambda value: self._count_tokens(tokenizer, value), content_type)
            except DocumentChunkingError as error:
                logger.error("knowledge_chunk_unsplittable document_id=%s native_chunk_index=%s content_type=%s page_start=%s page_end=%s error=%s", document_id, native_index, content_type, page_start, page_end, error)
                raise
            for fragment in fragments:
                text_for_embedding = ChunkMetadataBuilder.contextual_text(title, headings, fragment)
                token_count = self._count_tokens(tokenizer, text_for_embedding)
                if token_count > self.max_tokens:
                    logger.error("knowledge_chunk_over_budget document_id=%s native_chunk_index=%s content_type=%s token_count=%s page_start=%s page_end=%s", document_id, native_index, content_type, token_count, page_start, page_end)
                    raise DocumentChunkingError("Overflow fallback produced a chunk above the configured token limit.")
                category, reason, chunk_quality = self._classify_final_chunk(
                    fragment, content_type=content_type, token_count=token_count
                )
                if category == ChunkQualityCategory.LOW_INFORMATION:
                    self.last_metrics["chunks_skipped_low_information"] += 1
                    logger.info(
                        "chunk_skipped_low_information document_id=%s page=%s content_type=%s reason=%s",
                        document_id, page_start, content_type, reason,
                    )
                    continue
                if category == ChunkQualityCategory.CORRUPTED:
                    self.last_metrics["chunks_failed_quality"] += 1
                    self.last_metrics["chunks_quarantined"] += 1
                    corruption = {
                        "page": page_start,
                        "extraction_mode": extraction_mode,
                        "content_type": content_type,
                        "token_count": token_count,
                        "quality_score": chunk_quality.quality_score,
                        "failure_reasons": list(chunk_quality.failure_reasons) or [reason],
                        "text_preview": repr(fragment[:500]),
                    }
                    self.last_corruptions.append(corruption)
                    logger.error(
                        "chunk_corruption_detected document_id=%s page=%s extraction_mode=%s "
                        "content_type=%s token_count=%s score=%.2f reasons=%s text=%r",
                        document_id, page_start, extraction_mode, content_type, token_count,
                        chunk_quality.quality_score, ",".join(chunk_quality.failure_reasons) or reason,
                        fragment[:500],
                    )
                    # The ingestion service may repair this native page once with OCR.
                    # Never emit this fragment to future embeddings.
                    continue
                index = len(chunks)
                chunk_metadata = {"document_id": document_id, "original_filename": original_filename, "source": source, "page_start": page_start, "page_end": page_end, "headings": headings, "content_type": content_type, "language": None, "document_type": None, "level": None, "skill": None, "theme": None}
                chunks.append(DocumentChunk(id=f"knowledge-document:{document_id}:chunk:{index}", document_id=document_id, chunk_index=index, text_original=fragment, text_for_embedding=text_for_embedding, page_start=page_start, page_end=page_end, section=" > ".join(headings) or None, headings=headings, content_type=content_type, metadata=chunk_metadata, token_count=token_count))
                if serialized_page_number in {1, 2, 3}:
                    logger.info("rtl_final_chunk document_id=%s page=%s final_chunk_text=%r", document_id, serialized_page_number, fragment[:1200])
                self.last_metrics["chunks_kept"] += 1
        if extraction_mode == "full_page_ocr" and chunks:
            final_quality = ExtractionQualityValidator().assess(
                "\n".join(chunk.text_original for chunk in chunks)
            )
            # OCR is multilingual. Only apply the Arabic-specific requirement
            # when this page's final text contains meaningful Arabic evidence.
            arabic_rich = final_quality.arabic_unicode_count >= 3
            if (arabic_rich and final_quality.glyph_noise_count) or (arabic_rich and not final_quality.has_arabic_unicode):
                raise DocumentChunkingError(
                    "Full-page OCR chunks failed Arabic extraction-quality validation."
                )
        if not chunks and not _collecting_metrics:
            raise DocumentChunkingError("No non-empty chunks were produced.")
        return chunks

    def _get_tokenizer(self) -> Any:
        """Create the sizing tokenizer once for this chunking operation."""
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            from transformers import AutoTokenizer
        except ImportError as error:
            raise DocumentChunkingError("Docling chunking dependencies are not installed.") from error
        tokenizer_options = {"local_files_only": True} if self.local_files_only else {}
        # Transformers 5.15 incorrectly treats the old BERT config (which has
        # no transformers_version field) as Mistral metadata. Explicitly retain
        # BERT's existing, unpatched regex semantics; never apply the Mistral fix.
        if self.tokenizer_name == "bert-base-multilingual-cased":
            tokenizer_options["fix_mistral_regex"] = False
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, **tokenizer_options)
            model_limit = hf_tokenizer.model_max_length
            if isinstance(model_limit, int) and model_limit < self.max_tokens:
                raise DocumentChunkingError(
                    f"Configured chunk limit {self.max_tokens} exceeds tokenizer limit {model_limit} for {self.tokenizer_name}."
                )
            # Used for measuring only: avoid advisory warnings while HybridChunker
            # measures a raw oversized unit before splitting it.
            hf_tokenizer.model_max_length = max(model_limit, 1_000_000)
            self._tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=self.max_tokens)
            logger.info(
                "knowledge_chunk_tokenizer_initialized tokenizer=%s local_files_only=%s",
                self.tokenizer_name,
                self.local_files_only,
            )
            return self._tokenizer
        except DocumentChunkingError:
            raise
        except Exception as error:
            raise DocumentChunkingError("The configured chunk tokenizer could not be initialized.") from error

    def _chunk_page_extractions(
        self,
        *,
        page_extractions: list[Any],
        document_id: int,
        title: str,
        original_filename: str,
        source: str | None,
    ) -> list[DocumentChunk]:
        """Merge independently selected Docling page documents without losing page provenance."""
        merged: list[DocumentChunk] = []
        for page in page_extractions:
            corruption_start = len(self.last_corruptions)
            sections = []
            if page.extraction_mode == "full_page_ocr":
                sections = WorksheetStructureBuilder().sections(
                    page.document, page.page_number, has_image=bool(page.images)
                )
            page_chunks = self._chunk_worksheet_sections(
                sections, document_id=document_id, title=title,
                original_filename=original_filename, source=source,
                page_number=page.page_number, extraction_mode=page.extraction_mode,
            ) if len(sections) > 1 else self.chunk(
                document=page.document,
                document_id=document_id,
                title=title,
                original_filename=original_filename,
                source=source,
                extraction_mode=page.extraction_mode,
                serialized_text_override=page.reading_order_text if page.extraction_mode == "full_page_ocr" else None,
                serialized_page_number=page.page_number,
                _collecting_metrics=True,
            )
            for corruption in self.last_corruptions[corruption_start:]:
                # A page-range conversion may locally number its only page as 1;
                # the selected page object is the authoritative original page number.
                corruption["page"] = page.page_number
            for chunk in page_chunks:
                index = len(merged)
                metadata = {
                    **chunk.metadata,
                    "page_number": page.page_number,
                    "extraction_mode": page.extraction_mode,
                    "extraction_modes_used": [page.extraction_mode],
                    "quality_score": page.quality.quality_score,
                    "languages": list(page.quality.languages_detected),
                }
                merged.append(replace(
                    chunk,
                    id=f"knowledge-document:{document_id}:chunk:{index}",
                    chunk_index=index,
                    page_start=chunk.page_start or page.page_number,
                    page_end=chunk.page_end or page.page_number,
                    metadata=metadata,
                ))
        if not merged:
            raise DocumentChunkingError("No non-empty chunks were produced from selected page extractions.")
        logger.info(
            "knowledge_chunk_summary document_id=%s chunks_kept=%s chunks_skipped_low_information=%s chunks_failed_quality=%s",
            document_id,
            self.last_metrics["chunks_kept"],
            self.last_metrics["chunks_skipped_low_information"],
            self.last_metrics["chunks_failed_quality"],
        )
        return merged

    def _chunk_worksheet_sections(
        self,
        sections: list[Any],
        *,
        document_id: int,
        title: str,
        original_filename: str,
        source: str | None,
        page_number: int,
        extraction_mode: str,
    ) -> list[DocumentChunk]:
        """Use geometric exercise boundaries only for complex OCR worksheets.

        Normal prose and tables still use Docling HybridChunker.  This path never
        reverses or rewrites OCR words and emits no image marker as text.
        """
        tokenizer = self._get_tokenizer()
        chunks: list[DocumentChunk] = []
        for section in sections:
            text = self.cleaner.clean(section.text)
            headings = [f"تمرين {section.number}"] if section.number else []
            for fragment in self._split_to_budget(
                text,
                lambda value: ChunkMetadataBuilder.contextual_text(title, headings, value),
                lambda value: self._count_tokens(tokenizer, value),
                "worksheet_exercise",
            ):
                embedding_text = ChunkMetadataBuilder.contextual_text(title, headings, fragment)
                token_count = self._count_tokens(tokenizer, embedding_text)
                category, _reason, _quality = self._classify_final_chunk(
                    fragment, content_type="worksheet_exercise", token_count=token_count
                )
                if category != ChunkQualityCategory.VALID_SEMANTIC:
                    self.last_metrics["chunks_skipped_low_information" if category == ChunkQualityCategory.LOW_INFORMATION else "chunks_failed_quality"] += 1
                    continue
                index = len(chunks)
                chunks.append(DocumentChunk(
                    id=f"knowledge-document:{document_id}:chunk:{index}", document_id=document_id,
                    chunk_index=index, text_original=fragment, text_for_embedding=embedding_text,
                    page_start=page_number, page_end=page_number, section=" > ".join(headings) or None,
                    headings=headings, content_type="worksheet_exercise",
                    metadata={"document_id": document_id, "original_filename": original_filename, "source": source,
                              "page_start": page_number, "page_end": page_number, "headings": headings,
                              "content_type": "worksheet_exercise", "extraction_mode": extraction_mode,
                              "has_image": section.has_image, "requires_vision": section.requires_vision,
                              "structural_quality": section.structural_quality},
                    token_count=token_count,
                ))
                self.last_metrics["chunks_kept"] += 1
        return chunks

    @staticmethod
    def _classify_final_chunk(
        text: str,
        *,
        content_type: str,
        token_count: int,
    ) -> tuple[str, str, Any]:
        """Separate harmless workbook decoration from meaningful corruption."""
        quality = ExtractionQualityValidator().assess(text)
        compact = re.sub(r"\s+", "", text)
        meaningful_letters = sum(char.isalpha() for char in text)
        word_count = len(re.findall(r"\b\w+\b", text, re.UNICODE))
        decorative_characters = sum(char in "_.-–—□☐☑▪•" for char in compact)
        decorative_ratio = decorative_characters / max(1, len(compact))
        short_blank_field = (
            len(compact) <= 48
            and meaningful_letters <= 2
            and word_count <= 2
            and (decorative_ratio >= 0.35 or token_count <= 3)
        )
        if not compact:
            return ChunkQualityCategory.LOW_INFORMATION, "empty", quality
        if short_blank_field and content_type != "table":
            return ChunkQualityCategory.LOW_INFORMATION, "decorative_or_blank_field", quality
        if quality.glyph_name_count:
            return ChunkQualityCategory.CORRUPTED, "glyph_noise", quality
        if content_type == "table" and meaningful_letters + sum(char.isdigit() for char in text) >= 1:
            return ChunkQualityCategory.VALID_SEMANTIC, "meaningful_table_cell", quality
        structured_text = DocumentChunker._URL_OR_EMAIL.sub(" ", text)
        structured_text = DocumentChunker._REFERENCE_TOKEN.sub(" ", structured_text)
        structured_quality = ExtractionQualityValidator().assess(structured_text)
        reference_only = bool(DocumentChunker._URL_OR_EMAIL.search(text)) and not re.sub(
            r"[\s\W\d_]", "", structured_text, flags=re.UNICODE
        )
        if reference_only:
            return ChunkQualityCategory.LOW_INFORMATION, "structured_reference", quality
        substantial_garbage = (
            len(compact) >= 40
            and (
                structured_quality.quality_score < 0.45
                or structured_quality.single_character_token_ratio > 0.55
            )
        )
        severe_random_ascii = (
            len(compact) >= 12
            and structured_quality.suspicious_ascii_sequence_count >= 2
            and (
                structured_quality.quality_score < 0.45
                or structured_quality.symbol_ratio > 0.30
                or structured_quality.single_character_token_ratio > 0.55
            )
        )
        if severe_random_ascii:
            return ChunkQualityCategory.CORRUPTED, "severe_random_ascii_corruption", quality
        if substantial_garbage:
            return ChunkQualityCategory.CORRUPTED, "substantial_extraction_garbage", quality
        return ChunkQualityCategory.VALID_SEMANTIC, "semantic_content", quality

    @staticmethod
    def _document_diagnostics(document: Any) -> dict[str, int]:
        """Inspect both serializer paths without mutating the Docling document."""
        def items_count(attribute: str) -> int:
            return len(getattr(document, attribute, []) or [])

        def exported_text(traverse_pictures: bool) -> str:
            exporter = getattr(document, "export_to_text", None)
            if not callable(exporter):
                return ""
            return str(exporter(traverse_pictures=traverse_pictures) or "")

        without_pictures = exported_text(False)
        with_pictures = exported_text(True)
        return {
            "texts": items_count("texts"),
            "pictures": items_count("pictures"),
            "tables": items_count("tables"),
            "text_without_pictures_length": len(without_pictures),
            "text_with_pictures_length": len(with_pictures),
        }

    @staticmethod
    def _should_traverse_pictures(extraction_mode: str) -> bool:
        return extraction_mode == "full_page_ocr"

    @staticmethod
    def _count_tokens(tokenizer: Any, text: str) -> int:
        try: return int(tokenizer.count_tokens(text=text))
        except TypeError: return int(tokenizer.count_tokens(text))

    def _split_to_budget(self, text: str, contextualize: Callable[[str], str], count_tokens: Callable[[str], int], content_type: str) -> list[str]:
        text = self.cleaner.clean(text)
        if not text: return []
        if count_tokens(contextualize(text)) <= self.max_tokens: return [text]
        if count_tokens(contextualize("")) >= self.max_tokens: raise DocumentChunkingError("Document and heading context exhaust the configured token budget.")
        if content_type == "table": return self._split_table(text, contextualize, count_tokens)
        return self._pack_units(self._paragraph_units(text), contextualize, count_tokens)

    def _split_table(self, text: str, contextualize: Callable[[str], str], count_tokens: Callable[[str], int]) -> list[str]:
        rows = [line for line in text.splitlines() if "|" in line]
        if not rows: return self._pack_units(self._paragraph_units(text), contextualize, count_tokens)
        is_separator = lambda row: set(row.replace("|", "").strip()) <= {"-", ":", " "}
        header_lines = rows[:2] if len(rows) > 1 and is_separator(rows[1]) else rows[:1]
        header, body_rows = "\n".join(header_lines), rows[len(header_lines):]
        if not body_rows: return self._pack_units([header], contextualize, count_tokens)
        fragments: list[str] = []
        current: list[str] = []
        for row in body_rows:
            candidate = "\n".join([header, *current, row])
            if count_tokens(contextualize(candidate)) <= self.max_tokens:
                current.append(row); continue
            if current:
                fragments.append("\n".join([header, *current])); current = []
            if count_tokens(contextualize(f"{header}\n{row}")) <= self.max_tokens:
                current.append(row); continue
            row_parts = self._pack_units(self._cell_units(row), lambda part: contextualize(f"{header}\n{part}"), count_tokens)
            fragments.extend(f"{header}\n{part}" for part in row_parts)
        if current: fragments.append("\n".join([header, *current]))
        return fragments

    def _paragraph_units(self, text: str) -> list[str]:
        units: list[str] = []
        for paragraph in self._paragraphs.split(text):
            paragraph = paragraph.strip()
            if not paragraph: continue
            units.extend(sentence.strip() for sentence in self._sentences.split(paragraph) if sentence.strip())
        return units or [text]

    @staticmethod
    def _cell_units(row: str) -> list[str]:
        cells = [value.strip() for value in row.split("|") if value.strip()]
        return [f"| {cell} |" for cell in cells] or [row]

    def _pack_units(self, units: list[str], contextualize: Callable[[str], str], count_tokens: Callable[[str], int]) -> list[str]:
        fragments: list[str] = []
        current: list[str] = []
        for unit in units:
            candidate = "\n\n".join([*current, unit])
            if count_tokens(contextualize(candidate)) <= self.max_tokens:
                current.append(unit); continue
            if current:
                fragments.append("\n\n".join(current)); current = []
            if count_tokens(contextualize(unit)) <= self.max_tokens:
                current.append(unit); continue
            fragments.extend(self._pack_words(unit.split(), contextualize, count_tokens))
        if current: fragments.append("\n\n".join(current))
        if not fragments: raise DocumentChunkingError("No fragments could be created within the token budget.")
        return fragments

    def _pack_words(self, words: list[str], contextualize: Callable[[str], str], count_tokens: Callable[[str], int]) -> list[str]:
        fragments: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if count_tokens(contextualize(candidate)) <= self.max_tokens:
                current.append(word); continue
            if current:
                fragments.append(" ".join(current)); current = []
            if count_tokens(contextualize(word)) > self.max_tokens: raise DocumentChunkingError("A single token-like value exceeds the configured token budget.")
            current.append(word)
        if current: fragments.append(" ".join(current))
        return fragments
