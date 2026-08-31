"""Docling-first, page-level extraction quality control and adaptive OCR."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
import hashlib
import json
import logging
from pathlib import Path
import re
import time
import unicodedata
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class DocumentParsingError(RuntimeError):
    """Raised when a page cannot yield trustworthy text for chunking."""


class ModelResourceUnavailableError(DocumentParsingError):
    """Required Docling/HuggingFace model resources could not be loaded."""

class OcrPageTimeoutError(DocumentParsingError):
    """A targeted OCR conversion exceeded its bounded execution window."""


class NativeQualityLevel:
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE_NATIVE"
    BORDERLINE = "BORDERLINE_NATIVE"
    CORRUPTED = "CORRUPTED_NATIVE"


@dataclass(frozen=True)
class ExtractionQuality:
    page_number: int | None
    character_count: int
    word_count: int
    arabic_unicode_count: int
    latin_unicode_count: int
    digit_count: int
    symbol_count: int
    arabic_character_ratio: float
    latin_character_ratio: float
    symbol_ratio: float
    glyph_name_count: int
    replacement_character_count: int
    control_character_count: int
    suspicious_ascii_sequence_count: int
    single_character_token_ratio: float
    repeated_symbol_ratio: float
    meaningful_word_ratio: float
    digit_inside_word_ratio: float
    symbol_inside_word_ratio: float
    quality_score: float
    quality_passed: bool
    failure_reasons: tuple[str, ...]
    languages_detected: tuple[str, ...]

    @property
    def has_arabic_unicode(self) -> bool:
        return self.arabic_unicode_count > 0

    @property
    def glyph_noise_count(self) -> int:
        return self.glyph_name_count

    @property
    def needs_ocr_fallback(self) -> bool:
        return not self.quality_passed


class PageExtractionQualityAnalyzer:
    """Quality gate for multilingual PDF text. It detects corruption, not language preference."""

    _ARABIC = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
    _LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
    _WORD = re.compile(r"\b\w+\b", re.UNICODE)
    _REPEATED_SYMBOL = re.compile(r"([^\w\s])\1{2,}")
    _SUSPICIOUS_ALNUM = re.compile(r"(?<!\w)(?=[A-Za-z0-9]{3,}\b)(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]+")
    _TOKEN = re.compile(r"\S+", re.UNICODE)
    _GLYPH_NAME_PATTERNS = (
        "alefisolated", "alefinitial", "alefmedial", "behinitial", "hehinitial",
        "meemmedial", "thalfinal", "fathalow", "fatha", "kasra", "damma",
        "sukun", "shadda", "arabicindicdigit",
    )
    _FRENCH_HINTS = re.compile(r"\b(le|la|les|des|un|une|et|avec|dans|pour|réponds|texte|lis|consigne)\b", re.I)
    _ENGLISH_HINTS = re.compile(r"\b(the|and|with|read|make|sentences|answer|text|instructions)\b", re.I)

    def __init__(self, min_score: float = 0.65):
        self.min_score = min_score

    def assess(self, text: str, *, page_number: int | None = None) -> ExtractionQuality:
        normalized = text.strip()
        characters = [char for char in normalized if not char.isspace()]
        character_count = len(characters)
        lower_text = normalized.lower()
        arabic_count = sum(bool(self._ARABIC.match(char)) for char in characters)
        latin_count = sum(bool(self._LATIN.match(char)) for char in characters)
        digit_count = sum(char.isdigit() for char in characters)
        symbol_count = sum(not char.isalnum() and not self._ARABIC.match(char) for char in characters)
        glyph_count = sum(lower_text.count(pattern) for pattern in self._GLYPH_NAME_PATTERNS)
        replacement_count = normalized.count("\ufffd")
        control_count = sum(unicodedata.category(char).startswith("C") and char not in "\n\r\t" for char in normalized)
        words = self._WORD.findall(normalized)
        tokens = self._TOKEN.findall(normalized)
        single_ratio = (sum(len(word) == 1 for word in words) / len(words)) if words else 1.0
        meaningful_word_ratio = (sum(len(word) >= 3 and any(char.isalpha() for char in word) for word in words) / len(words)) if words else 0.0
        mixed_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
        digit_inside_word_ratio = (sum(any(char.isdigit() for char in token) for token in mixed_tokens) / len(mixed_tokens)) if mixed_tokens else 0.0
        symbol_inside_word_ratio = (sum(any(not char.isalnum() and not char.isspace() for char in token.strip(".,;:!?…؟")) for token in mixed_tokens) / len(mixed_tokens)) if mixed_tokens else 0.0
        repeated_symbols = len(self._REPEATED_SYMBOL.findall(normalized))
        repeated_ratio = repeated_symbols / max(1, len(words))
        suspicious_sequences = len(self._SUSPICIOUS_ALNUM.findall(normalized))
        non_space = max(1, character_count)
        arabic_ratio = arabic_count / non_space
        latin_ratio = latin_count / non_space
        symbol_ratio = symbol_count / non_space

        reasons: list[str] = []
        # A good OCR or native text is rarely perfect. Reserve headroom so score=1.0
        # is never assigned solely because no basic error signal was found.
        score = 0.95
        if not normalized:
            reasons.append("empty_text")
            score = 0.0
        if glyph_count:
            reasons.append("glyph_noise")
            score -= 0.8
        if replacement_count:
            reasons.append("replacement_characters")
            score -= min(0.5, replacement_count * 0.1)
        if control_count:
            reasons.append("control_characters")
            score -= min(0.4, control_count * 0.1)
        if arabic_ratio + latin_ratio < 0.45:
            reasons.append("low_letter_ratio")
            score -= 0.4
        if symbol_ratio > 0.25:
            reasons.append("high_symbol_ratio")
            score -= min(0.35, symbol_ratio)
        if single_ratio > 0.55 and len(words) >= 4:
            reasons.append("high_single_character_token_ratio")
            score -= 0.25
        if repeated_ratio > 0.1:
            reasons.append("repeated_symbols")
            score -= 0.2
        if suspicious_sequences >= 2 and (symbol_ratio > 0.1 or single_ratio > 0.35):
            reasons.append("suspicious_ascii_sequences")
            score -= 0.3
        if digit_inside_word_ratio > 0.20:
            reasons.append("digit_inside_word_ratio")
            score -= min(0.35, digit_inside_word_ratio * 0.45)
        if symbol_inside_word_ratio > 0.20:
            reasons.append("symbol_inside_word_ratio")
            score -= min(0.30, symbol_inside_word_ratio * 0.40)
        if meaningful_word_ratio < 0.35 and len(words) >= 4:
            reasons.append("low_meaningful_word_ratio")
            score -= 0.22
        score = max(0.0, min(1.0, score))
        languages: list[str] = []
        if arabic_count:
            languages.append("ar")
        if latin_count:
            french_evidence = len(set(self._FRENCH_HINTS.findall(normalized)))
            english_evidence = len(set(self._ENGLISH_HINTS.findall(normalized)))
            if french_evidence >= 2 and meaningful_word_ratio >= 0.5: languages.append("fr")
            if english_evidence >= 2 and meaningful_word_ratio >= 0.5: languages.append("en")
            if not any(language in languages for language in ("fr", "en")): languages.append("latin")
        passed = score >= self.min_score and not {"empty_text", "glyph_noise", "replacement_characters", "control_characters"}.intersection(reasons)
        return ExtractionQuality(
            page_number=page_number, character_count=character_count, word_count=len(words),
            arabic_unicode_count=arabic_count, latin_unicode_count=latin_count, digit_count=digit_count,
            symbol_count=symbol_count, arabic_character_ratio=arabic_ratio, latin_character_ratio=latin_ratio,
            symbol_ratio=symbol_ratio, glyph_name_count=glyph_count, replacement_character_count=replacement_count,
            control_character_count=control_count, suspicious_ascii_sequence_count=suspicious_sequences,
            single_character_token_ratio=single_ratio, repeated_symbol_ratio=repeated_ratio,
            meaningful_word_ratio=meaningful_word_ratio, digit_inside_word_ratio=digit_inside_word_ratio,
            symbol_inside_word_ratio=symbol_inside_word_ratio,
            quality_score=score, quality_passed=passed, failure_reasons=tuple(reasons), languages_detected=tuple(languages),
        )


# Compatibility name retained for callers and earlier tests.
ExtractionQualityValidator = PageExtractionQualityAnalyzer


@dataclass(frozen=True)
class PageExtraction:
    page_number: int
    document: Any
    extraction_mode: str
    quality: ExtractionQuality
    reading_order_text: str | None = None
    raw_docling_text: str | None = None
    images: list["PreservedImage"] = field(default_factory=list)


@dataclass(frozen=True)
class PageExtractionIssue:
    """A page deliberately omitted from chunking, with enough evidence to audit it."""

    page_number: int
    disposition: str
    extraction_mode_attempted: str
    native_quality: ExtractionQuality
    ocr_quality: ExtractionQuality | None
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PreservedImage:
    image_id: str
    document_id: int
    page: int
    bbox: dict[str, float] | None
    path: str | None
    caption: str | None
    nearby_text: str | None
    image_role: str = "unknown"
    associated_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedDocument:
    document: Any
    pages_count: int
    items_count: int
    extraction_mode: str
    page_extractions: list[PageExtraction]
    timings_ms: dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False
    ocr_strategy: str = "targeted_pages"
    ocr_required_page_ratio: float = 0.0
    cache_key: str | None = None
    cache_miss_reason: str | None = None
    cache_write_success: bool = False
    page_issues: list[PageExtractionIssue] = field(default_factory=list)


@dataclass(frozen=True)
class NativePreflight:
    pages_total: int
    items_count: int
    native_good_page_numbers: list[int]
    native_borderline_page_numbers: list[int]
    native_bad_page_numbers: list[int]
    detected_picture_count: int | None
    detected_table_count: int | None
    native_analysis_duration_ms: int
    page_decisions: list[dict[str, Any]] = field(default_factory=list)


ConverterFactory = Callable[[Path, bool, tuple[int, int] | None], Any]


class DocumentParserService:
    """Keeps good native pages and replaces only objectively bad pages with Docling OCR."""

    def __init__(self, converter_factory: ConverterFactory | None = None):
        self._converter_factory = converter_factory
        settings = get_settings()
        self._settings = settings
        self._quality_analyzer = PageExtractionQualityAnalyzer(settings.rag_extraction_min_quality_score)
        self._conversion_timings: list[dict[str, float | str | int | None]] = []
        self._cache_miss_reason: str | None = None
        self._active_preflight_converter: Any | None = None
        self._preflight_lifecycle: dict[str, Any] | None = None
        # A parser instance owns converters for one ingestion/preflight operation.
        # Reusing them prevents Docling from repeatedly resolving model resources.
        self._converters: dict[bool, Any] = {}

    @contextmanager
    def native_preflight_session(self):
        """Reuse one native Docling converter for all ranges in one preflight request."""
        lifecycle: dict[str, Any] = {
            "document_converter_instances_created": 0,
            "conversion_calls": 0,
            "page_ranges_processed": [],
        }
        self._preflight_lifecycle = lifecycle
        try:
            if self._converter_factory is None:
                self._active_preflight_converter = self._build_converter(use_ocr=False)
                lifecycle["document_converter_instances_created"] = 1
            yield lifecycle
        finally:
            self._active_preflight_converter = None
            self._preflight_lifecycle = None

    def parse_pdf(
        self,
        path: Path,
        *,
        document_id: int | None = None,
        force_reprocess: bool = False,
        strict_cache_write: bool = False,
    ) -> ParsedDocument:
        if not path.is_file():
            raise DocumentParsingError("The document file is missing from private storage.")
        cache_key = self._cache_key(path)
        cache_key_digest = self._cache_key_digest(cache_key)
        self._cache_miss_reason = "force_reprocess" if force_reprocess else None
        logger.info("knowledge_extraction_cache_key document_id=%s components=%s computed_cache_key=%s", document_id, cache_key, cache_key_digest)
        if self._converter_factory is None and document_id is not None and self._settings.rag_extraction_cache_enabled and not force_reprocess:
            cached = self._load_cache(document_id, cache_key)
            if cached is not None:
                logger.info("knowledge_extraction_cache_hit document_id=%s", document_id)
                return cached
        timings_ms: dict[str, float] = {}
        self._conversion_timings = []
        conversion_started = time.perf_counter()
        native_document = self._convert(path, use_ocr=False, page_range=None)
        timings_ms["native_docling_conversion"] = round((time.perf_counter() - conversion_started) * 1000, 2)
        pages_count, items_count = self._document_counts(native_document)
        if not pages_count or not items_count:
            raise DocumentParsingError("No extractable pages were found after native Docling parsing.")

        native_candidates: list[tuple[int, ExtractionQuality]] = []
        for page_number in range(1, pages_count + 1):
            native_text = self._page_text(native_document, page_number, traverse_pictures=False)
            native_quality = self._quality_analyzer.assess(native_text, page_number=page_number)
            native_candidates.append((page_number, native_quality))

        arabic_high_confidence_pages = sum(
            self._native_level(quality, arabic_profile=False) == NativeQualityLevel.HIGH_CONFIDENCE
            and quality.has_arabic_unicode
            for _, quality in native_candidates
        )
        arabic_profile = arabic_high_confidence_pages >= 2
        pages: list[PageExtraction] = []
        page_issues: list[PageExtractionIssue] = []
        for page_number, native_quality in native_candidates:
            level = self._native_level(native_quality, arabic_profile=arabic_profile)
            decision = self._native_page_decision(native_document, page_number, native_quality, level)
            self._log_quality(document_id, page_number, "native", native_quality, classification=level)
            if decision["native_usable"]:
                # Re-convert one native page to preserve a page-bounded Docling structure for merging/chunking.
                page_started = time.perf_counter()
                page_document = self._convert(path, use_ocr=False, page_range=(page_number, page_number))
                timings_ms[f"native_page_{page_number}"] = round((time.perf_counter() - page_started) * 1000, 2)
                pages.append(PageExtraction(page_number, page_document, "native", native_quality, images=self._extract_images(page_document, document_id, page_number)))
                self._log_selected(document_id, page_number, "native", native_quality.quality_score, None, decision["ocr_candidate_reason"])
                continue
            if not self._settings.rag_ocr_enabled:
                raise self._quality_error(page_number, native_quality, "OCR is disabled")
            if self._settings.rag_ocr_engine.lower() != "easyocr":
                raise self._quality_error(page_number, native_quality, "Configured OCR engine is unsupported")
            logger.warning("page_ocr_fallback document_id=%s page=%s engine=easyocr classification=%s", document_id, page_number, level)
            ocr_started = time.perf_counter()
            logger.info("page_ocr_started document_id=%s page=%s", document_id, page_number)
            try:
                ocr_document = self._convert_targeted_ocr(path, page_number)
            except OcrPageTimeoutError:
                elapsed_ms = round((time.perf_counter() - ocr_started) * 1000, 2)
                page_issues.append(PageExtractionIssue(page_number, "quarantined", "full_page_ocr", native_quality, None, ("ocr_timeout",)))
                logger.warning("page_ocr_timeout document_id=%s page=%s elapsed_ms=%s", document_id, page_number, elapsed_ms)
                continue
            timings_ms[f"ocr_page_{page_number}"] = round((time.perf_counter() - ocr_started) * 1000, 2)
            logger.info("page_ocr_completed document_id=%s page=%s elapsed_ms=%s", document_id, page_number, timings_ms[f"ocr_page_{page_number}"])
            ocr_text = self._page_text(ocr_document, page_number, traverse_pictures=True)
            ocr_quality = self._quality_analyzer.assess(ocr_text, page_number=page_number)
            self._log_quality(document_id, page_number, "ocr_candidate", ocr_quality)
            ocr_better = ocr_quality.quality_passed and ocr_quality.quality_score >= native_quality.quality_score + self._settings.rag_ocr_min_selection_gain
            if ocr_better:
                rtl_started = time.perf_counter()
                self._apply_geometric_reading_order(ocr_document)
                timings_ms[f"rtl_normalization_page_{page_number}"] = round((time.perf_counter() - rtl_started) * 1000, 2)
                normalized = self._geometric_reading_order(ocr_document, page_number)
                if page_number in {1, 2, 3}:
                    self._log_ocr_granularity(document_id, ocr_document, page_number)
                    logger.info("rtl_reading_order document_id=%s page=%s raw_docling_order=%r normalized_rtl_order=%r", document_id, page_number, ocr_text[:1200], normalized[:1200])
                image_started = time.perf_counter()
                images = self._extract_images(ocr_document, document_id, page_number)
                timings_ms[f"picture_extraction_page_{page_number}"] = round((time.perf_counter() - image_started) * 1000, 2)
                pages.append(PageExtraction(page_number, ocr_document, "full_page_ocr", ocr_quality, reading_order_text=normalized, raw_docling_text=ocr_text, images=images))
                self._log_selected(document_id, page_number, "ocr", native_quality.quality_score, ocr_quality.quality_score, "borderline_native_and_ocr_better")
                continue
            if level == NativeQualityLevel.BORDERLINE and native_quality.quality_passed:
                page_document = self._convert(path, use_ocr=False, page_range=(page_number, page_number))
                pages.append(PageExtraction(page_number, page_document, "native", native_quality))
                self._log_selected(document_id, page_number, "native", native_quality.quality_score, ocr_quality.quality_score, "borderline_native_not_worse_than_ocr")
                continue
            if self._is_structurally_blank_page(ocr_document, page_number):
                page_issues.append(PageExtractionIssue(
                    page_number=page_number,
                    disposition="skipped_low_information",
                    extraction_mode_attempted="full_page_ocr",
                    native_quality=native_quality,
                    ocr_quality=ocr_quality,
                    failure_reasons=("blank_or_decorative_page",),
                ))
                logger.info(
                    "page_extraction_skipped_low_information document_id=%s page=%s native_reasons=%s ocr_reasons=%s",
                    document_id, page_number, ",".join(native_quality.failure_reasons), ",".join(ocr_quality.failure_reasons),
                )
                continue
            page_issues.append(PageExtractionIssue(
                page_number=page_number,
                disposition="quarantined",
                extraction_mode_attempted="full_page_ocr",
                native_quality=native_quality,
                ocr_quality=ocr_quality,
                failure_reasons=ocr_quality.failure_reasons or native_quality.failure_reasons,
            ))
            logger.warning(
                "page_extraction_quarantined document_id=%s page=%s native_score=%.2f ocr_score=%.2f reasons=%s",
                document_id, page_number, native_quality.quality_score, ocr_quality.quality_score,
                ",".join(ocr_quality.failure_reasons or native_quality.failure_reasons),
            )

        logger.info(
            "document_extraction_summary document_id=%s native_pages=%s ocr_pages=%s failed_pages=%s skipped_pages=%s",
            document_id, sum(page.extraction_mode == "native" for page in pages), sum(page.extraction_mode == "full_page_ocr" for page in pages),
            sum(issue.disposition == "quarantined" for issue in page_issues),
            sum(issue.disposition == "skipped_low_information" for issue in page_issues),
        )
        timings_ms["ocr_pages_total"] = round(sum(value for key, value in timings_ms.items() if key.startswith("ocr_page_")), 2)
        timings_ms["ocr_initialization"] = round(sum(
            float(event["converter_init_ms"]) for event in self._conversion_timings if event["mode"] == "ocr"
        ), 2)
        logger.info("document_extraction_timings document_id=%s timings_ms=%s conversion_events=%s ocr_converter_created_per_page=true", document_id, timings_ms, self._conversion_timings)
        parsed = ParsedDocument(native_document, pages_count, items_count, "page_adaptive", pages, timings_ms, False, "targeted_pages", len([page for page in pages if page.extraction_mode == "full_page_ocr"]) / max(1, pages_count), cache_key_digest, self._cache_miss_reason, False, page_issues)
        if self._converter_factory is None and document_id is not None and self._settings.rag_extraction_cache_enabled:
            try:
                self._write_cache(document_id, cache_key, parsed)
                parsed = replace(parsed, cache_write_success=True)
            except Exception as error:
                logger.warning("knowledge_extraction_cache_write_failed document_id=%s", document_id, exc_info=True)
                if strict_cache_write:
                    raise DocumentParsingError("The extraction cache could not be persisted for this debug run.") from error
        return parsed

    def preflight_pdf(self, path: Path, *, page_range: tuple[int, int] | None = None) -> NativePreflight:
        """Run only the native conversion and the ingestion's existing page classifier."""
        if not path.is_file():
            raise DocumentParsingError("The document file is missing from private storage.")
        started = time.perf_counter()
        native_document = self._convert(path, use_ocr=False, page_range=page_range)
        pages_count, items_count = self._document_counts(native_document)
        if not pages_count:
            raise DocumentParsingError("No extractable pages were found after native Docling parsing.")
        page_numbers = sorted(int(number) for number in (getattr(native_document, "pages", {}) or {}).keys())
        if not page_numbers and page_range is not None:
            page_numbers = list(range(page_range[0], page_range[1] + 1))
        candidates = [
            (page_number, self._quality_analyzer.assess(
                self._page_text(native_document, page_number, traverse_pictures=False), page_number=page_number,
            ))
            for page_number in page_numbers
        ]
        arabic_profile = sum(
            self._native_level(quality, arabic_profile=False) == NativeQualityLevel.HIGH_CONFIDENCE
            and quality.has_arabic_unicode
            for _, quality in candidates
        ) >= 2
        groups = {
            NativeQualityLevel.HIGH_CONFIDENCE: [],
            NativeQualityLevel.BORDERLINE: [],
            NativeQualityLevel.CORRUPTED: [],
        }
        decisions = []
        for page_number, quality in candidates:
            level = self._native_level(quality, arabic_profile=arabic_profile)
            decision = self._native_page_decision(native_document, page_number, quality, level)
            decisions.append(decision)
            # Keep legacy quality counts observable; candidate selection uses the
            # table-aware decision separately below.
            groups[level].append(page_number)
        return NativePreflight(
            pages_total=pages_count,
            items_count=items_count,
            native_good_page_numbers=groups[NativeQualityLevel.HIGH_CONFIDENCE],
            native_borderline_page_numbers=groups[NativeQualityLevel.BORDERLINE],
            native_bad_page_numbers=groups[NativeQualityLevel.CORRUPTED],
            detected_picture_count=len(getattr(native_document, "pictures", []) or []),
            detected_table_count=len(getattr(native_document, "tables", []) or []),
            native_analysis_duration_ms=round((time.perf_counter() - started) * 1000),
            page_decisions=decisions,
        )

    def _normalized_ocr_languages(self) -> str:
        """Use one canonical language value in both OCR options and cache identity."""
        languages = {
            language.strip().lower()
            for language in self._settings.rag_ocr_languages.split(",")
            if language.strip()
        }
        return ",".join(sorted(languages))

    def _cache_key(self, path: Path) -> dict[str, str]:
        return {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pipeline_version": self._settings.rag_extraction_pipeline_version,
            "ocr_engine": self._settings.rag_ocr_engine,
            "ocr_languages": self._normalized_ocr_languages(),
        }

    @staticmethod
    def _cache_key_digest(key: dict[str, str]) -> str:
        return hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _cache_directory(document_id: int) -> Path:
        # Keep cache data in the private backend tree.  Older previews wrote one
        # directory above backend; _load_cache reads that location once for a
        # backwards-compatible warm-cache migration.
        return Path(__file__).resolve().parents[2] / "cache" / "knowledge-base" / str(document_id)

    @staticmethod
    def _legacy_cache_directory(document_id: int) -> Path:
        return Path(__file__).resolve().parents[3] / "cache" / "knowledge-base" / str(document_id)

    def _load_cache(self, document_id: int, key: dict[str, str] | None) -> ParsedDocument | None:
        cache_directories = [self._cache_directory(document_id)]
        legacy_cache_dir = self._legacy_cache_directory(document_id)
        if legacy_cache_dir != cache_directories[0]:
            cache_directories.append(legacy_cache_dir)
        cache_dir = next((directory for directory in cache_directories if (directory / "manifest.json").is_file()), None)
        if cache_dir is None:
            self._cache_miss_reason = "not_found"
            logger.info("knowledge_extraction_cache_miss document_id=%s reason=not_found", document_id)
            return None

        manifest_path = cache_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_key = manifest.get("key")
            if not isinstance(manifest_key, dict):
                self._cache_miss_reason = "corrupted_manifest"
                logger.warning("knowledge_extraction_cache_miss document_id=%s reason=corrupted_manifest", document_id)
                return None
            if key is not None:
                mismatch = next((name for name in key if manifest_key.get(name) != key[name]), None)
                if mismatch is not None:
                    self._cache_miss_reason = f"{mismatch}_mismatch"
                    logger.info("knowledge_extraction_cache_miss document_id=%s reason=%s", document_id, self._cache_miss_reason)
                    return None
            from docling_core.types.doc.document import DoclingDocument
            pages = []
            for item in manifest["pages"]:
                doc = DoclingDocument.load_from_json(cache_dir / item["document"])
                quality = ExtractionQuality(**item["quality"])
                images = [PreservedImage(**image) for image in item.get("images", [])]
                pages.append(PageExtraction(item["page_number"], doc, item["extraction_mode"], quality, images=images))
            if not pages:
                raise ValueError("Cache manifest has no page payloads")
            page_issues = [
                PageExtractionIssue(
                    page_number=item["page_number"],
                    disposition=item["disposition"],
                    extraction_mode_attempted=item["extraction_mode_attempted"],
                    native_quality=ExtractionQuality(**item["native_quality"]),
                    ocr_quality=ExtractionQuality(**item["ocr_quality"]) if item.get("ocr_quality") else None,
                    failure_reasons=tuple(item.get("failure_reasons", [])),
                )
                for item in manifest.get("page_issues", [])
            ]
            logger.info("knowledge_extraction_cache_hit document_id=%s source=%s", document_id, "legacy" if cache_dir == legacy_cache_dir else "backend")
            cache_key = self._cache_key_digest(key) if key is not None else None
            return ParsedDocument(pages[0].document, manifest["pages_count"], manifest["items_count"], "page_adaptive", pages, {"cache_load": 0.0}, True, manifest.get("ocr_strategy", "targeted_pages"), float(manifest.get("ocr_required_page_ratio", 0.0)), cache_key, None, True, page_issues)
        except Exception:
            self._cache_miss_reason = "corrupted_payload"
            logger.warning("knowledge_extraction_cache_invalid document_id=%s", document_id, exc_info=True)
            return None

    def load_cached_extraction_only(self, path: Path, *, document_id: int) -> ParsedDocument | None:
        """Read a matching serialized extraction without converter, OCR, or PDF parsing."""
        if not path.is_file():
            return None
        return self._load_cache(document_id, self._cache_key(path))

    def load_cached_extraction_for_rebuild(self, *, document_id: int) -> ParsedDocument | None:
        """Load only the private cache payload; never open the source PDF.

        This is intentionally reserved for an explicit, selected-document
        materialization rebuild.  The document-id-scoped manifest is the trusted
        input and no converter, OCR engine, or source-file hash is invoked.
        """
        return self._load_cache(document_id, None)

    def _write_cache(self, document_id: int, key: dict[str, str], parsed: ParsedDocument) -> None:
        cache_dir = self._cache_directory(document_id)
        logger.info("knowledge_extraction_cache_write_attempted document_id=%s directory=%s", document_id, cache_dir.resolve())
        cache_dir.mkdir(parents=True, exist_ok=True)
        pages = []
        for page in parsed.page_extractions:
            filename = f"page-{page.page_number}.json"
            page.document.save_as_json(cache_dir / filename)
            pages.append({"page_number": page.page_number, "document": filename, "extraction_mode": page.extraction_mode, "quality": page.quality.__dict__, "images": [image.__dict__ for image in page.images]})
        page_issues = [
            {
                "page_number": issue.page_number,
                "disposition": issue.disposition,
                "extraction_mode_attempted": issue.extraction_mode_attempted,
                "native_quality": issue.native_quality.__dict__,
                "ocr_quality": issue.ocr_quality.__dict__ if issue.ocr_quality else None,
                "failure_reasons": list(issue.failure_reasons),
            }
            for issue in parsed.page_issues
        ]
        (cache_dir / "manifest.json").write_text(json.dumps({"key": key, "pages_count": parsed.pages_count, "items_count": parsed.items_count, "ocr_strategy": parsed.ocr_strategy, "ocr_required_page_ratio": parsed.ocr_required_page_ratio, "pages": pages, "page_issues": page_issues}, ensure_ascii=False), encoding="utf-8")
        logger.info("knowledge_extraction_cache_write_success document_id=%s manifest_exists=%s payload_exists=%s", document_id, (cache_dir / "manifest.json").is_file(), all((cache_dir / page["document"]).is_file() for page in pages))

    def repair_page_with_ocr(
        self,
        path: Path,
        *,
        document_id: int,
        page_number: int,
        native_quality: ExtractionQuality,
        native_corruption_reasons: tuple[str, ...] = (),
    ) -> PageExtraction | None:
        """One late OCR attempt for a page whose final native chunk proved corrupt."""
        if not self._settings.rag_ocr_enabled or self._settings.rag_ocr_engine.lower() != "easyocr":
            return None
        try:
            ocr_document = self._convert(path, use_ocr=True, page_range=(page_number, page_number))
            ocr_text = self._page_text(ocr_document, page_number, traverse_pictures=True)
            ocr_quality = self._quality_analyzer.assess(ocr_text, page_number=page_number)
            self._log_quality(document_id, page_number, "ocr", ocr_quality)
            ocr_is_equally_cleaner = (
                ocr_quality.quality_passed
                and ocr_quality.quality_score == native_quality.quality_score
                and bool(native_corruption_reasons)
                and not ocr_quality.failure_reasons
            )
            if not ocr_quality.quality_passed or (
                ocr_quality.quality_score < native_quality.quality_score
                or (ocr_quality.quality_score == native_quality.quality_score and not ocr_is_equally_cleaner)
            ):
                logger.warning(
                    "late_page_repair_rejected document_id=%s page=%s native_score=%.2f ocr_score=%.2f",
                    document_id, page_number, native_quality.quality_score, ocr_quality.quality_score,
                )
                return None
            normalized = self._geometric_reading_order(ocr_document, page_number)
            return PageExtraction(page_number, ocr_document, "full_page_ocr", ocr_quality, reading_order_text=normalized, raw_docling_text=ocr_text, images=self._extract_images(ocr_document, document_id, page_number))
        except DocumentParsingError:
            logger.exception("late_page_repair_failed document_id=%s page=%s", document_id, page_number)
            return None
        except Exception:
            logger.exception("late_page_repair_failed document_id=%s page=%s", document_id, page_number)
            return None

    def _build_converter(self, *, use_ocr: bool) -> Any:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions, OcrMode, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = use_ocr
        options.do_table_structure = True
        options.generate_picture_images = True
        if use_ocr:
            options.ocr_options = EasyOcrOptions(
                lang=self._normalized_ocr_languages().split(","),
                mode=OcrMode.FULL_PAGE,
            )
        return DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})

    def _convert_targeted_ocr(self, path: Path, page_number: int) -> Any:
        """Bound a single page attempt; the existing partial-ingestion policy decides its outcome."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docling-ocr-page")
        future = executor.submit(self._convert, path, use_ocr=True, page_range=(page_number, page_number))
        try:
            return future.result(timeout=self._settings.rag_ocr_page_timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            raise OcrPageTimeoutError("Targeted OCR page conversion timed out.") from error
        finally:
            # Do not wait for a hung native extension; it must not block the worker's page loop.
            executor.shutdown(wait=False, cancel_futures=True)

    def _convert(self, path: Path, *, use_ocr: bool, page_range: tuple[int, int] | None) -> Any:
        attempts = max(0, self._settings.rag_model_download_max_retries) + 1
        for attempt in range(1, attempts + 1):
            try:
                if self._converter_factory is not None:
                    return self._converter_factory(path, use_ocr, page_range)
                converter_started = time.perf_counter()
                converter = self._active_preflight_converter if not use_ocr and self._active_preflight_converter is not None else self._converters.get(use_ocr)
                if converter is None:
                    converter = self._build_converter(use_ocr=use_ocr)
                    if not use_ocr and self._active_preflight_converter is None:
                        self._converters[False] = converter
                    elif use_ocr:
                        self._converters[True] = converter
                converter_init_ms = round((time.perf_counter() - converter_started) * 1000, 2)
                kwargs = {"source": path}
                if page_range is not None: kwargs["page_range"] = page_range
                conversion_started = time.perf_counter()
                converted = converter.convert(**kwargs).document
                if self._preflight_lifecycle is not None and not use_ocr:
                    self._preflight_lifecycle["conversion_calls"] += 1
                    self._preflight_lifecycle["page_ranges_processed"].append(page_range)
                conversion_ms = round((time.perf_counter() - conversion_started) * 1000, 2)
                self._conversion_timings.append({"mode": "ocr" if use_ocr else "native", "page": page_range[0] if page_range else None, "converter_init_ms": converter_init_ms, "conversion_ms": conversion_ms})
                logger.info("docling_conversion_timing mode=%s page=%s converter_init_ms=%s conversion_ms=%s", "ocr" if use_ocr else "native", page_range[0] if page_range else None, converter_init_ms, conversion_ms)
                return converted
            except Exception as error:
                if self._is_transient_model_error(error):
                    if attempt < attempts:
                        logger.warning("docling_model_resource_retry mode=%s attempt=%s/%s error_type=%s", "ocr" if use_ocr else "native", attempt, attempts, type(error).__name__)
                        self._converters.pop(use_ocr, None)
                        time.sleep(self._settings.rag_model_download_retry_seconds * attempt)
                        continue
                    raise ModelResourceUnavailableError("Required Docling model resource is temporarily unavailable.") from error
                mode = "full-page OCR" if use_ocr else "PDF text extraction"
                raise DocumentParsingError(f"Docling {mode} could not parse this PDF.") from error
        raise AssertionError("unreachable")

    @staticmethod
    def _is_transient_model_error(error: Exception) -> bool:
        """Keep network/model setup failures separate from document parse failures."""
        chain: list[BaseException] = []
        current: BaseException | None = error
        while current is not None and current not in chain:
            chain.append(current)
            current = current.__cause__ or current.__context__
        names = {type(item).__name__ for item in chain}
        message = " ".join(str(item).lower() for item in chain)
        transient_names = {"RemoteProtocolError", "ConnectTimeout", "ReadTimeout", "PoolTimeout", "ConnectError", "NetworkError"}
        return bool(names.intersection(transient_names)) and any(marker in message for marker in ("huggingface", "snapshot_download", "server disconnected", "connection", "timeout"))

    @staticmethod
    def _document_counts(document: Any) -> tuple[int, int]:
        return len(getattr(document, "pages", {}) or {}), sum(1 for _ in document.iterate_items())

    @staticmethod
    def _page_text(document: Any, page_number: int, *, traverse_pictures: bool) -> str:
        exporter = getattr(document, "export_to_text", None)
        if callable(exporter):
            try:
                return str(exporter(page_no=page_number, traverse_pictures=traverse_pictures) or "")
            except TypeError:
                return str(exporter(traverse_pictures=traverse_pictures) or "")
        return "\n".join(str(getattr(item[0] if isinstance(item, tuple) else item, "text", "")) for item in document.iterate_items())

    def _is_structurally_blank_page(self, document: Any, page_number: int) -> bool:
        """Skip only pages Docling itself reports as structurally empty, never empty OCR text alone."""
        if self._page_text(document, page_number, traverse_pictures=True).strip():
            return False
        try:
            entries = document.iterate_items(page_no=page_number, traverse_pictures=True)
        except TypeError:
            entries = document.iterate_items()
        for entry in entries:
            item = entry[0] if isinstance(entry, tuple) else entry
            provenance = next(iter(getattr(item, "prov", []) or []), None)
            item_page = getattr(provenance, "page_no", None)
            if item_page in (None, page_number):
                return False
        return True

    @staticmethod
    def _bbox_dict(item: Any) -> dict[str, float] | None:
        prov = next(iter(getattr(item, "prov", []) or []), None)
        bbox = getattr(prov, "bbox", None)
        if bbox is None:
            return None
        return {key: float(getattr(bbox, key)) for key in ("l", "t", "r", "b") if hasattr(bbox, key)} or None

    @staticmethod
    def _top_left_bbox(document: Any, item: Any) -> dict[str, float] | None:
        """Canonicalize Docling provenance before any visual reading-order sorting."""
        prov = next(iter(getattr(item, "prov", []) or []), None)
        bbox = getattr(prov, "bbox", None)
        if bbox is None:
            return None
        page = (getattr(document, "pages", {}) or {}).get(getattr(prov, "page_no", None))
        height = getattr(getattr(page, "size", None), "height", None)
        normalized = bbox.to_top_left_origin(page_height=float(height)) if height is not None and hasattr(bbox, "to_top_left_origin") else bbox
        l, t, r, b = (float(getattr(normalized, key)) for key in ("l", "t", "r", "b"))
        return {"l": min(l, r), "r": max(l, r), "t": min(t, b), "b": max(t, b)}

    def _geometric_reading_order(self, document: Any, page_number: int) -> str:
        """Order OCR text visually: lines top-to-bottom and Arabic blocks RTL per line."""
        blocks: list[tuple[float, float, float, str]] = []
        try:
            entries = document.iterate_items(page_no=page_number, traverse_pictures=True)
        except TypeError:
            entries = document.iterate_items()
        for entry in entries:
            item = entry[0] if isinstance(entry, tuple) else entry
            text = str(getattr(item, "text", "")).strip()
            bbox = self._top_left_bbox(document, item)
            prov = next(iter(getattr(item, "prov", []) or []), None)
            if not text or not bbox or getattr(prov, "page_no", None) != page_number:
                continue
            blocks.append((bbox.get("t", 0.0), bbox.get("l", 0.0), bbox.get("r", 0.0), text))
        lines: list[list[tuple[float, float, float, str]]] = []
        for block in sorted(blocks, key=lambda value: (value[0], value[1])):
            if not lines or abs(lines[-1][0][0] - block[0]) > 12:
                lines.append([block])
            else:
                lines[-1].append(block)
        ordered: list[str] = []
        for line in lines:
            arabic = sum(sum("\u0600" <= char <= "\u08ff" for char in block[3]) for block in line)
            latin = sum(sum(char.isascii() and char.isalpha() for char in block[3]) for block in line)
            ordered.extend(block[3] for block in sorted(line, key=lambda value: value[2] if arabic > latin else value[1], reverse=arabic > latin))
        return "\n".join(ordered)

    def _apply_geometric_reading_order(self, document: Any) -> None:
        """Reorder Docling sibling references in-place; never flatten text or reverse characters."""
        def item_text(item: Any) -> str:
            return str(getattr(item, "text", "") or "")

        def is_arabic_dominant(items: list[tuple[Any, dict[str, float]]]) -> bool:
            text = " ".join(item_text(item) for item, _ in items)
            arabic = sum("\u0600" <= char <= "\u08ff" for char in text)
            latin = sum(char.isascii() and char.isalpha() for char in text)
            return arabic > latin

        def sort_children(parent: Any) -> None:
            refs = list(getattr(parent, "children", []) or [])
            resolved: list[tuple[Any, Any, dict[str, float], int]] = []
            for position, ref in enumerate(refs):
                try:
                    child = ref.resolve(document)
                    prov = next(iter(getattr(child, "prov", []) or []), None)
                    bbox = self._top_left_bbox(document, child)
                    page = getattr(prov, "page_no", None)
                    if bbox and isinstance(page, int):
                        resolved.append((ref, child, bbox, page))
                except Exception:
                    continue
            if len(resolved) > 1:
                by_page: dict[int, list[tuple[Any, Any, dict[str, float], int]]] = {}
                for value in resolved:
                    by_page.setdefault(value[3], []).append(value)
                ordered_refs: list[Any] = []
                for page in sorted(by_page):
                    page_items = sorted(by_page[page], key=lambda value: value[2]["t"])
                    lines: list[list[tuple[Any, Any, dict[str, float], int]]] = []
                    for value in page_items:
                        if not lines or abs(lines[-1][0][2]["t"] - value[2]["t"]) > 12:
                            lines.append([value])
                        else:
                            lines[-1].append(value)
                    for line in lines:
                        rtl = is_arabic_dominant([(value[1], value[2]) for value in line])
                        ordered_refs.extend(value[0] for value in sorted(line, key=lambda value: value[2]["r"] if rtl else value[2]["l"], reverse=rtl))
                if len(ordered_refs) == len(refs):
                    parent.children = ordered_refs
            for ref in list(getattr(parent, "children", []) or []):
                try:
                    sort_children(ref.resolve(document))
                except Exception:
                    continue

        sort_children(getattr(document, "body", None))

    def _log_ocr_granularity(self, document_id: int | None, document: Any, page_number: int) -> None:
        """Audit retained OCR cells without assuming that Docling exposes pipeline internals."""
        page = (getattr(document, "pages", {}) or {}).get(page_number)
        cells: list[Any] = []
        for attribute in ("word_cells", "ocr_cells", "textline_cells", "cells"):
            value = getattr(page, attribute, None)
            if value:
                cells = list(value)
                break
        item_samples = []
        try:
            entries = document.iterate_items(page_no=page_number, traverse_pictures=True)
        except TypeError:
            entries = document.iterate_items()
        for entry in entries:
            item = entry[0] if isinstance(entry, tuple) else entry
            prov = next(iter(getattr(item, "prov", []) or []), None)
            if prov is not None and getattr(prov, "page_no", page_number) != page_number:
                continue
            text = str(getattr(item, "text", "")).strip()
            if text:
                item_samples.append({"text": text, "bbox": self._top_left_bbox(document, item)})
            if len(item_samples) == 5:
                break
        cell_samples = []
        for cell in cells[:12]:
            cell_samples.append({"text": str(getattr(cell, "text", "")), "bbox": str(getattr(cell, "bbox", getattr(cell, "rect", None))), "confidence": getattr(cell, "confidence", None)})
        logger.info(
            "ocr_granularity_audit document_id=%s page=%s retained_cell_count=%s retained_cells=%s docling_text_items=%s",
            document_id, page_number, len(cells), cell_samples, item_samples,
        )

    def _extract_images(self, document: Any, document_id: int | None, page_number: int) -> list[PreservedImage]:
        if document_id is None:
            return []
        image_dir = Path(__file__).resolve().parents[3] / "uploads" / "knowledge-base" / "images" / str(document_id)
        images: list[PreservedImage] = []
        text_blocks = self._page_text_blocks(document, page_number)
        for index, picture in enumerate(getattr(document, "pictures", []) or []):
            prov = next(iter(getattr(picture, "prov", []) or []), None)
            if getattr(prov, "page_no", None) != page_number:
                continue
            image_id = f"knowledge-document:{document_id}:page:{page_number}:image:{len(images)}"
            output = image_dir / f"page-{page_number}-image-{len(images)}.png"
            try:
                crop = picture.get_image(document)
                if crop is not None:
                    image_dir.mkdir(parents=True, exist_ok=True)
                    crop.save(output, format="PNG")
                    stored_path: str | None = str(output)
                else:
                    stored_path = None
            except Exception:
                logger.warning("picture_asset_not_available document_id=%s page=%s picture=%s", document_id, page_number, index)
                stored_path = None
            caption = picture.caption_text(document).strip() if hasattr(picture, "caption_text") else ""
            bbox = self._bbox_dict(picture)
            nearby_text = self._nearby_text(bbox, text_blocks, explicit_caption=caption)
            image_role = self._classify_image_role(bbox, text_blocks, caption, nearby_text)
            images.append(PreservedImage(
                image_id, document_id, page_number, bbox, stored_path, caption or None, nearby_text,
                image_role=image_role,
            ))
        return images

    def _page_text_blocks(self, document: Any, page_number: int) -> list[tuple[dict[str, float], str]]:
        blocks: list[tuple[dict[str, float], str]] = []
        for entry in document.iterate_items():
            item = entry[0] if isinstance(entry, tuple) else entry
            text = str(getattr(item, "text", "")).strip()
            bbox = self._bbox_dict(item)
            prov = next(iter(getattr(item, "prov", []) or []), None)
            if text and bbox and getattr(prov, "page_no", None) == page_number:
                blocks.append((bbox, text))
        return blocks

    @staticmethod
    def _nearby_text(image_bbox: dict[str, float] | None, blocks: list[tuple[dict[str, float], str]], *, explicit_caption: str) -> str | None:
        """Return only a geometrically close text block; explicit Docling captions win."""
        if explicit_caption:
            return None
        if not image_bbox:
            return None
        image_center_x = (image_bbox.get("l", 0.0) + image_bbox.get("r", 0.0)) / 2
        image_center_y = (image_bbox.get("t", 0.0) + image_bbox.get("b", 0.0)) / 2
        image_height = abs(image_bbox.get("b", 0.0) - image_bbox.get("t", 0.0))
        candidates: list[tuple[float, str]] = []
        for bbox, text in blocks:
            center_x = (bbox.get("l", 0.0) + bbox.get("r", 0.0)) / 2
            center_y = (bbox.get("t", 0.0) + bbox.get("b", 0.0)) / 2
            vertical_gap = abs(center_y - image_center_y)
            horizontal_gap = abs(center_x - image_center_x)
            # Captions normally sit immediately above/below; reject distant page text.
            if vertical_gap > max(90.0, image_height * 1.5) or horizontal_gap > 260.0:
                continue
            candidates.append((vertical_gap * 2 + horizontal_gap * 0.35, text))
        if not candidates:
            return None
        distance, text = min(candidates, key=lambda candidate: candidate[0])
        return text if distance <= max(130.0, image_height * 1.25) else None

    @staticmethod
    def _classify_image_role(
        bbox: dict[str, float] | None,
        blocks: list[tuple[dict[str, float], str]],
        caption: str,
        nearby_text: str | None,
    ) -> str:
        """A conservative pre-VLM classifier. It never removes source assets."""
        if caption:
            return "pedagogical_visual"
        if not bbox:
            return "unknown"
        area = abs(bbox.get("r", 0.0) - bbox.get("l", 0.0)) * abs(bbox.get("b", 0.0) - bbox.get("t", 0.0))
        text_extent = max((abs(block_bbox.get("r", 0.0) - block_bbox.get("l", 0.0)) * abs(block_bbox.get("b", 0.0) - block_bbox.get("t", 0.0)) for block_bbox, _ in blocks), default=0.0)
        activity_context = bool(nearby_text and re.search(r"(تمرين|نشاط|اقرأ|أجب|صل|ضع|exercice|activité|read|answer)", nearby_text, re.I))
        if activity_context and area >= 900:
            return "pedagogical_visual"
        if area <= 400 or (text_extent and area / text_extent < 0.08 and not nearby_text):
            return "decorative_or_layout"
        return "unknown"

    @staticmethod
    def _hard_corruption_detected(quality: ExtractionQuality) -> bool:
        return (
            quality.glyph_name_count > 0
            or quality.replacement_character_count > 0
            or (quality.suspicious_ascii_sequence_count >= 2 and (quality.digit_inside_word_ratio > 0.15 or quality.symbol_inside_word_ratio > 0.15))
        )

    def _table_evidence(self, document: Any, page_number: int) -> dict[str, Any]:
        page_tables = []
        for table in getattr(document, "tables", []) or []:
            provenance = next(iter(getattr(table, "prov", []) or []), None)
            if getattr(provenance, "page_no", None) == page_number:
                page_tables.append(table)
        non_empty_cells = meaningful_cells = 0
        for table in page_tables:
            try:
                frame = table.export_to_dataframe(document)
                values = frame.fillna("").astype(str).to_numpy().flatten().tolist()
            except Exception:
                continue
            for value in values:
                text = value.strip()
                if not text:
                    continue
                non_empty_cells += 1
                # CEFR labels such as A1, B2, C1 and Pré-A1 are meaningful
                # table content, even when prose-oriented text scoring is low.
                if any(char.isalpha() or char.isdigit() for char in text):
                    meaningful_cells += 1
        usable = bool(page_tables) and non_empty_cells >= 2 and meaningful_cells / max(1, non_empty_cells) >= 0.60
        return {
            "table_count": len(page_tables),
            "table_non_empty_cells": non_empty_cells,
            "table_meaningful_cell_ratio": round(meaningful_cells / max(1, non_empty_cells), 3),
            "table_structurally_usable": usable,
        }

    def _native_page_decision(self, document: Any, page_number: int, quality: ExtractionQuality, text_quality_class: str) -> dict[str, Any]:
        table = self._table_evidence(document, page_number)
        hard_corruption = self._hard_corruption_detected(quality)
        table_native_usable = table["table_structurally_usable"] and not hard_corruption
        if text_quality_class == NativeQualityLevel.HIGH_CONFIDENCE:
            native_usable, ocr_candidate, reason = True, False, "high_confidence_native_text"
        elif table_native_usable:
            native_usable, ocr_candidate, reason = True, False, "borderline_but_structurally_usable_table"
        elif hard_corruption:
            native_usable, ocr_candidate, reason = False, True, "hard_corruption"
        else:
            native_usable, ocr_candidate, reason = False, True, "native_text_quality_uncertain"
        return {
            "page": page_number,
            "text_quality_class": text_quality_class,
            "quality_score": quality.quality_score,
            "hard_corruption_detected": hard_corruption,
            **table,
            "native_usable": native_usable,
            "ocr_candidate": ocr_candidate,
            "ocr_candidate_reason": reason,
            "languages": list(quality.languages_detected),
            "failure_reasons": list(quality.failure_reasons),
        }

    def _native_level(self, quality: ExtractionQuality, *, arabic_profile: bool) -> str:
        hard_corruption = self._hard_corruption_detected(quality)
        script_mismatch = arabic_profile and not quality.has_arabic_unicode and "latin" in quality.languages_detected and quality.meaningful_word_ratio < 0.7
        if hard_corruption or quality.quality_score < self._settings.rag_native_borderline_score:
            return NativeQualityLevel.CORRUPTED
        if quality.quality_score >= self._settings.rag_native_high_confidence_score and not script_mismatch:
            return NativeQualityLevel.HIGH_CONFIDENCE
        return NativeQualityLevel.BORDERLINE

    @staticmethod
    def _log_quality(document_id: int | None, page: int, mode: str, quality: ExtractionQuality, *, classification: str | None = None) -> None:
        logger.info("page_%s_quality document_id=%s page=%s score=%.2f classification=%s languages=%s passed=%s reasons=%s", mode, document_id, page, quality.quality_score, classification or "candidate", ",".join(quality.languages_detected) or "unknown", quality.quality_passed, ",".join(quality.failure_reasons) or "none")

    @staticmethod
    def _log_selected(document_id: int | None, page: int, mode: str, native_score: float, ocr_score: float | None, reason: str) -> None:
        logger.info("page_extraction_selected document_id=%s page=%s selected=%s native_score=%.2f ocr_score=%s reason=%s", document_id, page, mode, native_score, f"{ocr_score:.2f}" if ocr_score is not None else "none", reason)

    @staticmethod
    def _quality_error(page: int, quality: ExtractionQuality, detail: str) -> DocumentParsingError:
        reasons = ",".join(quality.failure_reasons) or "quality_score"
        return DocumentParsingError(f"Page {page} extraction quality failed ({reasons}): {detail}.")
