from app.services.document_chunker import (
    ChunkQualityCategory,
    ChunkMetadataBuilder,
    DocumentChunker,
)
from app.services.document_cleaner import DocumentCleaner
from app.services.document_parser_service import (
    DocumentParserService,
    DocumentParsingError,
    ModelResourceUnavailableError,
    OcrPageTimeoutError,
    ExtractionQualityValidator,
    PageExtraction,
    PageExtractionIssue,
    ParsedDocument,
    NativeQualityLevel,
    PreservedImage,
    NativePreflight,
)
from app.services.knowledge_ingestion_service import KnowledgeIngestionError, KnowledgeIngestionService, ParsePreview
from app.services.knowledge_preflight_service import KnowledgePreflightService
from app.services.document_chunk import DocumentChunk
from app.services.worksheet_structure import WorksheetStructureBuilder
from types import SimpleNamespace
from dataclasses import replace
import json
import inspect
import asyncio
from io import BytesIO
import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers
from app.api.dependencies import require_admin
from app.api.routes.admin import preflight_knowledge_document, stored_pdf_is_valid


def _single_page_pdf_bytes() -> bytes:
    parts = [
        b"%PDF-1.4\n",
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Length 0 >>\nstream\n\nendstream\nendobj\n",
    ]
    offsets: list[int] = []
    payload = b""
    for part in parts:
        offsets.append(len(payload)); payload += part
    xref = b"xref\n0 5\n0000000000 65535 f \n" + b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return payload + xref + b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n" + str(len(payload)).encode() + b"\n%%EOF\n"


def test_cleaner_keeps_arabic_unicode_and_removes_controls():
    cleaned = DocumentCleaner().clean("  العربية\x00  والفرنسية Français\r\n\r\n\r\nA1  ")
    assert cleaned == "العربية والفرنسية Français\n\nA1"


def test_stored_pdf_validation_accepts_a_small_valid_single_page_pdf(tmp_path):
    stored = tmp_path / "small-valid.pdf"
    uploaded = _single_page_pdf_bytes()
    stored.write_bytes(uploaded)

    assert stored.read_bytes() == uploaded
    assert stored_pdf_is_valid(stored)


def test_stored_pdf_validation_rejects_malformed_and_incomplete_copies(tmp_path):
    malformed = tmp_path / "malformed.pdf"
    malformed.write_bytes(b"%PDF-1.4\nnot a PDF")
    incomplete = tmp_path / "incomplete.pdf"
    incomplete.write_bytes(_single_page_pdf_bytes()[:77])

    assert not stored_pdf_is_valid(malformed)
    assert not stored_pdf_is_valid(incomplete)


def test_invalid_stored_upload_is_removed_and_never_added_to_the_database(tmp_path, monkeypatch):
    import app.api.routes.admin as admin_routes

    class Db:
        added = []
        rolled_back = False
        def add(self, value): self.added.append(value)
        def commit(self): pytest.fail("Invalid stored PDF must not be committed")
        def refresh(self, _value): pytest.fail("Invalid stored PDF must not be refreshed")
        def rollback(self): self.rolled_back = True

    monkeypatch.setattr(admin_routes, "UPLOAD_DIRECTORY", tmp_path)
    upload = UploadFile(filename="broken.pdf", file=BytesIO(b"%PDF-1.4\nnot a PDF"), headers=Headers({"content-type": "application/pdf"}))
    db = Db()

    with pytest.raises(HTTPException) as error:
        asyncio.run(admin_routes.create_knowledge_document(
            file=upload, title=None, document_type=None, language=None, cefr_level=None,
            skill=None, source=None, description=None, admin=SimpleNamespace(id=1), db=db,
        ))

    assert error.value.status_code == 422
    assert error.value.detail == "Uploaded PDF could not be validated"
    assert db.added == [] and db.rolled_back
    assert list(tmp_path.glob("*.pdf")) == []


def test_contextual_text_preserves_hierarchy_without_rewriting_original_text():
    original = "يمكن للمتعلم أن يتحدث عن نفسه."
    contextual = ChunkMetadataBuilder.contextual_text("CECRL", ["Production orale", "A1"], original)
    assert original in contextual
    assert "Section: Production orale > A1" in contextual


def test_image_placeholders_are_not_in_embedding_text():
    contextual = ChunkMetadataBuilder.contextual_text("Workbook", [], "اقرأ النص <!-- image --> ثم أجب")
    assert "<!-- image -->" not in contextual
    assert "اقرأ النص" in contextual


def test_worksheet_structure_orders_arabic_blocks_and_keeps_option_lists_without_reversing_words():
    class Bbox:
        def __init__(self, l, t, r, b): self.l, self.t, self.r, self.b = l, t, r, b
    class Text:
        def __init__(self, text, l, t, r, b, label="text"):
            self.text, self.label = text, label
            self.prov = [SimpleNamespace(page_no=1, bbox=Bbox(l, t, r, b))]
    # Higher t values are first visually on this bottom-left-origin fixture.
    document = SimpleNamespace(
        pages={},
        texts=[
            Text("في الغابة", 10, 80, 70, 90),
            Text("العودة إلى المدرسة", 80, 80, 170, 90),
            Text("ألون عنوان النص الجديد.", 260, 110, 420, 120),
            Text("حكيمة الجبل", 250, 80, 330, 90),
            Text("في بيت أمي ميمونة", 350, 80, 470, 90),
            Text("3 أربط كل عبارة بما يناسبها.", 260, 50, 470, 60, "section_header"),
            Text("الحديقة", 360, 25, 440, 35),
            Text("مزهرة", 80, 25, 150, 35),
            Text("البيت", 360, 10, 440, 20),
            Text("جميل", 80, 10, 150, 20),
        ],
    )
    sections = WorksheetStructureBuilder().sections(document, 1, has_image=True)
    rendered = "\n".join(section.text for section in sections)
    assert "ألون عنوان النص الجديد." in rendered
    assert "الاختيارات:" in rendered
    assert "- في بيت أمي ميمونة" in rendered
    assert "تمرين 3" in rendered
    assert "العمود الأيمن:" in rendered and "العمود الأيسر:" in rendered
    assert "<!-- image -->" not in rendered
    assert all(section.has_image for section in sections)


def test_worksheet_structure_keeps_mixed_text_and_leaves_simple_page_to_hybrid_chunker():
    class Bbox:
        def __init__(self, l, t, r, b): self.l, self.t, self.r, self.b = l, t, r, b
    class Text:
        def __init__(self, text, l):
            self.text, self.label = text, "text"
            self.prov = [SimpleNamespace(page_no=1, bbox=Bbox(l, 20, l + 40, 30))]
    document = SimpleNamespace(pages={}, texts=[Text("A1 Bonjour", 10), Text("العربية", 80)])
    assert WorksheetStructureBuilder().sections(document, 1, has_image=False) == []


def test_worksheet_uses_word_geometry_for_rtl_line_without_reversing_words():
    class Bbox:
        def __init__(self, l, t, r, b): self.l, self.t, self.r, self.b = l, t, r, b
    class Cell:
        def __init__(self, text, l): self.text, self.bbox = text, Bbox(l, 10, l + 30, 20)
    page = SimpleNamespace(word_cells=[
        Cell("ألون", 300), Cell("عنوان", 220), Cell("النص", 140), Cell("الجديد.", 50),
    ])
    document = SimpleNamespace(pages={1: page}, texts=[SimpleNamespace()])
    blocks = WorksheetStructureBuilder().blocks(document, 1)
    assert [block.text for block in blocks] == ["ألون عنوان النص الجديد."]


def test_worksheet_word_geometry_preserves_arabic_number_and_marks_vision_requirement():
    class Bbox:
        def __init__(self, l, y): self.l, self.t, self.r, self.b = l, y, l + 30, y + 10
    class Cell:
        def __init__(self, text, l, y): self.text, self.bbox = text, Bbox(l, y)
    page = SimpleNamespace(word_cells=[
        Cell("التمرين", 300, 10), Cell("3", 240, 10), Cell("ألون", 180, 10), Cell("الرسم", 100, 10),
        Cell("اختيار", 250, 40), Cell("أول", 150, 40), Cell("اختيار", 250, 70), Cell("ثان", 150, 70),
    ])
    document = SimpleNamespace(pages={1: page}, texts=[])
    sections = WorksheetStructureBuilder().sections(document, 1, has_image=True)
    assert sections[0].text.startswith("تمرين 3\nالتمرين 3 ألون الرسم")
    assert sections[0].requires_vision and sections[0].structural_quality == "structured"


def test_page_metadata_uses_min_and_max_page_numbers():
    class Provenance:
        def __init__(self, page_no): self.page_no = page_no
    class Item:
        def __init__(self, page_no): self.prov = [Provenance(page_no)]
    assert ChunkMetadataBuilder.pages_from_doc_items([Item(43), Item(42)]) == (42, 43)


def token_count(text: str) -> int:
    return len(text.split())


def test_normal_paragraph_stays_intact_within_token_budget():
    chunker = DocumentChunker(max_tokens=12, tokenizer_name="test")
    contextualize = lambda text: f"Document: CECRL\n\n{text}"
    assert chunker._split_to_budget("A1 learners can introduce themselves.", contextualize, token_count, "text") == ["A1 learners can introduce themselves."]


def test_oversized_paragraph_preserves_all_text_and_respects_budget():
    chunker = DocumentChunker(max_tokens=10, tokenizer_name="test")
    contextualize = lambda text: f"Document: CECRL\n\n{text}"
    original = "First pedagogical sentence is preserved. Second pedagogical sentence is also preserved. Third sentence remains available."
    result = chunker._split_to_budget(original, contextualize, token_count, "text")
    assert len(result) > 1
    assert all(token_count(contextualize(part)) <= 10 for part in result)
    assert " ".join(result).replace(" ", "") == original.replace(" ", "")


def test_oversized_table_repeats_header_and_preserves_rows():
    chunker = DocumentChunker(max_tokens=18, tokenizer_name="test")
    contextualize = lambda text: f"Document: CECRL\n\n{text}"
    table = "| Level | Oral production |\n| --- | --- |\n| A1 | Can introduce self clearly |\n| A2 | Can describe familiar activities clearly |\n| B1 | Can present an opinion with reasons |"
    result = chunker._split_to_budget(table, contextualize, token_count, "table")
    assert len(result) > 1
    assert all(part.startswith("| Level | Oral production |") for part in result)
    assert all(token_count(contextualize(part)) <= 18 for part in result)
    assert all(value in " ".join(result) for value in ("A1", "A2", "B1", "introduce", "familiar", "activities", "opinion", "reasons"))


def test_arabic_overflow_preserves_unicode_and_metadata_context():
    chunker = DocumentChunker(max_tokens=11, tokenizer_name="test")
    original = "يمكن للمتعلم أن يعرّف بنفسه. ويمكنه أن يصف أنشطته اليومية. ويمكنه أن يتحدث عن اهتماماته."
    contextualize = lambda text: ChunkMetadataBuilder.contextual_text("الإطار الأوروبي", ["التعبير الشفهي", "A1"], text)
    result = chunker._split_to_budget(original, contextualize, token_count, "text")
    assert all(token_count(contextualize(part)) <= 11 for part in result)
    assert "يمكن" in " ".join(result)
    metadata = {"document_id": 7, "page_start": 42, "page_end": 43, "headings": ["التعبير الشفهي", "A1"], "source": "CECRL"}
    assert metadata["document_id"] == 7 and metadata["page_start"] == 42 and metadata["headings"][-1] == "A1"


def test_unicode_arabic_and_mixed_arabic_english_text_do_not_require_ocr():
    quality = ExtractionQualityValidator().assess("العربية لغة جميلة مع English exercises.")
    assert quality.has_arabic_unicode
    assert not quality.needs_ocr_fallback


def test_clean_cefr_table_is_native_usable_even_when_text_quality_is_borderline():
    import pandas as pd

    class Table:
        prov = [SimpleNamespace(page_no=1)]
        @staticmethod
        def export_to_dataframe(_document):
            return pd.DataFrame([["Level", "Descriptor"], ["A1", "Can understand familiar words"], ["C2", "Can understand virtually everything"]])

    parser = DocumentParserService(converter_factory=lambda *_args: None)
    quality = replace(ExtractionQualityValidator().assess("A1 B2 C1 C2 Pré-A1", page_number=1), quality_score=0.70, quality_passed=True)
    decision = parser._native_page_decision(SimpleNamespace(tables=[Table()]), 1, quality, NativeQualityLevel.BORDERLINE)

    assert decision["table_structurally_usable"]
    assert decision["native_usable"]
    assert not decision["ocr_candidate"]
    assert decision["ocr_candidate_reason"] == "borderline_but_structurally_usable_table"


def test_corrupted_table_text_remains_an_ocr_candidate():
    import pandas as pd

    class Table:
        prov = [SimpleNamespace(page_no=1)]
        @staticmethod
        def export_to_dataframe(_document):
            return pd.DataFrame([["A1", "C2"], ["Personal", "Educational"]])

    parser = DocumentParserService(converter_factory=lambda *_args: None)
    quality = ExtractionQualityValidator().assess("alefisolated meemmedial thalfinal", page_number=1)
    decision = parser._native_page_decision(SimpleNamespace(tables=[Table()]), 1, quality, NativeQualityLevel.CORRUPTED)

    assert decision["hard_corruption_detected"]
    assert not decision["native_usable"]
    assert decision["ocr_candidate"]
    assert decision["ocr_candidate_reason"] == "hard_corruption"


def test_preflight_excludes_structurally_usable_borderline_tables_from_ocr_candidates(tmp_path):
    class Db:
        def commit(self): pass

    class Parser:
        def preflight_pdf(self, _path, *, page_range=None):
            start, end = page_range
            assert (start, end) == (1, 2)
            return NativePreflight(
                2, 2, [1], [2], [], 0, 1, 1,
                page_decisions=[
                    {"page": 1, "ocr_candidate": False, "ocr_candidate_reason": "high_confidence_native_text"},
                    {"page": 2, "ocr_candidate": False, "ocr_candidate_reason": "borderline_but_structurally_usable_table"},
                ],
            )

    pdf = tmp_path / "table-preflight.pdf"; pdf.write_bytes(b"data")
    document = SimpleNamespace(id=90, file_path=str(pdf), preflight_status=None, preflight_source_sha256=None, preflight_analysis_version=None, preflight_pages_total=None, preflight_pages_analyzed=None, preflight_analysis_failed_pages=None, preflight_native_good_pages=None, preflight_native_borderline_pages=None, preflight_native_bad_pages=None, preflight_ocr_candidate_page_count=None, preflight_ocr_required_page_ratio=None, preflight_recommended_strategy=None, preflight_estimated_complexity=None, preflight_page_details=None, preflight_analyzed_at=None)
    service = KnowledgePreflightService(parser=Parser())
    service._page_count = lambda _path: 2

    report = service.analyze(Db(), document)
    assert report.native_borderline_page_numbers == [2]
    assert report.ocr_candidate_pages == []


def test_glyph_name_arabic_text_requires_ocr_fallback():
    quality = ExtractionQualityValidator().assess(
        "alefisolatedfathaisolated behinitial meemmedial thalfinalfathalow"
    )
    assert quality.glyph_noise_count >= 2
    assert not quality.has_arabic_unicode
    assert quality.needs_ocr_fallback


@pytest.mark.parametrize(
    ("text", "languages"),
    [
        ("Lis le texte puis réponds aux questions.", {"fr"}),
        ("اقرأ النص ثم أجب عن الأسئلة. Lis le texte puis réponds.", {"ar", "fr"}),
        ("كوّن جملاً باستعمال الصور. Make sentences using the pictures.", {"ar", "en"}),
    ],
)
def test_multilingual_clean_pages_pass_quality_without_content_loss(text, languages):
    quality = ExtractionQualityValidator().assess(text)
    assert quality.quality_passed
    assert languages.issubset(set(quality.languages_detected))


def test_random_ascii_symbol_garbage_fails_quality_gate():
    quality = ExtractionQualityValidator().assess("Kw A,«aw «.3 34w... 33% ,3qm30... @990 JSA...")
    assert not quality.quality_passed
    assert quality.failure_reasons


def test_short_latin_digit_symbol_garbage_is_not_classified_as_french():
    quality = ExtractionQualityValidator().assess("c 3 gnaw, 'ml '43% m3 ¢")
    assert "fr" not in quality.languages_detected
    assert quality.quality_score < 0.60


def test_ocr_quality_is_calibrated_below_one():
    quality = ExtractionQualityValidator().assess("اقرأ النص ثم أجب عن الأسئلة.")
    assert 0.80 <= quality.quality_score < 1.0


class _FakeItem:
    def __init__(self, text):
        self.text = text


class _FakeDoc:
    pages = {1: object()}

    def __init__(self, text):
        self._items = [_FakeItem(text)]

    def iterate_items(self):
        return iter(self._items)

    def export_to_text(self, **_kwargs):
        return self._items[0].text


def test_parser_activates_ocr_only_for_corrupted_text_layer(tmp_path):
    pdf = tmp_path / "arabic.pdf"
    pdf.write_bytes(b"%PDF")
    calls = []

    def convert(_path, use_ocr, _page_range):
        calls.append(use_ocr)
        return _FakeDoc("نص عربي صحيح مع English text" if use_ocr else "alefisolated meemmedial")

    parsed = DocumentParserService(converter_factory=convert).parse_pdf(pdf)
    assert calls == [False, True]
    assert parsed.extraction_mode == "page_adaptive"
    assert parsed.page_extractions[0].extraction_mode == "full_page_ocr"


def test_targeted_ocr_timeout_is_reported_without_exposing_source_path(tmp_path, monkeypatch):
    import time
    pdf = tmp_path / "slow.pdf"; pdf.write_bytes(b"%PDF")
    parser = DocumentParserService(converter_factory=lambda *_args: (time.sleep(0.1), object())[1])
    parser._settings.rag_ocr_page_timeout_seconds = 0.001
    with pytest.raises(OcrPageTimeoutError, match="Targeted OCR page conversion timed out"):
        parser._convert_targeted_ocr(pdf, 85)


def test_native_preflight_reuses_one_converter_for_batches_and_split_ranges(tmp_path, monkeypatch):
    class Converter:
        def __init__(self): self.ranges = []
        def convert(self, *, source, page_range):
            self.ranges.append((source, page_range))
            return SimpleNamespace(document=object())

    pdf = tmp_path / "batches.pdf"
    pdf.write_bytes(b"%PDF")
    converter = Converter()
    parser = DocumentParserService()
    builds = []
    monkeypatch.setattr(parser, "_build_converter", lambda *, use_ocr: builds.append(use_ocr) or converter)

    with parser.native_preflight_session() as diagnostics:
        for page_range in ((1, 30), (31, 60), (61, 75), (76, 90)):
            parser._convert(pdf, use_ocr=False, page_range=page_range)

    assert builds == [False]
    assert [page_range for _, page_range in converter.ranges] == [(1, 30), (31, 60), (61, 75), (76, 90)]
    assert diagnostics == {
        "document_converter_instances_created": 1,
        "conversion_calls": 4,
        "page_ranges_processed": [(1, 30), (31, 60), (61, 75), (76, 90)],
    }


def test_cached_converter_is_reused_without_a_second_model_initialization(tmp_path, monkeypatch):
    pdf = tmp_path / "cached-model.pdf"
    pdf.write_bytes(b"%PDF")
    parser = DocumentParserService()
    builds = []

    class Converter:
        def convert(self, **_kwargs):
            return SimpleNamespace(document=object())

    monkeypatch.setattr(parser, "_build_converter", lambda *, use_ocr: builds.append(use_ocr) or Converter())
    parser._convert(pdf, use_ocr=False, page_range=(1, 1))
    parser._convert(pdf, use_ocr=False, page_range=(2, 2))
    assert builds == [False]


def test_transient_model_download_failure_retries_then_preserves_infrastructure_error(tmp_path, monkeypatch):
    pdf = tmp_path / "model-download.pdf"
    pdf.write_bytes(b"%PDF")
    parser = DocumentParserService()
    parser._settings.rag_model_download_max_retries = 1

    class RemoteProtocolError(Exception):
        pass

    calls = []
    def failing_build(*, use_ocr):
        calls.append(use_ocr)
        raise RemoteProtocolError("Server disconnected without sending a response while downloading from HuggingFace")

    monkeypatch.setattr(parser, "_build_converter", failing_build)
    with pytest.raises(ModelResourceUnavailableError):
        parser._convert(pdf, use_ocr=False, page_range=None)
    assert calls == [False, False]


def test_real_conversion_failure_remains_pdf_parsing_error(tmp_path, monkeypatch):
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF")
    parser = DocumentParserService()
    monkeypatch.setattr(parser, "_build_converter", lambda *, use_ocr: (_ for _ in ()).throw(ValueError("malformed PDF content")))
    with pytest.raises(DocumentParsingError, match="PDF text extraction could not parse"):
        parser._convert(pdf, use_ocr=False, page_range=None)


def test_parser_skips_ocr_for_normal_arabic_text_layer(tmp_path):
    pdf = tmp_path / "normal.pdf"
    pdf.write_bytes(b"%PDF")
    calls = []

    def convert(_path, use_ocr, _page_range):
        calls.append(use_ocr)
        return _FakeDoc("هذا نص عربي سليم.")

    parsed = DocumentParserService(converter_factory=convert).parse_pdf(pdf)
    assert calls == [False, False]
    assert parsed.extraction_mode == "page_adaptive"
    assert parsed.page_extractions[0].extraction_mode == "native"


def test_cache_hit_short_circuits_docling_and_ocr(tmp_path, monkeypatch):
    pdf = tmp_path / "cached.pdf"
    pdf.write_bytes(b"%PDF cached")
    parser = DocumentParserService()
    cached = ParsedDocument(object(), 1, 1, "page_adaptive", [], cache_hit=True)
    monkeypatch.setattr(parser, "_load_cache", lambda *_args: cached)
    monkeypatch.setattr(parser, "_convert", lambda *_args, **_kwargs: pytest.fail("Docling must not run on a cache hit"))

    assert parser.parse_pdf(pdf, document_id=12) is cached


def test_cache_key_normalizes_ocr_languages(tmp_path, monkeypatch):
    pdf = tmp_path / "languages.pdf"
    pdf.write_bytes(b"%PDF languages")
    parser = DocumentParserService(converter_factory=lambda *_args: None)
    monkeypatch.setattr(parser._settings, "rag_ocr_languages", " en, ar,ar ")

    assert parser._cache_key(pdf)["ocr_languages"] == "ar,en"


def test_cache_manifest_reports_the_exact_invalidating_component(tmp_path, monkeypatch):
    pdf = tmp_path / "manifest.pdf"
    pdf.write_bytes(b"%PDF manifest")
    parser = DocumentParserService(converter_factory=lambda *_args: None)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(parser, "_cache_directory", lambda _document_id: cache_dir)
    monkeypatch.setattr(parser, "_legacy_cache_directory", lambda _document_id: tmp_path / "legacy-cache")
    key = parser._cache_key(pdf)
    stale_key = {**key, "pipeline_version": "previous-pipeline"}
    (cache_dir / "manifest.json").write_text(json.dumps({"key": stale_key}), encoding="utf-8")

    assert parser._load_cache(12, key) is None
    assert parser._cache_miss_reason == "pipeline_version_mismatch"


def test_cache_write_persists_manifest_and_each_page_payload(tmp_path, monkeypatch):
    class CacheDocument:
        def save_as_json(self, destination):
            destination.write_text('{"cached": true}', encoding="utf-8")

    pdf = tmp_path / "write.pdf"
    pdf.write_bytes(b"%PDF write")
    parser = DocumentParserService(converter_factory=lambda *_args: None)
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(parser, "_cache_directory", lambda _document_id: cache_dir)
    quality = ExtractionQualityValidator().assess("نص عربي سليم")
    page = PageExtraction(1, CacheDocument(), "full_page_ocr", quality)
    parsed = ParsedDocument(CacheDocument(), 1, 1, "page_adaptive", [page])

    parser._write_cache(12, parser._cache_key(pdf), parsed)

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pages"][0]["document"] == "page-1.json"
    assert (cache_dir / "page-1.json").is_file()


def test_failed_ocr_page_is_quarantined_instead_of_aborting_the_document(tmp_path):
    pdf = tmp_path / "still-corrupt.pdf"
    pdf.write_bytes(b"%PDF")

    def convert(_path, _use_ocr, _page_range):
        return _FakeDoc("alefisolated meemmedial thalfinal")

    parsed = DocumentParserService(converter_factory=convert).parse_pdf(pdf)

    assert parsed.page_extractions == []
    assert len(parsed.page_issues) == 1
    issue = parsed.page_issues[0]
    assert issue.page_number == 1
    assert issue.disposition == "quarantined"
    assert issue.extraction_mode_attempted == "full_page_ocr"
    assert "glyph_noise" in issue.failure_reasons


class _EmptyDoc:
    pages = {1: object()}

    def iterate_items(self, **_kwargs):
        return iter(())

    def export_to_text(self, **_kwargs):
        return ""


def test_structurally_blank_ocr_page_is_skipped_not_quarantined(tmp_path):
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"%PDF")

    def convert(_path, use_ocr, _page_range):
        return _EmptyDoc() if use_ocr else _FakeDoc("alefisolated meemmedial")

    parsed = DocumentParserService(converter_factory=convert).parse_pdf(pdf)

    assert parsed.page_extractions == []
    assert [(issue.page_number, issue.disposition) for issue in parsed.page_issues] == [(1, "skipped_low_information")]


class _FakeMultiPageDoc(_FakeDoc):
    pages = {1: object(), 2: object()}

    def __init__(self, page_text):
        self.page_text = page_text
        super().__init__("\n".join(page_text.values()))

    def export_to_text(self, *, page_no=None, **_kwargs):
        return self.page_text.get(page_no, "\n".join(self.page_text.values()))


def test_only_the_bad_page_gets_ocr_in_a_multi_page_pdf(tmp_path):
    pdf = tmp_path / "multi.pdf"
    pdf.write_bytes(b"%PDF")
    calls = []

    def convert(_path, use_ocr, page_range):
        calls.append((use_ocr, page_range))
        if page_range is None:
            return _FakeMultiPageDoc({1: "Lis le texte et réponds.", 2: "alefisolated meemmedial"})
        if use_ocr:
            return _FakeMultiPageDoc({2: "اقرأ النص ثم أجب عن الأسئلة."})
        return _FakeMultiPageDoc({1: "Lis le texte et réponds."})

    parsed = DocumentParserService(converter_factory=convert).parse_pdf(pdf, document_id=9)
    assert [(page.page_number, page.extraction_mode) for page in parsed.page_extractions] == [(1, "native"), (2, "full_page_ocr")]
    assert (True, (2, 2)) in calls
    assert not any(use_ocr and page_range != (2, 2) for use_ocr, page_range in calls)


def test_preflight_uses_native_conversion_only_and_matches_ingestion_classification(tmp_path):
    pdf = tmp_path / "preflight.pdf"
    pdf.write_bytes(b"%PDF")
    calls = []

    def convert(_path, use_ocr, page_range):
        calls.append((use_ocr, page_range))
        return _FakeMultiPageDoc({1: "هذا نص عربي سليم.", 2: "alefisolated meemmedial"})

    report = DocumentParserService(converter_factory=convert).preflight_pdf(pdf)

    assert calls == [(False, None)]
    assert report.native_good_page_numbers == [1]
    assert report.native_bad_page_numbers == [2]
    assert report.native_borderline_page_numbers == []


def test_preflight_strategy_thresholds_are_deterministic(monkeypatch):
    service = KnowledgePreflightService(parser=object())
    monkeypatch.setattr(service.settings, "rag_preflight_native_only_max_bad_ratio", 0.02)
    monkeypatch.setattr(service.settings, "rag_preflight_ocr_heavy_ratio", 0.60)

    assert service._strategy(0.02) == "native_only"
    assert service._strategy(0.03) == "native_with_targeted_ocr"
    assert service._strategy(0.60) == "ocr_heavy"
    assert service._complexity(230, 0.01) == "high"


def test_preflight_reuses_stored_result_and_invalidates_when_source_changes(tmp_path):
    class Db:
        commits = 0
        def commit(self): self.commits += 1

    class Parser:
        calls = 0
        def preflight_pdf(self, _path, *, page_range=None):
            self.calls += 1
            return NativePreflight(3, 3, [1], [2], [3], 0, 0, 12)

    pdf = tmp_path / "stored.pdf"
    pdf.write_bytes(b"first")
    document = SimpleNamespace(
        id=77, file_path=str(pdf), preflight_status=None, preflight_source_sha256=None,
        preflight_analysis_version=None, preflight_pages_total=None, preflight_native_good_pages=None,
        preflight_native_borderline_pages=None, preflight_native_bad_pages=None,
        preflight_ocr_candidate_page_count=None, preflight_ocr_required_page_ratio=None,
        preflight_recommended_strategy=None, preflight_estimated_complexity=None,
        preflight_page_details=None, preflight_analyzed_at=None,
    )
    parser = Parser()
    service = KnowledgePreflightService(parser=parser)
    service._page_count = lambda _path: 3
    db = Db()

    first = service.analyze(db, document)
    second = service.analyze(db, document)
    pdf.write_bytes(b"changed")
    third = service.analyze(db, document)

    assert not first.preflight_cache_hit
    assert second.preflight_cache_hit
    assert not third.preflight_cache_hit
    assert parser.calls == 2
    assert first.to_response()["ocr_candidate_pages"] == [2, 3]
    assert "text" not in str(first.to_response()).lower()


def test_large_preflight_uses_bounded_native_batches_and_keeps_final_partial_batch(tmp_path):
    class Parser:
        calls = []
        def preflight_pdf(self, _path, *, page_range=None):
            self.calls.append(page_range)
            start, end = page_range
            return NativePreflight(end - start + 1, end - start + 1, list(range(start, end + 1)), [], [], 0, 0, 1)

    service = KnowledgePreflightService(parser=Parser())
    service.settings.rag_preflight_batch_size = 30
    assert service._ranges(292) == [(1, 30), (31, 60), (61, 90), (91, 120), (121, 150), (151, 180), (181, 210), (211, 240), (241, 270), (271, 292)]
    successful, failed, _ = service._analyze_range(tmp_path / "large.pdf", (271, 292))
    assert not failed and successful[0].native_good_page_numbers == list(range(271, 293))
    assert service.parser.calls == [(271, 292)]


def test_preflight_technical_batch_failure_is_partial_not_corrupted_or_ocr_candidate(tmp_path):
    class Db:
        def commit(self): pass

    class Parser:
        def preflight_pdf(self, _path, *, page_range=None):
            start, end = page_range
            if 31 <= start <= 60:
                raise DocumentParsingError("native conversion failed")
            return NativePreflight(end - start + 1, end - start + 1, list(range(start, end + 1)), [], [], 0, 0, 1)

    pdf = tmp_path / "partial.pdf"; pdf.write_bytes(b"data")
    document = SimpleNamespace(id=88, file_path=str(pdf), preflight_status=None, preflight_source_sha256=None, preflight_analysis_version=None, preflight_pages_total=None, preflight_pages_analyzed=None, preflight_analysis_failed_pages=None, preflight_native_good_pages=None, preflight_native_borderline_pages=None, preflight_native_bad_pages=None, preflight_ocr_candidate_page_count=None, preflight_ocr_required_page_ratio=None, preflight_recommended_strategy=None, preflight_estimated_complexity=None, preflight_page_details=None, preflight_analyzed_at=None)
    service = KnowledgePreflightService(parser=Parser())
    service._page_count = lambda _path: 60
    service.settings.rag_preflight_batch_size = 30
    service.settings.rag_preflight_min_batch_size = 5

    report = service.analyze(Db(), document)
    assert report.preflight_status == "partial"
    assert report.pages_total == 60 and len(report.native_good_page_numbers) == 30
    assert report.analysis_failed_page_numbers == list(range(31, 61))
    assert report.ocr_candidate_pages == []
    assert report.recommended_strategy == "native_only"


def test_failed_preflight_with_known_page_count_is_undetermined(tmp_path):
    class Db:
        def commit(self): pass

    class Parser:
        def preflight_pdf(self, *_args, **_kwargs): raise DocumentParsingError("converter crashed")

    pdf = tmp_path / "failed.pdf"; pdf.write_bytes(b"data")
    document = SimpleNamespace(id=89, file_path=str(pdf), preflight_status=None, preflight_source_sha256=None, preflight_analysis_version=None, preflight_pages_total=None, preflight_pages_analyzed=None, preflight_analysis_failed_pages=None, preflight_native_good_pages=None, preflight_native_borderline_pages=None, preflight_native_bad_pages=None, preflight_ocr_candidate_page_count=None, preflight_ocr_required_page_ratio=None, preflight_recommended_strategy=None, preflight_estimated_complexity=None, preflight_page_details=None, preflight_analyzed_at=None)
    service = KnowledgePreflightService(parser=Parser())
    service._page_count = lambda _path: 300
    service.settings.rag_preflight_batch_size = 300
    service.settings.rag_preflight_min_batch_size = 300

    report = service.analyze(Db(), document)
    assert report.pages_total == 300 and report.preflight_status == "failed"
    assert report.recommended_strategy is None and report.estimated_complexity is None


def test_preflight_endpoint_requires_admin_authorization():
    dependencies = inspect.signature(preflight_knowledge_document).parameters.values()
    assert any(getattr(parameter.default, "dependency", None) is require_admin for parameter in dependencies)


def test_borderline_native_page_evaluates_ocr_and_selects_better_candidate(tmp_path, monkeypatch):
    pdf = tmp_path / "borderline.pdf"
    pdf.write_bytes(b"%PDF")
    calls = []

    def convert(_path, use_ocr, page_range):
        calls.append((use_ocr, page_range))
        if page_range is None:
            return _FakeDoc("c 3 gnaw, 'ml '43% m3 ¢")
        return _FakeDoc("اقرأ النص ثم أجب عن الأسئلة." if use_ocr else "c 3 gnaw, 'ml '43% m3 ¢")

    parsed = DocumentParserService(converter_factory=convert).parse_pdf(pdf)
    assert parsed.page_extractions[0].extraction_mode == "full_page_ocr"
    assert (True, (1, 1)) in calls


def test_native_quality_levels_treat_arabic_profile_garbage_as_non_high_confidence():
    parser = DocumentParserService(converter_factory=lambda *_args: None)
    quality = ExtractionQualityValidator().assess("x y z q r s t uv")
    assert parser._native_level(quality, arabic_profile=True) != NativeQualityLevel.HIGH_CONFIDENCE


class _OcrPictureDoc:
    texts = []
    pictures = [object()]
    tables = []

    @staticmethod
    def export_to_text(*, traverse_pictures=False):
        return "نص عربي من OCR" if traverse_pictures else ""


def test_full_page_ocr_picture_text_is_detected_and_enables_picture_traversal():
    chunker = DocumentChunker(max_tokens=12, tokenizer_name="test")
    diagnostics = chunker._document_diagnostics(_OcrPictureDoc())
    assert diagnostics["text_without_pictures_length"] == 0
    assert diagnostics["text_with_pictures_length"] > 0
    assert chunker._should_traverse_pictures("full_page_ocr")
    assert not chunker._should_traverse_pictures("pdf_text")


def test_full_page_ocr_picture_text_produces_arabic_chunk_and_uses_provider(monkeypatch):
    import docling.chunking
    import docling_core.transforms.chunker.tokenizer.huggingface as huggingface_tokenizer
    import transformers

    seen = {}

    class FakeAutoTokenizer:
        model_max_length = 512

    class FakeTokenizer:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def count_tokens(text):
            return len(text.split())

    class NativeChunk:
        text = "نص عربي سليم"

        class meta:
            headings = []
            doc_items = []

    class FakeHybridChunker:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        @staticmethod
        def chunk(**_kwargs):
            return [NativeChunk()]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _name: FakeAutoTokenizer())
    monkeypatch.setattr(huggingface_tokenizer, "HuggingFaceTokenizer", FakeTokenizer)
    monkeypatch.setattr(docling.chunking, "HybridChunker", FakeHybridChunker)

    chunks = DocumentChunker(max_tokens=12, tokenizer_name="test").chunk(
        document=_OcrPictureDoc(),
        document_id=18,
        title="مفتاح القراءة",
        original_filename="arabic.pdf",
        source=None,
        extraction_mode="full_page_ocr",
    )
    assert type(seen["serializer_provider"]).__name__ == "TraversePicturesProvider"
    assert chunks and "نص عربي" in chunks[0].text_original
    assert "alefisolated" not in chunks[0].text_original


def test_normal_pdf_keeps_default_serializer(monkeypatch):
    import docling.chunking
    import docling_core.transforms.chunker.tokenizer.huggingface as huggingface_tokenizer
    import transformers

    seen = {}

    class FakeAutoTokenizer:
        model_max_length = 512

    class FakeTokenizer:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def count_tokens(text):
            return len(text.split())

    class NativeChunk:
        text = "Regular text"

        class meta:
            headings = []
            doc_items = []

    class FakeHybridChunker:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        @staticmethod
        def chunk(**_kwargs):
            return [NativeChunk()]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _name: FakeAutoTokenizer())
    monkeypatch.setattr(huggingface_tokenizer, "HuggingFaceTokenizer", FakeTokenizer)
    monkeypatch.setattr(docling.chunking, "HybridChunker", FakeHybridChunker)
    DocumentChunker(max_tokens=12, tokenizer_name="test").chunk(
        document=_OcrPictureDoc(), document_id=1, title="Normal", original_filename="normal.pdf", source=None
    )
    assert "serializer_provider" not in seen


def test_chunker_initializes_tokenizer_once_for_multiple_cached_pages(monkeypatch):
    import docling.chunking
    import docling_core.transforms.chunker.tokenizer.huggingface as huggingface_tokenizer
    import transformers

    calls = []

    class FakeAutoTokenizer:
        model_max_length = 512

    class FakeTokenizer:
        def __init__(self, **_kwargs): pass
        @staticmethod
        def count_tokens(text): return len(text.split())

    class NativeChunk:
        text = "اقرأ le texte français"
        class meta:
            headings = []
            doc_items = []

    class FakeHybridChunker:
        def __init__(self, **_kwargs): pass
        @staticmethod
        def chunk(**_kwargs): return [NativeChunk()]

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return FakeAutoTokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(huggingface_tokenizer, "HuggingFaceTokenizer", FakeTokenizer)
    monkeypatch.setattr(docling.chunking, "HybridChunker", FakeHybridChunker)
    quality = ExtractionQualityValidator().assess("اقرأ le texte français")
    pages = [PageExtraction(number, _OcrPictureDoc(), "native", quality) for number in range(1, 5)]

    chunks = DocumentChunker(max_tokens=20, tokenizer_name="bert-base-multilingual-cased", local_files_only=True).chunk(
        document=object(), document_id=19, title="Guide", original_filename="guide.pdf", source=None,
        page_extractions=pages,
    )
    assert len(chunks) == 4
    assert calls == [("bert-base-multilingual-cased", {"local_files_only": True, "fix_mistral_regex": False})]


@pytest.mark.parametrize("text", ["Bonjour le monde", "اقرأ النص ثم أجب", "اقرأ le texte puis réponds"])
def test_offline_and_normal_chunkers_use_equivalent_tokenization(text, monkeypatch):
    import docling_core.transforms.chunker.tokenizer.huggingface as huggingface_tokenizer
    import transformers

    class FakeAutoTokenizer:
        model_max_length = 512

    class FakeTokenizer:
        def __init__(self, **_kwargs): pass
        @staticmethod
        def count_tokens(value): return len(value.split())

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: FakeAutoTokenizer())
    monkeypatch.setattr(huggingface_tokenizer, "HuggingFaceTokenizer", FakeTokenizer)
    normal = DocumentChunker(max_tokens=20, tokenizer_name="bert-base-multilingual-cased")
    offline = DocumentChunker(max_tokens=20, tokenizer_name="bert-base-multilingual-cased", local_files_only=True)
    assert normal._count_tokens(normal._get_tokenizer(), text) == offline._count_tokens(offline._get_tokenizer(), text)


@pytest.mark.parametrize("text", [".", "___________", "................", "1. ________", "□ □ □"])
def test_decorative_workbook_chunks_are_skipped(text):
    category, _, _ = DocumentChunker._classify_final_chunk(text, content_type="text", token_count=2)
    assert category == ChunkQualityCategory.LOW_INFORMATION


@pytest.mark.parametrize(
    ("text", "content_type"),
    [
        ("اقرأ", "text"),
        ("Lis et réponds.", "text"),
        ("A1", "table"),
        ("نص عربي سليم", "text"),
        ("اقرأ النص. Lis le texte.", "text"),
    ],
)
def test_short_or_multilingual_semantic_chunks_are_kept(text, content_type):
    category, _, _ = DocumentChunker._classify_final_chunk(text, content_type=content_type, token_count=4)
    assert category == ChunkQualityCategory.VALID_SEMANTIC


@pytest.mark.parametrize(
    "text",
    [
        "Le CEA assure la formation continue des enseignants mis à disposition dans la zone Maroc.",
        "https://www.education.gouv.fr/bo/22/Hebdo27/MENE2209721N.htm",
        "اقرأ النص. Lis le texte. A1 CM2 https://example.org/ressource",
        "1. Référence : MENE2209721N — 2024-2025, page 12.",
    ],
)
def test_structured_latin_and_reference_content_is_not_ascii_corruption(text):
    category, reason, _ = DocumentChunker._classify_final_chunk(text, content_type="text", token_count=12)
    assert category != ChunkQualityCategory.CORRUPTED
    assert reason != "severe_random_ascii_corruption"


def test_genuine_short_random_ascii_garbage_remains_corrupted():
    category, reason, _ = DocumentChunker._classify_final_chunk("a8x@@@1qz//::x9...", content_type="text", token_count=8)
    assert category == ChunkQualityCategory.CORRUPTED
    assert reason == "severe_random_ascii_corruption"


@pytest.mark.parametrize(
    "text",
    [
        "Kw A,«aw «.3 34w .3, 3w :33Kw... 33% ,3qm30... @990 JSA...",
        "alefisolated fathalow meemmedial arabicindicdigit",
    ],
)
def test_meaningful_garbage_is_rejected_not_skipped(text):
    category, _, _ = DocumentChunker._classify_final_chunk(text, content_type="text", token_count=15)
    assert category == ChunkQualityCategory.CORRUPTED


def test_document_with_decorative_chunks_still_returns_semantic_chunks(monkeypatch):
    import docling.chunking
    import docling_core.transforms.chunker.tokenizer.huggingface as huggingface_tokenizer
    import transformers

    class FakeAutoTokenizer:
        model_max_length = 512

    class FakeTokenizer:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def count_tokens(text):
            return len(text.split())

    class NativeChunk:
        class meta:
            headings = []
            doc_items = []

        def __init__(self, text):
            self.text = text

    class FakeHybridChunker:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def chunk(**_kwargs):
            return [NativeChunk("________"), NativeChunk("اقرأ النص ثم أجب عن الأسئلة."), NativeChunk("□ □ □")]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _name: FakeAutoTokenizer())
    monkeypatch.setattr(huggingface_tokenizer, "HuggingFaceTokenizer", FakeTokenizer)
    monkeypatch.setattr(docling.chunking, "HybridChunker", FakeHybridChunker)
    chunker = DocumentChunker(max_tokens=20, tokenizer_name="test")
    chunks = chunker.chunk(document=_OcrPictureDoc(), document_id=12, title="Workbook", original_filename="book.pdf", source=None)
    assert [chunk.chunk_index for chunk in chunks] == [0]
    assert chunks[0].text_original.startswith("اقرأ")
    assert chunker.last_metrics == {"chunks_kept": 1, "chunks_skipped_low_information": 2, "chunks_failed_quality": 0, "chunks_quarantined": 0}


def _preview_page(page_number=1, mode="native"):
    quality = ExtractionQualityValidator().assess("نص عربي سليم")
    return PageExtraction(page_number, object(), mode, quality)


def _quarantined_ocr_issue(page_number=1):
    native_quality = ExtractionQualityValidator().assess("alefisolated meemmedial")
    ocr_quality = ExtractionQualityValidator().assess("")
    return PageExtractionIssue(
        page_number=page_number,
        disposition="quarantined",
        extraction_mode_attempted="full_page_ocr",
        native_quality=native_quality,
        ocr_quality=ocr_quality,
        failure_reasons=ocr_quality.failure_reasons,
    )


def test_image_is_associated_only_with_a_confident_same_page_chunk():
    image = PreservedImage("knowledge-document:12:page:1:image:0", 12, 1, None, "private.png", "اقرأ النص", None)
    page = replace(_preview_page(), images=[image])
    chunk = _preview_chunk()
    chunks, pages = KnowledgeIngestionService._associate_images([chunk], [page])
    assert chunks[0].metadata["image_ids"] == [image.image_id]
    assert pages[0].images[0].associated_chunk_ids == [chunk.id]


def test_nearby_image_context_uses_geometry_without_inventing_caption():
    parser = DocumentParserService(converter_factory=lambda *_args: None)
    image_bbox = {"l": 100.0, "t": 100.0, "r": 200.0, "b": 180.0}
    blocks = [
        ({"l": 110.0, "t": 185.0, "r": 195.0, "b": 205.0}, "اقرأ الصورة ثم أجب."),
        ({"l": 400.0, "t": 500.0, "r": 500.0, "b": 530.0}, "نص بعيد."),
    ]
    assert parser._nearby_text(image_bbox, blocks, explicit_caption="") == "اقرأ الصورة ثم أجب."
    assert parser._nearby_text(image_bbox, blocks, explicit_caption="Caption from Docling") is None


def test_image_role_is_conservative_before_vlm():
    parser = DocumentParserService(converter_factory=lambda *_args: None)
    assert parser._classify_image_role({"l": 0, "t": 0, "r": 10, "b": 10}, [], "", None) == "decorative_or_layout"
    assert parser._classify_image_role({"l": 0, "t": 0, "r": 100, "b": 100}, [], "Figure 1", None) == "pedagogical_visual"
    assert parser._classify_image_role({"l": 0, "t": 0, "r": 100, "b": 100}, [], "", None) == "unknown"


def test_image_association_uses_confident_nearby_context():
    image = PreservedImage("knowledge-document:12:page:1:image:0", 12, 1, None, None, None, "اقرأ النص ثم أجب", image_role="pedagogical_visual")
    page = replace(_preview_page(), images=[image])
    chunk = replace(_preview_chunk(), text_original="اقرأ النص ثم أجب عن الأسئلة")
    chunks, pages = KnowledgeIngestionService._associate_images([chunk], [page])
    assert chunks[0].metadata["image_ids"] == [image.image_id]
    assert pages[0].images[0].associated_chunk_ids == [chunk.id]


def _preview_chunk(index=0):
    return DocumentChunk(
        id=f"chunk:{index}", document_id=12, chunk_index=index, text_original="نص عربي سليم",
        text_for_embedding="Document: Workbook\n\nنص عربي سليم", page_start=1, page_end=1,
        section=None, headings=[], content_type="text", metadata={}, token_count=5,
    )


def test_preview_exposes_clean_embedding_text_only_when_debug_is_requested():
    chunk = replace(_preview_chunk(), text_original="نص <!-- image -->", text_for_embedding="Document: Workbook\n\nنص عربي")
    preview = ParsePreview(12, 1, 1, 1, {}, [chunk], 1, 1, [], "complete", [], {}, False, "targeted_pages", 1.0, "key", None, True)

    assert "text_for_embedding" not in preview.to_response()["chunks"][0]
    assert preview.to_response(include_debug=True)["chunks"][0]["text_for_embedding"] == "Document: Workbook\n\nنص عربي"


def test_late_native_page_repair_replaces_chunks_and_completes(monkeypatch):
    import app.services.knowledge_ingestion_service as ingestion_module

    pages = [_preview_page()]
    parsed = ParsedDocument(object(), 1, 1, "page_adaptive", pages)

    class Parser:
        repair_calls = 0

        def parse_pdf(self, *_args, **_kwargs): return parsed

        def repair_page_with_ocr(self, *_args, **_kwargs):
            self.repair_calls += 1
            return _preview_page(1, "full_page_ocr")

    class Chunker:
        calls = 0

        def __init__(self, **_kwargs):
            self.last_metrics = {"chunks_kept": 1, "chunks_skipped_low_information": 0, "chunks_failed_quality": 0, "chunks_quarantined": 0}
            self.last_corruptions = []

        def chunk(self, **_kwargs):
            type(self).calls += 1
            if type(self).calls == 1:
                self.last_metrics["chunks_failed_quality"] = self.last_metrics["chunks_quarantined"] = 1
                self.last_corruptions = [{"page": 1, "quality_score": 0.7, "failure_reasons": ["severe_random_ascii_corruption"]}]
                return [_preview_chunk()]
            return [_preview_chunk()]

    monkeypatch.setattr(ingestion_module, "DocumentChunker", Chunker)
    parser = Parser()
    document = SimpleNamespace(id=12, file_path="fake.pdf", title="Workbook", original_filename="book.pdf", source=None, status=None)
    db = SimpleNamespace(commit=lambda: None)
    preview = KnowledgeIngestionService(parser=parser).parse_preview(db, document)
    assert parser.repair_calls == 1
    assert Chunker.calls == 2
    assert preview.quality_status == "complete"
    assert preview.statistics["pages_late_repaired"] == 1


def test_unrepairable_single_page_is_partial_not_document_failure(monkeypatch):
    import app.services.knowledge_ingestion_service as ingestion_module

    parsed = ParsedDocument(
        object(), 20, 20, "page_adaptive", [_preview_page(page) for page in range(2, 21)],
        page_issues=[_quarantined_ocr_issue(1)],
    )

    class Parser:
        def parse_pdf(self, *_args, **_kwargs): return parsed
        def repair_page_with_ocr(self, *_args, **_kwargs): return None

    class Chunker:
        def __init__(self, **_kwargs):
            self.last_metrics = {"chunks_kept": 1, "chunks_skipped_low_information": 0, "chunks_failed_quality": 0, "chunks_quarantined": 0}
            self.last_corruptions = []
        def chunk(self, **_kwargs): return [_preview_chunk()]

    monkeypatch.setattr(ingestion_module, "DocumentChunker", Chunker)
    document = SimpleNamespace(id=12, file_path="fake.pdf", title="Workbook", original_filename="book.pdf", source=None, status=None)
    db = SimpleNamespace(commit=lambda: None)
    preview = KnowledgeIngestionService(parser=Parser()).parse_preview(db, document)
    assert preview.quality_status == "partial"
    assert preview.statistics["pages_quarantined"] == 1
    assert preview.statistics["quarantined_page_numbers"] == [1]
    assert preview.statistics["ocr_failures"] == 1
    assert preview.chunks[0].text_original == "نص عربي سليم"


def test_failed_page_ratio_above_threshold_fails_document(monkeypatch):
    import app.services.knowledge_ingestion_service as ingestion_module
    parsed = ParsedDocument(object(), 1, 1, "page_adaptive", [_preview_page(2)], page_issues=[_quarantined_ocr_issue(1)])

    class Parser:
        def parse_pdf(self, *_args, **_kwargs): return parsed
        def repair_page_with_ocr(self, *_args, **_kwargs): return None

    class Chunker:
        def __init__(self, **_kwargs):
            self.last_metrics = {"chunks_kept": 1, "chunks_skipped_low_information": 0, "chunks_failed_quality": 0, "chunks_quarantined": 0}
            self.last_corruptions = []
        def chunk(self, **_kwargs): return [_preview_chunk()]

    monkeypatch.setattr(ingestion_module, "DocumentChunker", Chunker)
    document = SimpleNamespace(id=12, file_path="fake.pdf", title="Workbook", original_filename="book.pdf", source=None, status=None)
    with pytest.raises(KnowledgeIngestionError) as error:
        KnowledgeIngestionService(parser=Parser()).parse_preview(SimpleNamespace(commit=lambda: None), document)
    assert error.value.result_summary == {
        "quality_status": "failed",
        "pages_total": 1,
        "pages_quarantined_count": 1,
        "quarantined_page_numbers": [1],
        "failed_page_ratio": 1.0,
        "max_failed_page_ratio": KnowledgeIngestionService().settings.rag_max_failed_page_ratio,
        "failure_reason": "failed_page_ratio_exceeded",
        "chunks_valid": 1,
        "chunks_quarantined_count": 0,
        "ocr_failures_count": 1,
        "warnings_count": 1,
    }


def test_zero_valid_chunks_fails_document(monkeypatch):
    import app.services.knowledge_ingestion_service as ingestion_module
    parsed = ParsedDocument(object(), 1, 1, "page_adaptive", [_preview_page()])

    class Parser:
        def parse_pdf(self, *_args, **_kwargs): return parsed

    class Chunker:
        def __init__(self, **_kwargs):
            self.last_metrics = {"chunks_kept": 0, "chunks_skipped_low_information": 1, "chunks_failed_quality": 0, "chunks_quarantined": 0}
            self.last_corruptions = []
        def chunk(self, **_kwargs): return []

    monkeypatch.setattr(ingestion_module, "DocumentChunker", Chunker)
    document = SimpleNamespace(id=12, file_path="fake.pdf", title="Workbook", original_filename="book.pdf", source=None, status=None)
    with pytest.raises(KnowledgeIngestionError):
        KnowledgeIngestionService(parser=Parser()).parse_preview(SimpleNamespace(commit=lambda: None), document)
