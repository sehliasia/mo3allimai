"""Natural-language exercise search: deterministic constraints, hybrid
retrieval (faked here), multi-chunk reconstruction, 0 LLM, strict CEFR filter
and real-document fixtures (the descriptive paragraph must never be returned)."""

import re

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.database.base import Base
from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.schemas.exercise_generator import ExerciseSearchIn
from app.services.context_builder import ContextBuilder, ContextSourceBlock
from app.services.exercise_search_service import (
    ExerciseSearchService,
    build_retrieval_query,
    enrich_exercise_blocks,
    parse_query,
)
from app.services.retrieval_service import RetrievalResponse, RetrievalResult


def _result(rank, chunk):
    return RetrievalResult(
        rank=rank, score=1.0 / rank, vector_score=1.0 / rank, original_rank=rank,
        fused_rank=rank, rrf_score=1.0 / rank,
        chunk_id=chunk["id"], document_id=chunk["document_id"],
        document_title=chunk["document_title"],
        source_page_start=chunk["page"], source_page_end=chunk["page"],
        content_type=chunk.get("content_type", "worksheet_exercise"),
        language=chunk.get("language", "ar"), cefr_level=chunk.get("cefr_level"),
        structural_quality=None, has_image=False, requires_vision=False,
        heading_context=chunk["heading"], content=chunk["content"],
        appeared_in_dense=True, appeared_in_sparse=True, dense_rank=rank, sparse_rank=rank,
    )


class FakeRetrieval:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, db, query, *, top_k=12, **kwargs):
        tokens = [token.casefold() for token in re.findall(r"\w+", query, re.UNICODE)]
        def overlap(chunk):
            text = (" ".join(chunk["heading"]) + " " + chunk["content"]).casefold()
            return sum(1 for token in tokens if token and token in text)
        ranked = sorted(self.chunks, key=lambda c: (-overlap(c), c["id"]))
        results = [_result(rank, chunk) for rank, chunk in enumerate(ranked, 1)]
        return RetrievalResponse(
            query=query, model="fake-retrieval", top_k=top_k, results=results[:top_k],
            stale_references_skipped=0, candidate_top_k=top_k,
            retrieval_mode="hybrid", dense_candidate_count=top_k,
            sparse_candidate_count=top_k, union_candidate_count=len(results),
        )


def _school_chunks():
    return [
        {"id": 11, "document_id": 100, "document_title": "Cahier d'école Miftah",
         "page": 2, "heading": ["École — Vocabulaire"], "cefr_level": None,
         "content": "Exercice 1 : Complète les phrases.\n"
                    "1. Je vais à l'______ tous les matins.\n2. Le ______ est ouvert."},
        {"id": 12, "document_id": 100, "document_title": "Cahier d'école Miftah",
         "page": 2, "heading": ["École — Expression"], "cefr_level": None,
         "content": "Activité 2 : Relie chaque mot à son image de l'école.\n"
                    "1. la classe 2. le tableau 3. le cartable"},
        {"id": 13, "document_id": 100, "document_title": "Cahier d'école Miftah",
         "page": 4, "heading": ["Grammaire — Conjugaison"], "cefr_level": "A2",
         "content": "Exercice 3 : Conjugue le verbe au présent.\n1. أنا ___ (يكتب)\n2. هي ___ (يقرأ)"},
        {"id": 14, "document_id": 100, "document_title": "Cahier d'école Miftah",
         "page": 1, "heading": ["Introduction"], "cefr_level": None,
         "content": "Pour chaque niveau, des exercices de production sont proposés aux élèves "
                    "afin de consolider leurs acquis."},
    ]


def _famille_chunks():
    return [
        {"id": 21, "document_id": 200, "document_title": "Cahier Famille Miftah 1",
         "page": 3, "heading": ["Niveau A1 — La famille"], "cefr_level": "A1",
         "content": "Exercice 1 : Complète les phrases.\n1. هذا أبي\n2. هذه أمي\n3. هذا أخي"},
        {"id": 22, "document_id": 200, "document_title": "Cahier Famille Miftah 1",
         "page": 4, "heading": ["Niveau A1 — La famille"], "cefr_level": "A1",
         "content": "Activité : Choisis la bonne réponse.\na) الأب\nb) الأم\nc) الابن"},
    ]


def _service(chunks):
    retrieval = FakeRetrieval(chunks)
    context_builder = ContextBuilder(max_chunks=20, max_tokens=6000)
    return ExerciseSearchService(retrieval=retrieval, context_builder=context_builder,
                                 settings=Settings())


def _search(service, query, **overrides):
    payload = dict(query=query, limit=12, level=None, skills=[], exercise_type=None,
                   source_document_ids=[])
    payload.update(overrides)
    return service.search(None, ExerciseSearchIn(**payload))


# -- Query comprehension: constraints only when stated ---------------------

def test_parse_family_query_states_no_level_or_skill():
    parsed = parse_query("Je cherche des exercices sur la famille")
    assert parsed.level is None
    assert parsed.skills == []
    assert parsed.theme_tokens == ("famille",)


def test_parse_école_query_keeps_theme():
    parsed = parse_query("exercices sur l'école")
    assert parsed.level is None
    assert parsed.theme_tokens == ("école",)


def test_parse_vocabulaire_query_detects_skill_only():
    parsed = parse_query("recherche de vocabulaire")
    assert parsed.level is None
    assert parsed.skills == ["vocabulary"]


def test_parse_conjugaison_query_maps_to_grammar():
    parsed = parse_query("exercices sur la conjugaison")
    assert parsed.skills == ["grammar"]
    assert parsed.level is None


def test_parse_without_level_never_invents_one():
    assert parse_query("exercices sur l'école").level is None


def test_parse_a1_query_extracts_level():
    parsed = parse_query("exercices A1 de vocabulaire sur l'école")
    assert parsed.level == "A1"
    assert parsed.skills == ["vocabulary"]


def test_parse_b1_query_extracts_level():
    parsed = parse_query("exercices au niveau B1 de compréhension écrite")
    assert parsed.level == "B1"
    assert parsed.skills == ["reading"]


def test_parse_arabic_level_label():
    assert parse_query("تمارين في المستوى أ2 عن الأسرة").level == "A2"


def test_build_retrieval_query_keeps_theme_and_constraints():
    query = build_retrieval_query(parse_query("exercices A1 de vocabulaire sur l'école"))
    assert "niveau A1" in query
    assert "Vocabulaire" in query
    assert "école" in query


def test_search_retrieves_the_shared_admin_kb_without_owner_or_cefr_exclusion():
    # The exercise search must query the same shared knowledge base as the other
    # generators: no owner filter, no source_type filter, and no hard CEFR
    # exclusion pushed into retrieval (a requested level is enforced later at
    # the item level so admin documents with unset CEFR still surface).
    recorded = {}

    class RecordingRetrieval(FakeRetrieval):
        def search(self, db, query, *, top_k=12, **kwargs):
            recorded["filters"] = kwargs.get("filters")
            return super().search(db, query, top_k=top_k, **kwargs)

    retrieval = RecordingRetrieval(_school_chunks())
    context_builder = ContextBuilder(max_chunks=20, max_tokens=6000)
    service = ExerciseSearchService(retrieval=retrieval, context_builder=context_builder,
                                    settings=Settings())
    out = _search(service, "exercices A1 sur l'école")
    filters = recorded["filters"]
    assert filters.document_ids is None
    assert filters.cefr_level is None
    assert not hasattr(filters, "owner_id") or filters.owner_id is None
    assert not hasattr(filters, "source_type") or filters.source_type is None


# -- Search over a "real document" fixture --------------------------------

def test_search_l_école_returns_only_real_exercises_never_the_descriptive_paragraph():
    out = _search(_service(_school_chunks()), "exercices sur l'école")
    assert out.meta.llm_calls == 0
    assert out.meta.retrieval_mode == "hybrid"
    assert out.total >= 3
    assert all("Pour chaque niveau" not in item.prompt for item in out.items)
    types = {item.exercise_type for item in out.items}
    assert {"fill_blank", "matching"} <= types
    kinds = {item.title for item in out.items}
    assert any("Exercice" in title for title in kinds)


def test_search_a1_vocabulary_school_excludes_explicit_a2_chunk():
    out = _search(_service(_school_chunks()), "exercices A1 de vocabulaire sur l'école")
    for item in out.items:
        assert not (item.level_source == "explicit" and item.level != "A1")
    assert all(item.prompt != "Exercice 3 : Conjugue le verbe au présent.\n1. أنا ___ (يكتب)\n2. هي ___ (يقرأ)"
               for item in out.items)
    # Vocabulary skill is explicit from the document heading.
    assert any(item.skill == "Vocabulaire" and item.skill_source == "explicit" for item in out.items)


def test_search_family_uses_explicit_a1_from_heading():
    out = _search(_service(_famille_chunks()), "exercices sur la famille")
    assert out.meta.llm_calls == 0
    assert len(out.items) == 2
    for item in out.items:
        assert item.level == "A1"
        assert item.level_source == "explicit"
        assert item.document_title == "Cahier Famille Miftah 1"
        assert item.document_id == 200
        assert item.page_start in (3, 4)


def test_search_family_a1_strict_keeps_only_a1():
    out = _search(_service(_famille_chunks()), "exercices sur la famille", level="A1")
    assert out.total == 2
    assert all(item.level == "A1" for item in out.items)


def test_search_zero_llm_guaranteed_by_design():
    service = _service(_school_chunks())
    assert not hasattr(service, "llm")


# -- Multi-chunk reconstruction ------------------------------------------

def test_search_reassembles_exercise_split_across_chunks():
    chunks = [
        {"id": 31, "document_id": 300, "document_title": "Séquence 3",
         "page": 5, "heading": ["Exercices de révision"], "cefr_level": None,
         "content": "Exercice 3 : Complète les phrases."},
        {"id": 32, "document_id": 300, "document_title": "Séquence 3",
         "page": 5, "heading": ["Exercices de révision"], "cefr_level": None,
         "content": "1. Mon ______ s'appelle Karim.\n2. Ma sœur est ______."},
    ]
    out = _search(_service(chunks), "exercices de révision")
    assert out.total == 1
    item = out.items[0]
    assert "Exercice 3" in item.prompt and "Mon ______" in item.prompt  # merged block
    assert sorted(item.chunk_ids) == [31, 32]


# -- In-memory fragmented-exercise reconstruction (enrich_exercise_blocks) --

def _db_with_chunks(rows):
    import tempfile
    workdir = tempfile.mkdtemp(prefix="exercise-enrich-", dir="C:\\Users\\lenovo\\AppData\\Local\\Temp\\opencode")
    engine = create_engine(f"sqlite:///{workdir}/exercise-enrich.db")
    Base.metadata.create_all(engine)
    session = Session(engine)
    document = KnowledgeDocument(
        title=rows[0][5], original_filename="wb.pdf", stored_filename="wb.pdf",
        file_path=f"{workdir}/wb.pdf", mime_type="application/pdf", file_size=12,
        uploaded_by=1,
    )
    session.add(document)
    session.flush()
    chunks = []
    for doc_id, index, cid, page, heading, title, content in rows:
        chunks.append(KnowledgeChunk(
            id=cid, document_id=document.id, chunk_index=index, content=content,
            content_for_embedding=content, token_count=max(3, index + 1),
            content_type="worksheet_exercise", source_page_start=page,
            source_page_end=page, quality_status="complete",
            heading_context=[heading], chunk_metadata={},
            chunk_hash=f"h{index}", ingestion_version="test",
        ))
    session.add_all(chunks)
    session.commit()
    from app.services.retrieval_service import RetrievalResult
    blocks = []
    for doc_id, index, cid, page, heading, title, content in rows:
        blocks.append(RetrievalResult(
            rank=1, score=1.0, vector_score=1.0, original_rank=1, fused_rank=1,
            chunk_id=cid, document_id=document.id, document_title=title,
            source_page_start=page, source_page_end=page, content_type="worksheet_exercise",
            language="ar", cefr_level=None, structural_quality=None,
            has_image=False, requires_vision=False, heading_context=[heading],
            content=content,
        ))
    return session, blocks


def _block_from(result):
    return ContextSourceBlock(
        source_number=1, document_id=result.document_id, document_title=result.document_title,
        chunk_ids=[result.chunk_id], page_start=result.source_page_start, page_end=result.source_page_end,
        heading_context=list(result.heading_context), content_type=result.content_type,
        structural_quality=result.structural_quality, has_image=False, requires_vision=False,
        image_not_interpreted=False, vector_scores=[result.vector_score],
        reranker_scores=[result.reranker_score], original_ranks=[result.original_rank],
        reranked_ranks=[result.reranked_rank], content=result.content,
        estimated_token_count=max(1, len(result.content.split())),
    )


def test_enrich_reunites_fragmented_arabic_exercise_options_with_title():
    # A real pattern: the title chunk ("تمرين 4"), the directive chunk and the
    # "الاختيارات:" options chunk are separate rows several chunk_index apart in
    # the same document. ContextBuilder never returns them merged; the
    # enrichment must rebuild the full exercise in memory.
    rows = [
        (1, 30, 1000, 27, "تمرين 4", "Sokkan", "بعض الواجبات لاحظ"),
        (1, 50, 1006, 27, "تمرين 4", "Sokkan", "تمرين 4"),
        (1, 51, 1007, 27, "تمرين 4", "Sokkan", "اختر الإجابة الصحيحة"),
        (1, 52, 1008, 27, "", "Sokkan", "الاختيارات:\n- البيت\n- المدرسة\n- السوق"),
    ]
    db, results = _db_with_chunks(rows)
    title_result = next(r for r in results if r.content.strip() == "تمرين 4")
    blocks = enrich_exercise_blocks([_block_from(title_result)], db=db, window=8, max_tokens=5000)
    merged = blocks[0]
    assert len(blocks) == 1
    assert "تمرين 4" in merged.content
    assert "الاختيارات:" in merged.content           # options pulled in
    assert "- المدرسة" in merged.content
    assert sorted(merged.chunk_ids) == [1006, 1007, 1008]  # pre-fragment (idx 30) stays out of the window
    assert merged.page_start == merged.page_end == 27


def test_enrich_never_fuses_two_distinct_exercises():
    rows = [
        (1, 0, 2000, 1, "تمرين 1", "Cahier", "تمرين 1"),
        (1, 1, 2001, 1, "تمرين 1", "Cahier", "أكمل الجملة"),
        (1, 5, 2005, 2, "تمرين 2", "Cahier", "تمرين 2"),
        (1, 6, 2006, 2, "تمرين 2", "Cahier", "اختر الإجابة الصحيحة"),
    ]
    db, results = _db_with_chunks(rows)
    ex1 = next(r for r in results if r.content.strip() == "تمرين 1")
    blocks = enrich_exercise_blocks([_block_from(ex1)], db=db, window=6, max_tokens=5000)
    content = blocks[0].content
    # The walk must stop at the new explicit exercise title "تمرين 2".
    assert "تمرين 1" in content
    assert "أكمل الجملة" in content
    assert "تمرين 2" not in content
    assert sorted(blocks[0].chunk_ids) == [2000, 2001]


def test_enrich_respects_token_budget():
    rows = [
        (1, 0, 3000, 1, "تمرين 7", "Cahier", "تمرين 7"),
        (1, 1, 3001, 1, "تمرين 7", "Cahier", "كلمة كلمة كلمة كلمة كلمة كلمة كلمة كلمة كلمة كلمة"),
        (1, 2, 3002, 1, "تمرين 7", "Cahier", "ب" * 200),
    ]
    db, results = _db_with_chunks(rows)
    title = next(r for r in results if r.content.strip() == "تمرين 7")
    # max_tokens=2: only the 1-word title fits; the 10-word neighbour must not.
    blocks = enrich_exercise_blocks([_block_from(title)], db=db, window=2, max_tokens=2)
    assert len(blocks) == 1
    assert sorted(blocks[0].chunk_ids) == [3000]



# -- Facets derive from actual results ------------------------------------

def test_search_facets_come_from_found_results():
    out = _search(_service(_school_chunks()), "exercices sur l'école")
    assert "documents" in out.facets
    assert any(facet.value == "Cahier d'école Miftah" for facet in out.facets["documents"])
    assert out.facets["levels"]
    assert sum(facet.count for facet in out.facets["documents"]) == out.total


# -- Empty / filtered-to-zero search -------------------------------------

def test_search_zero_hits_when_nothing_matches():
    chunks = _school_chunks()
    # Freeze each chunk's content away from every possible token.
    chunks = [dict(chunk, content="محتوى غير مرتبط", heading=["Autre"]) for chunk in chunks]
    out = _search(_service(chunks), "exercices sur la famille")
    assert out.total == 0
    assert out.meta.llm_calls == 0