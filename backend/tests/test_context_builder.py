from app.services.context_builder import ContextBuilder
from app.services.retrieval_service import RetrievalResult
from types import SimpleNamespace


def _result(chunk_id, content, *, rank=None, page=4, heading=None, quality="structured", vision=False, document_id=15):
    rank = rank or chunk_id
    return RetrievalResult(
        rank=rank, score=0.9 - rank / 100, vector_score=0.9 - rank / 100, original_rank=rank,
        chunk_id=chunk_id, document_id=document_id, document_title="wc-lesson-plans", source_page_start=page,
        source_page_end=page, content_type="paragraph", language="en", cefr_level="A1",
        structural_quality=quality, has_image=vision, requires_vision=vision, heading_context=heading or ["Lesson Goals"],
        content=content, reranker_score=0.5 if rank == 1 else None, reranked_rank=rank if rank == 1 else None,
    )


def test_context_groups_complementary_lesson_goal_chunks_without_losing_bullets():
    context = ContextBuilder(max_chunks=6, max_tokens=300).build("Quels sont les objectifs ?", [
        _result(1, "LESSON 1: QUALIFYING NATIONS OF THE WORLD CUP", rank=1),
        _result(2, "- Say which sports they like\n- Identify the countries who qualified", rank=2),
        _result(3, "- Categorize qualified countries by FIFA continental zones\n- Identify and describe flags", rank=3),
    ])
    assert len(context.source_blocks) == 1
    assert context.source_blocks[0].chunk_ids == [1, 2, 3]
    assert "Identify the countries" in context.context_text
    assert "Categorize qualified countries" in context.context_text
    assert "Pages: 4" in context.context_text


def test_context_deduplicates_title_excludes_unreliable_and_keeps_vision_metadata():
    context = ContextBuilder(max_chunks=6, max_tokens=300).build("سؤال", [
        _result(1, "Lesson Goals", rank=1, vision=True),
        _result(2, "Lesson Goals\n- Identify countries", rank=2),
        _result(3, "Lesson Goals", rank=3),
        _result(4, "unreliable layout", rank=4, quality="layout_unreliable"),
    ])
    assert context.included_chunk_ids == [1, 2]
    assert set(context.excluded_chunk_ids) == {3, 4}
    assert context.has_requires_vision is True
    assert context.source_blocks[0].image_not_interpreted is True
    assert "not interpreted" in context.context_text


def test_context_preserves_retrieval_order_and_drops_lower_priority_chunks_at_budget():
    context = ContextBuilder(max_chunks=2, max_tokens=8).build("French query العربية", [
        _result(1, "un deux trois", rank=2, heading=["A"]),
        _result(2, "quatre cinq six", rank=1, heading=["B"]),
        _result(3, "sept huit neuf", rank=3, heading=["C"]),
    ])
    assert context.included_chunk_ids == [1, 2]
    assert 3 in context.excluded_chunk_ids
    assert context.estimated_token_count <= 8
    assert "French query العربية" == context.query


def test_context_expands_only_an_immediate_compatible_neighbor_without_vector_search():
    first = SimpleNamespace(id=1, document_id=15, chunk_index=4, source_page_start=4, ingestion_version="test")
    neighbor = SimpleNamespace(
        id=2, document_id=15, chunk_index=5, content="- Tell the teams and countries they cheer for",
        source_page_start=5, source_page_end=5, content_type="paragraph", heading_context=["Lesson Goals"],
        chunk_metadata={"structural_quality": "structured"},
    )
    unrelated = SimpleNamespace(
        id=3, document_id=15, chunk_index=3, content="Vocabulary list", source_page_start=4, source_page_end=4,
        content_type="paragraph", heading_context=["Vocabulary"], chunk_metadata={},
    )

    class Scalars:
        def __init__(self, values): self.values = values
        def all(self): return self.values

    class FakeDb:
        def __init__(self): self.calls = 0
        def scalars(self, _statement):
            self.calls += 1
            return Scalars([first] if self.calls == 1 else [unrelated, neighbor])

    context = ContextBuilder(max_chunks=3, max_tokens=200).build(
        "objectifs", [_result(1, "- Say which sports they like", rank=1)], db=FakeDb()
    )
    assert context.included_chunk_ids == [1, 2]
    assert "Tell the teams" in context.context_text
    assert "Vocabulary list" not in context.context_text


def test_short_top_anchor_prioritizes_two_useful_same_page_siblings_over_unrelated_results():
    anchor = SimpleNamespace(id=100, document_id=88, chunk_index=4, source_page_start=4, ingestion_version="v1")
    sibling_one = SimpleNamespace(
        id=108, document_id=88, chunk_index=5, content="Students will be able to:\n- Identify the relevant countries",
        source_page_start=4, source_page_end=4, content_type="paragraph", heading_context=["Goals"],
        chunk_metadata={"structural_quality": "structured"}, ingestion_version="v1",
    )
    sibling_two = SimpleNamespace(
        id=109, document_id=88, chunk_index=6, content="- Categorize countries\n- Describe their flags",
        source_page_start=4, source_page_end=4, content_type="paragraph", heading_context=["Goals"],
        chunk_metadata={"structural_quality": "structured"}, ingestion_version="v1",
    )
    garbage = SimpleNamespace(
        id=110, document_id=88, chunk_index=7, content="_____", source_page_start=4, source_page_end=4,
        content_type="paragraph", heading_context=["Goals"], chunk_metadata={}, ingestion_version="v1",
    )

    class Scalars:
        def __init__(self, values): self.values = values
        def all(self): return self.values

    class FakeDb:
        def __init__(self): self.calls = 0
        def scalars(self, _statement):
            self.calls += 1
            return Scalars([anchor] if self.calls == 1 else [anchor, sibling_one, sibling_two, garbage])

    retrieved = [
        _result(100, "LESSON 1: A short title", rank=1, page=4, heading=["Goals"], document_id=88),
        _result(200, "Procedure on a different page", rank=2, page=5, document_id=88),
        _result(201, "Another unrelated activity", rank=3, page=10, document_id=88),
        _result(202, "Further unrelated content", rank=4, page=3, document_id=88),
        _result(203, "More unrelated content", rank=5, page=2, document_id=88),
        _result(204, "Lower priority unrelated content", rank=6, page=8, document_id=88),
        _result(108, sibling_one.content, rank=9, page=4, heading=["Goals"], document_id=88),
        _result(109, sibling_two.content, rank=10, page=4, heading=["Goals"], document_id=88),
    ]
    context = ContextBuilder(max_chunks=6, max_tokens=300, neighbor_expansion=False).build("goals", retrieved, db=FakeDb())
    assert 100 in context.included_chunk_ids and 108 in context.included_chunk_ids and 109 in context.included_chunk_ids
    assert 204 not in context.included_chunk_ids
    assert 110 not in context.included_chunk_ids
    assert context.source_blocks[0].original_ranks[0] == 1
    assert context.estimated_token_count <= 300 and len(context.included_chunk_ids) == 6


def test_neighbor_colliding_with_anchor_diagnostic_unit_is_rejected():
    anchor = _result(1, "Consigne : écoutez.", rank=1, page=4, heading=["Exercise"], document_id=7)
    source = SimpleNamespace(id=1, document_id=7, chunk_index=1)
    duplicate = SimpleNamespace(
        id=2, document_id=7, chunk_index=2, source_page_start=4, source_page_end=4,
        content_type="list", heading_context=["Exercise"], content="Suite distincte.", chunk_metadata={},
    )

    class Scalars:
        def __init__(self, values): self.values = values
        def all(self): return self.values
    class Db:
        def __init__(self): self.calls = 0
        def scalars(self, _query):
            self.calls += 1
            return Scalars([source] if self.calls == 1 else [duplicate])

    context = ContextBuilder(max_chunks=3, max_tokens=200).build("écoute", [anchor], db=Db())
    assert context.included_chunk_ids == [1]
    assert context.neighbors_rejected_as_duplicates == 1
