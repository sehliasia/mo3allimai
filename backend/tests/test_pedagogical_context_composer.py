from __future__ import annotations

from app.services.pedagogical_context_composer import CompositionCategory, PedagogicalContextComposer
from app.services.context_builder import ContextBuilder
from app.services.retrieval_service import RetrievalResult


def _result(rank: int, *, chunk_id: int | None = None, document_id: int = 1, role: str = "REFERENCE", concrete: bool = False, page: int = 1, heading: list[str] | None = None) -> RetrievalResult:
    return RetrievalResult(
        rank=rank, score=.02, vector_score=.8, original_rank=rank, chunk_id=chunk_id or rank,
        document_id=document_id, document_title=f"Document {document_id}", source_page_start=page, source_page_end=page,
        content_type="worksheet_exercise" if concrete else "paragraph", language="fr", cefr_level=None,
        structural_quality="structured", has_image=False, requires_vision=False, heading_context=heading or [str(chunk_id or rank)],
        content="Consigne : répondez." if concrete else "Référence pédagogique.", fused_rank=rank,
        pedagogical_role=role, is_concrete_classroom_material=concrete,
    )


def test_categories_are_deterministic_and_generic_task_is_not_concrete():
    composer = PedagogicalContextComposer()
    assert composer.category(_result(1, concrete=True)) == CompositionCategory.CONCRETE
    assert composer.category(_result(1, role="METHODOLOGY")) == CompositionCategory.GUIDANCE
    assert composer.category(_result(1, role="TASK")) == CompositionCategory.SUPPORTING


def test_activity_composition_prefers_concrete_then_guidance_then_support():
    composer = PedagogicalContextComposer()
    candidates = [_result(1, role="REFERENCE"), _result(2, role="METHODOLOGY", document_id=2), _result(3, concrete=True, document_id=3)]
    result = composer.compose(candidates, intent="concrete_activity")
    assert [item.chunk_id for item in result.selected[:3]] == [3, 2, 1]
    assert result.concrete_material_available
    assert result.selected[0].composition_selection_reason == "selected_concrete_material"


def test_no_concrete_gracefully_falls_back_and_methodology_prefers_guidance():
    composer = PedagogicalContextComposer()
    candidates = [_result(1, role="REFERENCE"), _result(2, role="METHODOLOGY", document_id=2)]
    activity = composer.compose(candidates, intent="concrete_activity")
    methodology = composer.compose(candidates, intent="methodology")
    assert not activity.concrete_material_available and len(activity.selected) == 2
    assert methodology.selected[0].pedagogical_role == "METHODOLOGY"


def test_document_diversity_duplicates_and_bounds_are_preserved():
    composer = PedagogicalContextComposer()
    candidates = [
        _result(1, chunk_id=1, document_id=1, page=1, heading=["A"]),
        _result(2, chunk_id=2, document_id=1, page=1, heading=["A"]),
        _result(3, chunk_id=3, document_id=1, page=2, heading=["B"]),
        _result(4, chunk_id=4, document_id=1, page=3, heading=["C"]),
        _result(5, chunk_id=5, document_id=2, page=1, heading=["D"]),
    ]
    result = composer.compose(candidates, intent="general")
    ids = [item.chunk_id for item in result.selected]
    assert 2 not in ids and ids.count(1) <= 1
    assert sum(item.document_id == 1 for item in result.selected) == 3
    assert 5 in ids and len(ids) <= 6
    assert [item.composition_rank for item in result.selected] == list(range(1, len(ids) + 1))
    assert any(item.composition_selection_reason == "fallback_fill_same_document" for item in result.selected)


def test_extended_pool_is_bounded_and_only_used_for_missing_concrete_or_underfill():
    composer = PedagogicalContextComposer()
    primary = [_result(index, chunk_id=index, document_id=index, role="REFERENCE") for index in range(1, 11)]
    extended_concrete = _result(12, chunk_id=12, document_id=12, concrete=True)
    beyond_bound = _result(21, chunk_id=21, document_id=21, concrete=True)
    result = composer.compose([*primary, extended_concrete, beyond_bound], intent="concrete_activity")
    ids = [item.chunk_id for item in result.selected]
    assert 12 in ids and 21 not in ids
    assert result.diagnostics["strong_concrete_available_primary_top10"] is False
    assert result.diagnostics["strong_concrete_available_extended_top20"] is True
    assert any(item.composition_selection_reason == "selected_extended_concrete_material" for item in result.selected)


def test_primary_sufficient_does_not_select_extended_candidates():
    composer = PedagogicalContextComposer()
    primary = [_result(index, chunk_id=index, document_id=index, concrete=index == 1) for index in range(1, 11)]
    extended = _result(11, chunk_id=11, document_id=11, concrete=True)
    result = composer.compose([*primary, extended], intent="concrete_activity")
    assert 11 not in [item.chunk_id for item in result.selected]


def test_final_contextbuilder_remains_token_bounded_after_composer_selection():
    composer = PedagogicalContextComposer()
    candidates = [
        _result(index, chunk_id=index, document_id=index, concrete=index == 1)
        for index in range(1, 7)
    ]
    composed = composer.compose(candidates, intent="concrete_activity")
    context = ContextBuilder(max_chunks=6, max_tokens=8, neighbor_expansion=False).build("activité", composed.selected)
    assert context.estimated_token_count <= 8
    assert len(context.included_results) == len(context.included_chunk_ids)
    assert any(item.is_concrete_classroom_material for item in composed.selected)
