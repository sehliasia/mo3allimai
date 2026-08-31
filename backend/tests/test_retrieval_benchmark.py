from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.services.retrieval_benchmark import (
    assert_effective_mode,
    calculate_metrics,
    load_benchmark_cases,
    render_comparison_summary,
    render_final_h6_report,
    render_h5_final_context_summary,
    render_markdown_report,
    run_dense_benchmark,
    serialize_result,
)
from app.services.retrieval_service import RetrievalResponse, RetrievalResult


FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_benchmark.json"


def _result(**overrides) -> RetrievalResult:
    data = {
        "rank": 1, "score": 0.91, "vector_score": 0.91, "original_rank": 1,
        "chunk_id": 7, "document_id": 4, "document_title": "Guide pédagogique", "source_page_start": 2,
        "source_page_end": 2, "content_type": "worksheet_exercise", "language": "ar", "cefr_level": "A1",
        "structural_quality": "structured", "has_image": False, "requires_vision": False,
        "heading_context": ["La famille", "Activité"], "content": "Activité : présente ta famille.",
    }
    data.update(overrides)
    return RetrievalResult(**data)


def test_benchmark_fixture_is_valid_and_has_representative_multilingual_cases():
    cases = load_benchmark_cases(FIXTURE)
    assert len(cases) == 20
    assert {case.cefr_level for case in cases} == {"A1", "A2", "B1"}
    assert sum(case.language == "ar" for case in cases) >= 5
    assert all(case.expected_topics and case.desired_content_types for case in cases)
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == "9f78feb87f1a2e396260497c7101b5fa6cb3d2b809e9bfb45600c268b7e1db97"


def test_invalid_benchmark_fixture_is_rejected(tmp_path):
    fixture = tmp_path / "invalid.json"
    fixture.write_text('{"version": 1, "cases": [{"id": "x"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        load_benchmark_cases(fixture)


def test_metrics_are_deterministic_and_mark_skill_metadata_unavailable():
    case = load_benchmark_cases(FIXTURE)[0]
    results = [_result(), _result(rank=2, chunk_id=8, content_type="paragraph", heading_context=["Autre"], content="Texte neutre")]
    first = calculate_metrics(case, results)
    second = calculate_metrics(case, results)
    assert first == second
    assert first["TOPIC_RELEVANCE_AT_K"]["value"] == 0.5
    assert first["LEVEL_COMPATIBILITY_AT_K"]["value"] == 1.0
    assert first["SKILL_RELEVANCE_AT_K"] == {
        "available": False, "value": None,
        "definition": "unavailable: RetrievalResult has no canonical per-chunk skills metadata",
    }
    assert first["CONCRETE_RESOURCE_AT_K"]["value"] is True


def test_result_serializer_is_bounded_and_handles_missing_metadata():
    result = _result(language=None, cefr_level=None, structural_quality=None, heading_context=[], content="mot\n" * 300)
    item = serialize_result(result, preview_chars=40)
    assert item["language"] is None and item["cefr_level"] is None
    assert "\n" not in item["preview"] and item["preview"].endswith("…")
    assert len(item["preview"]) <= 40


def test_result_serializer_preserves_neighbor_provenance_for_h5_diagnostics():
    item = serialize_result(_result(neighbor_of=44))
    assert item["neighbor_of"] == 44


def test_h5_final_summary_reports_final_context_not_composer_only_metrics():
    report = render_h5_final_context_summary({
        "case": ([_result()], {
            "final_context_chunk_count": 5,
            "final_context_estimated_tokens": 900,
            "final_context_token_budget": 1800,
            "final_context_document_count": 2,
            "final_context_max_chunks_per_document": 3,
            "final_context_category_counts": {},
            "strong_concrete_available_primary_top10": True,
            "strong_concrete_available_extended_top20": True,
            "strong_concrete_selected_by_composer": True,
            "final_context_concrete_retained": True,
            "selected_from_extended_pool_count": 1,
            "final_retained_extended_count": 1,
            "same_document_fallback_fill_count": 0,
            "final_retained_same_document_fallback_count": 0,
            "neighbor_count": 1,
            "neighbors_rejected_as_duplicates": 2,
            "composer_duplicate_rate": 0.2,
            "final_context_duplicate_rate": 0.0,
            "final_context_bounded": True,
        }),
    })
    assert "Mean distinct documents / mean-max chunks per document: 2.00/3.00-3" in report
    assert "Neighbors added/rejected as duplicates: 1/2" in report
    assert "Final contexts bounded by ContextBuilder: 1/1" in report


def test_h6_report_keeps_retrieval_top10_and_final_context_as_distinct_units():
    case = load_benchmark_cases(FIXTURE)[0]
    candidate = _result(
        document_title="CEFR Companion volume fra", pedagogical_role="EXERCISE",
        is_concrete_classroom_material=True, final_rank=17,
        composition_category="CONCRETE_CLASSROOM_MATERIAL",
        composition_selection_reason="selected_extended_concrete_material",
    )
    retrieval = {case.id: ([candidate], {"effective_retrieval_mode": "hybrid", "reranking_applied": False, "composition_candidates": [candidate]})}
    final = {case.id: ([candidate], {
        "composition_candidates": [candidate], "final_context_chunk_count": 1,
        "final_context_estimated_tokens": 12, "final_context_token_budget": 1800,
        "final_context_document_count": 1, "final_context_max_chunks_per_document": 1,
        "final_context_category_counts": {"CONCRETE_CLASSROOM_MATERIAL": 1},
        "final_context_concrete_retained": True,
        "strong_concrete_available_primary_top10": False,
        "strong_concrete_available_extended_top20": True,
        "strong_concrete_selected_by_composer": True,
        "selected_from_extended_pool_count": 1, "final_retained_extended_count": 1,
        "same_document_fallback_fill_count": 0, "final_retained_same_document_fallback_count": 0,
        "neighbor_count": 0, "neighbors_rejected_as_duplicates": 0,
        "composer_duplicate_rate": 0.0, "final_context_duplicate_rate": 0.0,
    })}
    report = render_final_h6_report(
        cases=[case], dense_runs=retrieval, hybrid_runs=retrieval,
        pedagogical_runs=retrieval, final_context_runs=final,
    )
    assert "A/B/C evaluation unit: retrieval Top-10. D evaluation unit: actual bounded RAGContext." in report
    assert "Extended-pool recovery" in report and "selected_extended_concrete_material" not in report
    assert f"| {case.id} | 17 |" in report
    assert "Reranker OFF" in report and "no CEFR penalty" in report
    assert "Decision for this run: READY_FOR_H7_REVIEW" in report


def test_empty_results_render_without_llm_reranker_or_mutation():
    case = load_benchmark_cases(FIXTURE)[0]
    calls = []
    def run_case(received):
        calls.append(received.id)
        return [], {"stale_references_skipped": 0}
    report = render_markdown_report(cases=[case], run_case=run_case, model_id="fake", top_k=10)
    assert calls == [case.id]
    assert "No canonical results" in report
    assert "SKILL_RELEVANCE_AT_K**: unavailable" in report
    assert "Retrieval mode: dense" in report
    assert "Reranker: false" in report


def test_dense_runner_forces_existing_search_without_filters_or_reranker():
    cases = load_benchmark_cases(FIXTURE)[:2]
    calls = []

    def search(db, query, **kwargs):
        calls.append((db, query, kwargs))
        return RetrievalResponse(
            query=query, model="fake", top_k=kwargs["top_k"], results=[_result()],
            stale_references_skipped=0, candidate_top_k=kwargs["top_k"],
        )

    runs = run_dense_benchmark(cases=cases, search=search, db=object(), top_k=10)
    assert list(runs) == [case.id for case in cases]
    assert [call[2] for call in calls] == [{"top_k": 10, "rerank": False}] * 2


def test_hybrid_report_shows_effective_rrf_configuration_and_provenance():
    case = load_benchmark_cases(FIXTURE)[0]
    both = _result(rank=1, fused_rank=1, appeared_in_dense=True, dense_rank=2, appeared_in_sparse=True, sparse_rank=1, rrf_score=.0325)
    dense_only = _result(rank=2, chunk_id=8, fused_rank=2, appeared_in_dense=True, dense_rank=1, appeared_in_sparse=False, sparse_rank=None, rrf_score=.0163)
    sparse_only = _result(rank=3, chunk_id=9, fused_rank=3, appeared_in_dense=False, dense_rank=None, appeared_in_sparse=True, sparse_rank=2, rrf_score=.0161)
    report = render_markdown_report(
        cases=[case], run_case=lambda _: ([both, dense_only, sparse_only], {
            "effective_retrieval_mode": "hybrid", "reranking_applied": False,
            "dense_candidate_count": 20, "sparse_candidate_count": 20, "union_candidate_count": 30,
        }), model_id="fake", top_k=10, retrieval_mode="hybrid",
        dense_candidate_top_k=20, sparse_candidate_top_k=20, rrf_k=60,
    )
    assert "# Hybrid retrieval" in report
    assert "Retrieval mode: hybrid" in report
    assert "Dense candidate Top-K: 20" in report and "Sparse candidate Top-K: 20" in report and "RRF k: 60" in report
    assert "Final provenance — both arms: 1; dense only: 1; sparse only: 1" in report
    assert "| Final rank |" in report and "| 1 |" in report


def test_h4_report_displays_adjustments_without_changing_h3_provenance():
    case = load_benchmark_cases(FIXTURE)[0]
    item = _result(
        fused_rank=2, final_rank=1, rrf_score=.032, pedagogical_adjustment_total=.09,
        role_adjustment=.06, concreteness_adjustment=.03, final_score=1.04,
        adjustment_reasons=["exercise_role_match", "concrete_classroom_material"],
    )
    report = render_markdown_report(
        cases=[case], run_case=lambda _: ([item], {"dense_candidate_count": 20, "sparse_candidate_count": 20, "union_candidate_count": 30}),
        model_id="fake", top_k=10, retrieval_mode="hybrid", dense_candidate_top_k=20,
        sparse_candidate_top_k=20, rrf_k=60, pedagogical_ranking=True,
    )
    assert "# Hybrid + Pedagogical ranking H4" in report
    assert "Pedagogical ranking: enabled" in report
    assert "+0.090" in report and "exercise_role_match" in report


def test_effective_mode_guard_rejects_dense_output_labelled_hybrid():
    with pytest.raises(ValueError, match="requested hybrid"):
        assert_effective_mode({"case": ([_result()], {"effective_retrieval_mode": "dense", "reranking_applied": False})}, requested_mode="hybrid")


def test_comparison_summary_is_deterministic_and_cefr_diagnostic_is_non_ranking():
    cases = load_benchmark_cases(FIXTURE)[:2]
    dense = {
        cases[0].id: ([_result(document_title="CEFR Companion volume fra")], {}),
        cases[1].id: ([_result(chunk_id=8, document_title="Guide pédagogique")], {}),
    }
    hybrid = {
        cases[0].id: ([_result(document_title="CEFR Companion volume fra", content="La famille activité")], {}),
        cases[1].id: ([_result(chunk_id=8, document_title="Guide pédagogique")], {}),
    }
    first = render_comparison_summary(cases=cases, dense_runs=dense, hybrid_runs=hybrid, top_k=10)
    second = render_comparison_summary(cases=cases, dense_runs=dense, hybrid_runs=hybrid, top_k=10)
    assert first == second
    assert "# Dense vs Hybrid H3 Summary" in first
    dense_report = render_markdown_report(cases=cases, run_case=lambda case: dense[case.id], model_id="fake", top_k=10)
    assert "CEFR Companion results across all Top-10s: 1" in dense_report
    assert [result.chunk_id for result in dense[cases[0].id][0]] == [7]
