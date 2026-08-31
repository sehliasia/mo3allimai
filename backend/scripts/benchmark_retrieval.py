"""Read-only H3 dense/hybrid benchmark; historical H1 artifacts are never overwritten."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.session import SessionLocal
from app.services.embedding_providers import get_embedding_provider
from app.services.qdrant_service import QdrantService
from app.services.retrieval_benchmark import (
    assert_effective_mode,
    compose_benchmark_contexts,
    load_benchmark_cases,
    render_comparison_summary,
    render_final_h6_report,
    render_h5_final_context_summary,
    render_markdown_report,
    run_dense_benchmark,
)
from app.services.pedagogical_retrieval_ranker import PedagogicalRankingRequest
from app.services.retrieval_service import RetrievalError, RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only dense/hybrid retrieval benchmark; no indexing, reranking, or LLM calls.")
    parser.add_argument("--fixture", type=Path, default=Path("tests/fixtures/retrieval_benchmark.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=("dense", "hybrid", "pedagogical", "compare", "compare-h4", "compare-h4-1", "compare-h4-2", "compare-h5", "compare-h5-1", "compare-h5-2", "final-h6"), default="dense")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cases = load_benchmark_cases(args.fixture)
    provider = get_embedding_provider()
    qdrant = QdrantService()

    try:
        with SessionLocal() as db:
            modes = (
                ("dense", "hybrid", "pedagogical") if args.mode == "final-h6"
                else ("dense", "hybrid") if args.mode == "compare"
                else ("hybrid", "pedagogical") if args.mode in {"compare-h4", "compare-h4-1", "compare-h4-2", "compare-h5", "compare-h5-1", "compare-h5-2"}
                else (args.mode,)
            )
            reports = []
            mode_runs = {}
            for mode in modes:
                effective_mode = "hybrid" if mode == "pedagogical" else mode
                service = RetrievalService(provider=provider, qdrant=qdrant, mode=effective_mode, pedagogical_ranking_enabled=mode == "pedagogical")
                runs = run_dense_benchmark(
                    cases=cases, search=service.search, db=db, top_k=args.top_k,
                    pedagogical_request_for_case=(lambda case: PedagogicalRankingRequest(intent=case.intent, cefr_level=case.cefr_level, skills=case.skills)) if mode == "pedagogical" else None,
                    composition_pool_size=20 if args.mode in {"compare-h5-1", "compare-h5-2", "final-h6"} and mode == "pedagogical" else None,
                )
                assert_effective_mode(runs, requested_mode=effective_mode)
                mode_runs[mode] = runs
                if args.mode != "final-h6":
                    reports.append(render_markdown_report(
                        cases=cases,
                        run_case=lambda case, runs=runs: runs[case.id],
                        model_id=provider.model_id,
                        top_k=args.top_k,
                        retrieval_mode=effective_mode,
                        dense_candidate_top_k=service.hybrid_dense_top_k if effective_mode == "hybrid" else None,
                        sparse_candidate_top_k=service.hybrid_sparse_top_k if effective_mode == "hybrid" else None,
                        rrf_k=service.rrf_k if effective_mode == "hybrid" else None,
                        pedagogical_ranking=mode == "pedagogical",
                    ))
            if args.mode == "compare":
                reports.append(render_comparison_summary(
                    cases=cases,
                    dense_runs=mode_runs["dense"],
                    hybrid_runs=mode_runs["hybrid"],
                    top_k=args.top_k,
                ))
            if args.mode in {"compare-h4", "compare-h4-1", "compare-h4-2"}:
                reports.append(render_comparison_summary(
                    cases=cases,
                    dense_runs=mode_runs["hybrid"],
                    hybrid_runs=mode_runs["pedagogical"],
                    top_k=args.top_k,
                    title=("Hybrid H3 vs Hybrid + Pedagogical H4.2 Summary" if args.mode == "compare-h4-2" else "Hybrid H3 vs Hybrid + Pedagogical H4.1 Summary" if args.mode == "compare-h4-1" else "Hybrid H3 vs Hybrid + Pedagogical H4 Summary"),
                    left_label="Hybrid H3",
                    right_label="Hybrid + H4.2" if args.mode == "compare-h4-2" else "Hybrid + H4.1" if args.mode == "compare-h4-1" else "Hybrid + H4",
                ))
            if args.mode in {"compare-h5", "compare-h5-1", "compare-h5-2"}:
                h5_runs = compose_benchmark_contexts(mode_runs["pedagogical"], cases=cases, db=db)
                reports.append(render_markdown_report(
                    cases=cases, run_case=lambda case, runs=h5_runs: runs[case.id], model_id=provider.model_id,
                    top_k=args.top_k, retrieval_mode="hybrid", dense_candidate_top_k=20,
                    sparse_candidate_top_k=20, rrf_k=60, pedagogical_ranking=True,
                ).replace("# Hybrid + Pedagogical ranking H4", "# H4.2 + Balanced Pedagogical Context H5"))
                reports.append(render_comparison_summary(
                    cases=cases, dense_runs=mode_runs["pedagogical"], hybrid_runs=h5_runs, top_k=args.top_k,
                    title="H4.2 ranked pool vs H5.2 deduplicated final context Summary" if args.mode == "compare-h5-2" else "H4.2 ranked pool vs H5.1 balanced final context Summary" if args.mode == "compare-h5-1" else "H4.2 ranked pool vs H5 balanced context Summary", left_label="H4.2", right_label="H5.2 context" if args.mode == "compare-h5-2" else "H5.1 context" if args.mode == "compare-h5-1" else "H5 context",
                ))
                reports.append(render_h5_final_context_summary(h5_runs))
            if args.mode == "final-h6":
                h5_runs = compose_benchmark_contexts(mode_runs["pedagogical"], cases=cases, db=db)
                reports.append(render_final_h6_report(
                    cases=cases,
                    dense_runs=mode_runs["dense"],
                    hybrid_runs=mode_runs["hybrid"],
                    pedagogical_runs=mode_runs["pedagogical"],
                    final_context_runs=h5_runs,
                    top_k=args.top_k,
                ))
            report = "\n\n".join(reports)
    except (RetrievalError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output or Path(
        "reports/retrieval_dense_h3.md" if args.mode == "dense"
        else "reports/retrieval_pedagogical_h4_1.md" if args.mode == "compare-h4-1"
        else "reports/retrieval_pedagogical_h4_2.md" if args.mode == "compare-h4-2"
        else "reports/retrieval_context_h5.md" if args.mode == "compare-h5"
        else "reports/retrieval_context_h5_1.md" if args.mode == "compare-h5-1"
        else "reports/retrieval_context_h5_2.md" if args.mode == "compare-h5-2"
        else "reports/retrieval_final_h6.md" if args.mode == "final-h6"
        else "reports/retrieval_pedagogical_h4.md" if args.mode in {"pedagogical", "compare-h4"}
        else "reports/retrieval_hybrid_h3.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"benchmark_cases={len(cases)}")
    print(f"reranking_applied=false")
    print(f"mode={args.mode}\nreport={output}")


if __name__ == "__main__":
    main()
