"""Read-only, deterministic reporting helpers for dense retrieval benchmarks.

This module deliberately consumes :class:`RetrievalResult` objects only.  It
does not embed, query Qdrant, call an LLM, or mutate PostgreSQL; the CLI owns
the production ``RetrievalService.search(..., rerank=False)`` call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from app.services.pedagogical_knowledge_service import PedagogicalRole
from app.services.retrieval_service import RetrievalResult
from app.services.pedagogical_retrieval_ranker import PedagogicalRankingRequest, PedagogicalRetrievalRanker
from app.services.pedagogical_context_composer import CompositionCategory, PedagogicalContextComposer


BENCHMARK_VERSION = 1
_VALID_SKILLS = frozenset({"speaking", "listening", "reading", "writing"})
_VALID_INTENTS = frozenset({"activity", "exercise", "dialogue", "task", "methodology", "general"})
_ROLE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (PedagogicalRole.DIALOGUE, ("dialogue", "حوار")),
    (PedagogicalRole.EXERCISE, ("exercice", "exercise", "worksheet_exercise", "تمرين")),
    (PedagogicalRole.TASK, ("jeu de rôle", "role play", "role-play", "tâche", "mission", "consigne", "مهمة", "تعليمات")),
    (PedagogicalRole.ASSESSMENT, ("évaluation", "evaluation", "correction", "assessment", "تقويم", "تصحيح")),
    (PedagogicalRole.ACTIVITY, ("activité", "activity", "نشاط")),
    (PedagogicalRole.METHODOLOGY, ("méthodolog", "methodolog", "démarche", "déroulement", "pedagogical guide", "guide pédagogique")),
)
_CONCRETE_ROLES = frozenset({PedagogicalRole.ACTIVITY, PedagogicalRole.EXERCISE, PedagogicalRole.DIALOGUE, PedagogicalRole.TASK})


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    id: str
    query: str
    cefr_level: str
    skills: tuple[str, ...]
    intent: str
    expected_topics: tuple[str, ...]
    desired_content_types: tuple[str, ...]
    language: str


def load_benchmark_cases(path: Path) -> list[RetrievalBenchmarkCase]:
    """Load the checked-in benchmark fixture and reject malformed cases early."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load retrieval benchmark fixture: {path}") from exc
    if not isinstance(payload, dict) or payload.get("version") != BENCHMARK_VERSION:
        raise ValueError(f"Retrieval benchmark fixture must declare version {BENCHMARK_VERSION}.")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Retrieval benchmark fixture must contain a non-empty cases array.")
    cases: list[RetrievalBenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Benchmark case {index} must be an object.")
        required_text = ("id", "query", "cefr_level", "intent", "language")
        if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in required_text):
            raise ValueError(f"Benchmark case {index} has a missing required text field.")
        case_id = raw["id"].strip()
        if case_id in seen_ids:
            raise ValueError(f"Duplicate benchmark case id: {case_id}")
        seen_ids.add(case_id)
        level = raw["cefr_level"].strip().upper()
        skills = _string_tuple(raw.get("skills"), field="skills", case_id=case_id)
        topics = _string_tuple(raw.get("expected_topics"), field="expected_topics", case_id=case_id)
        desired = _string_tuple(raw.get("desired_content_types"), field="desired_content_types", case_id=case_id)
        if not topics:
            raise ValueError(f"Benchmark case {case_id} must contain expected_topics.")
        if any(skill not in _VALID_SKILLS for skill in skills):
            raise ValueError(f"Benchmark case {case_id} contains an unsupported skill.")
        if raw["intent"].strip() not in _VALID_INTENTS:
            raise ValueError(f"Benchmark case {case_id} contains an unsupported intent.")
        cases.append(RetrievalBenchmarkCase(
            id=case_id, query=raw["query"].strip(), cefr_level=level, skills=skills,
            intent=raw["intent"].strip(), expected_topics=topics,
            desired_content_types=desired, language=raw["language"].strip().lower(),
        ))
    return cases


def _string_tuple(value: Any, *, field: str, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Benchmark case {case_id} field {field} must be a list of non-empty strings.")
    return tuple(item.strip() for item in value)


def pedagogical_role(result: RetrievalResult) -> str:
    """Use the existing role evidence policy for reporting, without reordering results."""
    if result.content_type == "worksheet_exercise":
        return PedagogicalRole.EXERCISE
    evidence = " ".join((result.content_type, *result.heading_context, result.content)).casefold()
    for role, markers in _ROLE_MARKERS:
        if any(marker in evidence for marker in markers):
            return role
    return PedagogicalRole.REFERENCE if result.content_type in {"table", "list"} else PedagogicalRole.OTHER


def serialize_result(result: RetrievalResult, *, preview_chars: int = 220) -> dict[str, Any]:
    """Serialize a bounded diagnostic view; canonical content is never altered."""
    preview = re.sub(r"\s+", " ", result.content or "").strip()
    if len(preview) > preview_chars:
        preview = preview[: max(0, preview_chars - 1)].rstrip() + "…"
    return {
        "rank": result.rank,
        "score": result.score,
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "document_title": result.document_title,
        "page_start": result.source_page_start,
        "page_end": result.source_page_end,
        "content_type": result.content_type,
        "pedagogical_role": pedagogical_role(result),
        "heading_context": list(result.heading_context),
        "language": result.language,
        "cefr_level": result.cefr_level,
        "structural_quality": result.structural_quality,
        "preview": preview,
        "appeared_in_dense": result.appeared_in_dense,
        "dense_rank": result.dense_rank,
        "dense_score": result.dense_score,
        "appeared_in_sparse": result.appeared_in_sparse,
        "sparse_rank": result.sparse_rank,
        "sparse_score": result.sparse_score,
        "rrf_score": result.rrf_score,
        "fused_rank": result.fused_rank,
        "pedagogical_adjustment_total": result.pedagogical_adjustment_total,
        "role_adjustment": result.role_adjustment,
        "concreteness_adjustment": result.concreteness_adjustment,
        "level_adjustment": result.level_adjustment,
        "skill_adjustment": result.skill_adjustment,
        "final_score": result.final_score,
        "final_rank": result.final_rank,
        "adjustment_reasons": result.adjustment_reasons,
        "pedagogical_role": result.pedagogical_role or pedagogical_role(result),
        "role_source": result.role_source,
        "is_concrete_classroom_material": result.is_concrete_classroom_material,
        "concreteness_reasons": result.concreteness_reasons,
        "skill_evidence_reason": result.skill_evidence_reason,
        "composition_category": result.composition_category,
        "composition_rank": result.composition_rank,
        "composition_selection_reason": result.composition_selection_reason,
        "neighbor_of": result.neighbor_of,
    }


def calculate_metrics(case: RetrievalBenchmarkCase, results: Iterable[RetrievalResult], *, k: int = 10) -> dict[str, dict[str, Any]]:
    """Calculate only metadata/lexical-evidence metrics, marking unknown data unavailable."""
    top = list(results)[:k]
    roles = [pedagogical_role(result) for result in top]
    desired_types = {value.casefold() for value in case.desired_content_types}
    metadata_levels = [result.cefr_level for result in top if result.cefr_level]
    topic_terms = tuple(term.casefold() for term in case.expected_topics)
    topic_matches = [
        any(term in " ".join((*result.heading_context, result.content)).casefold() for term in topic_terms)
        for result in top
    ]
    duplicate_keys = [
        (result.document_id, result.source_page_start, result.source_page_end, " ".join(result.heading_context).casefold())
        for result in top
    ]
    duplicate_count = len(duplicate_keys) - len(set(duplicate_keys))
    return {
        "TOPIC_RELEVANCE_AT_K": _metric(bool(top), sum(topic_matches) / len(top) if top else None, "lexical match against fixture expected_topics in canonical headings/content"),
        "SKILL_RELEVANCE_AT_K": _metric(False, None, "unavailable: RetrievalResult has no canonical per-chunk skills metadata"),
        "LEVEL_COMPATIBILITY_AT_K": _metric(bool(metadata_levels), sum(level.upper() == case.cefr_level for level in metadata_levels) / len(metadata_levels) if metadata_levels else None, "exact known CEFR metadata match; unlabelled chunks are excluded"),
        "DESIRED_CONTENT_TYPE_OR_ROLE_AT_K": _metric(bool(top), any(result.content_type.casefold() in desired_types or role.casefold() in desired_types for result, role in zip(top, roles)), "at least one requested raw content_type or deterministic pedagogical role"),
        "CONCRETE_RESOURCE_AT_K": _metric(bool(top), any(role in _CONCRETE_ROLES for role in roles), "at least one heuristic ACTIVITY/EXERCISE/DIALOGUE/TASK result"),
        "ACTIVITY_AT_K": _metric(bool(top), any(role == PedagogicalRole.ACTIVITY for role in roles), "heuristic pedagogical role"),
        "EXERCISE_AT_K": _metric(bool(top), any(role == PedagogicalRole.EXERCISE for role in roles), "heuristic pedagogical role"),
        "DIALOGUE_OR_TASK_AT_K": _metric(bool(top), any(role in {PedagogicalRole.DIALOGUE, PedagogicalRole.TASK} for role in roles), "heuristic pedagogical role"),
        "SOURCE_DIVERSITY_AT_K": _metric(bool(top), len({result.document_id for result in top}) if top else None, "distinct document_id count"),
        "DUPLICATE_RATE_AT_K": _metric(bool(top), duplicate_count / len(top) if top else None, "same document/page/heading diagnostic key"),
    }


def run_dense_benchmark(
    *,
    cases: Iterable[RetrievalBenchmarkCase],
    search: Callable[..., Any],
    db: Any,
    top_k: int,
    pedagogical_request_for_case: Callable[[RetrievalBenchmarkCase], PedagogicalRankingRequest] | None = None,
    composition_pool_size: int | None = None,
) -> dict[str, tuple[list[RetrievalResult], dict[str, Any]]]:
    """Execute only the existing dense search boundary for each fixture case.

    ``search`` is injected so tests can prove the benchmark neither invokes a
    reranker nor any generative provider.  The benchmark intentionally leaves
    all metadata filters unset: requested learner level/language are relevance
    expectations, not source metadata constraints.
    """
    output: dict[str, tuple[list[RetrievalResult], dict[str, Any]]] = {}
    for case in cases:
        kwargs: dict[str, Any] = {"top_k": top_k, "rerank": False}
        if pedagogical_request_for_case is not None:
            kwargs["pedagogical_request"] = pedagogical_request_for_case(case)
        if composition_pool_size is not None:
            kwargs["composition_pool_size"] = composition_pool_size
        response = search(db, case.query, **kwargs)
        effective_mode = getattr(response, "retrieval_mode", "dense")
        output[case.id] = (
            list(response.results),
            {
                "stale_references_skipped": response.stale_references_skipped,
                "effective_retrieval_mode": effective_mode,
                "reranking_applied": bool(getattr(response, "reranking_applied", False)),
                "dense_candidate_count": int(getattr(response, "dense_candidate_count", 0)),
                "sparse_candidate_count": int(getattr(response, "sparse_candidate_count", 0)),
                "union_candidate_count": int(getattr(response, "union_candidate_count", 0)),
                "pedagogical_desired_in_union": getattr(response, "pedagogical_desired_in_union", None),
                "pedagogical_union_diagnostics": getattr(response, "pedagogical_union_diagnostics", None),
                "composition_candidates": getattr(response, "composition_candidates", None),
            },
        )
    return output


def assert_effective_mode(
    runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]], *, requested_mode: str
) -> None:
    """Prevent a benchmark label from claiming a mode the service did not execute."""
    effective_modes = {diagnostics.get("effective_retrieval_mode") for _, diagnostics in runs.values()}
    if effective_modes != {requested_mode}:
        raise ValueError(
            f"Benchmark requested {requested_mode} retrieval but effective modes were: {sorted(effective_modes)}."
        )
    if any(diagnostics.get("reranking_applied") for _, diagnostics in runs.values()):
        raise ValueError("Benchmark must run with reranking disabled.")


def compose_benchmark_contexts(
    runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]], *, cases: Iterable[RetrievalBenchmarkCase], db: Any = None
) -> dict[str, tuple[list[RetrievalResult], dict[str, Any]]]:
    """Benchmark-only H5 view over the same already ranked H4 candidate lists."""
    composer = PedagogicalContextComposer()
    from app.services.context_builder import ContextBuilder
    builder = ContextBuilder()
    output: dict[str, tuple[list[RetrievalResult], dict[str, Any]]] = {}
    for case in cases:
        results, diagnostics = runs[case.id]
        composition = composer.compose(diagnostics.get("composition_candidates") or results, intent=case.intent)
        context = builder.build(case.query, composition.selected, db=db)
        selected_results = context.included_results
        selected_ids = {item.chunk_id for item in composition.selected}
        final_counts = {category: sum(PedagogicalContextComposer.category(item) == category for item in selected_results) for category in ("CONCRETE_CLASSROOM_MATERIAL", "PEDAGOGICAL_GUIDANCE", "SUPPORTING_RESOURCE")}
        output[case.id] = (selected_results, {
            **diagnostics,
            **composition.diagnostics,
            "composition_candidate_pool_size": composition.candidate_pool_size,
            "selected_resource_count": len(composition.selected),
            "selected_document_count": len({item.document_id for item in composition.selected}),
            "max_chunks_per_document": max((sum(item.document_id == document_id for item in composition.selected) for document_id in {item.document_id for item in composition.selected}), default=0),
            "composition_category_counts": composition.category_counts,
            "concrete_material_available": composition.concrete_material_available,
            "estimated_token_count": sum(len(re.findall(r"\w+|[^\s\w]", item.content, re.UNICODE)) for item in composition.selected),
            "final_context_chunk_count": len(selected_results),
            "final_context_estimated_tokens": context.estimated_token_count,
            "final_context_token_budget": builder.max_tokens,
            "final_context_bounded": (
                len(selected_results) < len(composition.selected)
                or context.estimated_token_count < sum(
                    len(re.findall(r"\w+|[^\s\w]", item.content, re.UNICODE))
                    for item in composition.selected
                )
            ),
            "final_context_document_count": len({item.document_id for item in selected_results}),
            "final_context_max_chunks_per_document": max(
                (
                    sum(item.document_id == document_id for item in selected_results)
                    for document_id in {item.document_id for item in selected_results}
                ),
                default=0,
            ),
            "final_context_category_counts": final_counts,
            "final_context_concrete_retained": any(item.is_concrete_classroom_material for item in selected_results),
            "neighbor_count": sum(item.chunk_id not in selected_ids for item in selected_results),
            "neighbors_rejected_as_duplicates": context.neighbors_rejected_as_duplicates,
            "composer_duplicate_rate": _duplicate_rate(composition.selected),
            "final_context_duplicate_rate": _duplicate_rate(selected_results),
            "final_retained_extended_count": sum(
                (item.final_rank or item.rank) > 10 for item in selected_results if item.neighbor_of is None
            ),
            "final_retained_same_document_fallback_count": sum(
                item.composition_selection_reason == "fallback_fill_same_document" for item in selected_results
            ),
        })
    return output


def _duplicate_rate(results: Iterable[RetrievalResult]) -> float:
    items = list(results)
    if not items:
        return 0.0
    keys = [(item.document_id, item.source_page_start, item.source_page_end, tuple(item.heading_context)) for item in items]
    return (len(keys) - len(set(keys))) / len(keys)


def _metric(available: bool, value: Any, definition: str) -> dict[str, Any]:
    return {"available": available, "value": value, "definition": definition}


def render_markdown_report(
    *, cases: Iterable[RetrievalBenchmarkCase],
    run_case: Callable[[RetrievalBenchmarkCase], tuple[list[RetrievalResult], dict[str, Any]]],
    model_id: str,
    top_k: int,
    retrieval_mode: str = "dense",
    dense_candidate_top_k: int | None = None,
    sparse_candidate_top_k: int | None = None,
    rrf_k: int | None = None,
    pedagogical_ranking: bool = False,
) -> str:
    """Render a bounded, human-reviewable report from an injected read-only search."""
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError("Benchmark retrieval mode must be dense or hybrid.")
    title = "Dense retrieval" if retrieval_mode == "dense" else "Hybrid + Pedagogical ranking H4" if pedagogical_ranking else "Hybrid retrieval"
    lines = [
        f"# {title}", "",
        f"- Benchmark fixture version: {BENCHMARK_VERSION}",
        f"- Embedding model: `{model_id}`", f"- Retrieval mode: {retrieval_mode}",
        f"- Final Top-K: {top_k}", "- Reranker: false",
        "- Judgement note: topic and pedagogical-role metrics are deterministic heuristics; review previews for semantic relevance.", "",
    ]
    if retrieval_mode == "hybrid":
        lines[6:6] = [
            f"- Dense candidate Top-K: {dense_candidate_top_k}",
            f"- Sparse candidate Top-K: {sparse_candidate_top_k}",
            f"- RRF k: {rrf_k}",
            f"- Pedagogical ranking: {'enabled' if pedagogical_ranking else 'disabled'}",
        ]
    all_results: list[RetrievalResult] = []
    rank_one_results: list[RetrievalResult] = []
    for case in cases:
        results, diagnostics = run_case(case)
        all_results.extend(results[:top_k])
        rank_one_results.extend(results[:1])
        serialized = [serialize_result(result) for result in results]
        metrics = calculate_metrics(case, results, k=top_k)
        lines.extend([f"## {case.id}", "", f"**Query:** {case.query}", "", f"**Expected:** level `{case.cefr_level}` · skills `{', '.join(case.skills)}` · intent `{case.intent}` · language `{case.language}` · desired `{', '.join(case.desired_content_types)}`", ""])
        if retrieval_mode == "hybrid":
            lines.extend([
                "### Hybrid diagnostics", "",
                f"- Dense candidates: {diagnostics.get('dense_candidate_count', 0)}",
                f"- Sparse candidates: {diagnostics.get('sparse_candidate_count', 0)}",
                f"- Union candidates: {diagnostics.get('union_candidate_count', 0)}",
                f"- Final results: {len(results[:top_k])}",
                f"- Desired pedagogical candidate in H3 union: {diagnostics.get('pedagogical_desired_in_union', 'n/a')}",
                *_union_diagnostic_lines(diagnostics.get("pedagogical_union_diagnostics")),
                *_composition_diagnostic_lines(diagnostics),
                *_hybrid_provenance_lines(results[:top_k]), "",
                "| Final rank | Document | Page | Type / role | Composition | Concrete evidence | Skill evidence | Dense rank | Sparse rank | RRF score | Ped. adj. | Final score | Reasons | Heading | Preview |",
                "| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ])
        else:
            lines.extend(["| Rank | Document | Page | Type / role | Score | Heading | Preview |", "| ---: | --- | --- | --- | ---: | --- | --- |"])
        if serialized:
            for item in serialized:
                page = _page(item["page_start"], item["page_end"])
                heading = " > ".join(item["heading_context"]) or "—"
                if retrieval_mode == "hybrid":
                    fused_rank = item["fused_rank"] if item["fused_rank"] is not None else item["rank"]
                    rrf_score = item["rrf_score"] if item["rrf_score"] is not None else item["score"]
                    final_rank = item["final_rank"] if pedagogical_ranking and item["final_rank"] is not None else fused_rank
                    adjustment = item["pedagogical_adjustment_total"] if pedagogical_ranking else 0.0
                    final_score = item["final_score"] if pedagogical_ranking else None
                    reasons = ", ".join(item["adjustment_reasons"] or []) if pedagogical_ranking else "—"
                    concrete = "yes" if item["is_concrete_classroom_material"] else "no"
                    concrete_evidence = ", ".join(item["concreteness_reasons"] or []) or concrete
                    skill_evidence = item["skill_evidence_reason"] or "—"
                    composition = ", ".join(str(value) for value in (item["composition_category"], item["composition_rank"], item["composition_selection_reason"]) if value is not None) or "—"
                    if item["neighbor_of"] is not None:
                        composition = f"neighbor_of={item['neighbor_of']}"
                    lines.append(f"| {final_rank} | {_cell(item['document_title'])} | {page} | {item['content_type']} / {item['pedagogical_role']} | {_cell(composition)} | {_cell(concrete_evidence)} | {_cell(skill_evidence)} | {_number(item['dense_rank'])} | {_number(item['sparse_rank'])} | {rrf_score:.8f} | {adjustment:+.3f} | {_score(final_score)} | {_cell(reasons)} | {_cell(heading)} | {_cell(item['preview'])} |")
                else:
                    lines.append(f"| {item['rank']} | {_cell(item['document_title'])} | {page} | {item['content_type']} / {item['pedagogical_role']} | {item['score']:.5f} | {_cell(heading)} | {_cell(item['preview'])} |")
        else:
            lines.append("| — | No canonical results | — | — | — | — | — | — | — | — | — | — | — | — | — |" if retrieval_mode == "hybrid" else "| — | No canonical results | — | — | — | — | — |")
        lines.extend(["", "### Evaluation", ""])
        for name, metric in metrics.items():
            value = "unavailable" if not metric["available"] else str(metric["value"])
            lines.append(f"- **{name}**: {value} — {metric['definition']}")
        if retrieval_mode == "hybrid":
            role_metrics = pedagogical_ranking_metrics(case, results)
            lines.extend([
                f"- **TOP_3_DESIRED_PEDAGOGICAL_ROLE**: {role_metrics['top_3_contains_desired_role']}",
                f"- **TOP_5_DESIRED_PEDAGOGICAL_ROLE**: {role_metrics['top_5_contains_desired_role']}",
                f"- **FIRST_DESIRED_PEDAGOGICAL_ROLE_RANK**: {role_metrics['first_desired_role_rank']}",
                f"- **CONCRETE_PEDAGOGICAL_ITEMS_TOP_5**: {role_metrics['concrete_top_5']}",
                f"- **CONCRETE_PEDAGOGICAL_ITEMS_TOP_10**: {role_metrics['concrete_top_10']}",
            ])
        if diagnostics.get("stale_references_skipped"):
            lines.append(f"- **STALE_REFERENCES_SKIPPED**: {diagnostics['stale_references_skipped']}")
        lines.append("")
    cefr_total = sum(_is_cefr_companion(result) for result in all_results)
    cefr_rank_one = sum(_is_cefr_companion(result) for result in rank_one_results)
    lines.extend([
        "## CEFR Companion diagnostic", "",
        f"- CEFR Companion results across all Top-{top_k}s: {cefr_total}",
        f"- Queries with CEFR Companion at rank 1: {cefr_rank_one}",
        "- Diagnostic only; it does not alter ranking.", "",
    ])
    if pedagogical_ranking:
        concrete_by_type: dict[str, int] = {}
        structural = ambiguous = rejected = 0
        for result in all_results:
            if result.is_concrete_classroom_material:
                concrete_by_type[result.content_type] = concrete_by_type.get(result.content_type, 0) + 1
            reasons = result.concreteness_reasons or []
            structural += sum(reason not in {"canonical_worksheet_exercise"} and not reason.endswith("_ignored") for reason in reasons)
            rejected += sum(reason.endswith("_ignored") for reason in reasons)
            ambiguous += int(result.skill_evidence_reason == "ambiguous_multiskill_chunk")
        lines.extend(["## H4.2 signal diagnostics", "", f"- Concrete candidates by content type: {concrete_by_type}", f"- Concrete candidates due to strong structural evidence: {structural}", f"- Ambiguous multi-skill chunks neutralized: {ambiguous}", f"- Generic-descriptor / serialized-enumeration evidence rejected: {rejected}", ""])
    return "\n".join(lines)


def render_comparison_summary(
    *,
    cases: Iterable[RetrievalBenchmarkCase],
    dense_runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]],
    hybrid_runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]],
    top_k: int,
    title: str = "Dense vs Hybrid H3 Summary",
    left_label: str = "Dense",
    right_label: str = "Hybrid",
) -> str:
    """Render deterministic H3 diagnostics only; it never changes result ordering."""
    case_list = list(cases)
    metric_names = (
        "TOPIC_RELEVANCE_AT_K", "DESIRED_CONTENT_TYPE_OR_ROLE_AT_K", "CONCRETE_RESOURCE_AT_K",
        "ACTIVITY_AT_K", "EXERCISE_AT_K", "DIALOGUE_OR_TASK_AT_K", "SOURCE_DIVERSITY_AT_K", "DUPLICATE_RATE_AT_K",
    )
    dense_metrics = _aggregate_metrics(case_list, dense_runs, top_k=top_k, names=metric_names)
    hybrid_metrics = _aggregate_metrics(case_list, hybrid_runs, top_k=top_k, names=metric_names)
    lines = [f"# {title}", "", f"| Metric | {left_label} | {right_label} | Absolute delta |", "| --- | ---: | ---: | ---: |"]
    boolean_metrics = {"DESIRED_CONTENT_TYPE_OR_ROLE_AT_K", "CONCRETE_RESOURCE_AT_K", "ACTIVITY_AT_K", "EXERCISE_AT_K", "DIALOGUE_OR_TASK_AT_K"}
    for name in metric_names:
        dense_value = dense_metrics[name]
        hybrid_value = hybrid_metrics[name]
        if name in boolean_metrics:
            lines.append(f"| {name} | {dense_value}/{len(case_list)} | {hybrid_value}/{len(case_list)} | {hybrid_value - dense_value:+d} |")
        else:
            lines.append(f"| {name} | {dense_value:.3f} | {hybrid_value:.3f} | {hybrid_value - dense_value:+.3f} |")
    left_role = _aggregate_pedagogical_metrics(case_list, dense_runs)
    right_role = _aggregate_pedagogical_metrics(case_list, hybrid_runs)
    lines.extend(["", "## Pedagogical-ranking diagnostics", "", f"| Metric | {left_label} | {right_label} | Delta |", "| --- | ---: | ---: | ---: |"])
    for name in ("top_3_contains_desired_role", "top_5_contains_desired_role", "concrete_top_5", "concrete_top_10"):
        lines.append(f"| {name.upper()} | {left_role[name]} | {right_role[name]} | {right_role[name] - left_role[name]:+d} |")
    lines.append(f"| MEAN_FIRST_DESIRED_PEDAGOGICAL_ROLE_RANK (missing excluded) | {_mean_display(left_role['first_desired_role_ranks'])} | {_mean_display(right_role['first_desired_role_ranks'])} | {_mean_delta(left_role['first_desired_role_ranks'], right_role['first_desired_role_ranks'])} |")
    improved = unchanged = regressed = 0
    for case in case_list:
        dense_topic = calculate_metrics(case, dense_runs[case.id][0], k=top_k)["TOPIC_RELEVANCE_AT_K"]["value"] or 0.0
        hybrid_topic = calculate_metrics(case, hybrid_runs[case.id][0], k=top_k)["TOPIC_RELEVANCE_AT_K"]["value"] or 0.0
        if hybrid_topic > dense_topic:
            improved += 1
        elif hybrid_topic < dense_topic:
            regressed += 1
        else:
            unchanged += 1
    lines.extend(["", f"- Queries where topic relevance improved: {improved}", f"- Queries where topic relevance unchanged: {unchanged}", f"- Queries where topic relevance regressed: {regressed}"])
    return "\n".join(lines)


def _aggregate_pedagogical_metrics(cases: Iterable[RetrievalBenchmarkCase], runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {"top_3_contains_desired_role": 0, "top_5_contains_desired_role": 0, "concrete_top_5": 0, "concrete_top_10": 0, "first_desired_role_ranks": []}
    for case in cases:
        metrics = pedagogical_ranking_metrics(case, runs[case.id][0])
        for name in ("top_3_contains_desired_role", "top_5_contains_desired_role"):
            output[name] += int(bool(metrics[name]))
        for name in ("concrete_top_5", "concrete_top_10"):
            output[name] += int(metrics[name])
        if metrics["first_desired_role_rank"] is not None:
            output["first_desired_role_ranks"].append(int(metrics["first_desired_role_rank"]))
    return output


def _mean_display(values: list[int]) -> str:
    return "n/a" if not values else f"{sum(values) / len(values):.3f}"


def _mean_delta(left: list[int], right: list[int]) -> str:
    if not left or not right:
        return "n/a"
    return f"{sum(right) / len(right) - sum(left) / len(left):+.3f}"


def pedagogical_ranking_metrics(case: RetrievalBenchmarkCase, results: Iterable[RetrievalResult]) -> dict[str, int | bool | None]:
    """Benchmark-only role diagnostics; they do not participate in ranking."""
    items = list(results)
    desired = {value.casefold() for value in case.desired_content_types}
    ranker = PedagogicalRetrievalRanker()
    roles = [ranker.role(result).casefold() for result in items]
    matching = [index for index, (result, role) in enumerate(zip(items, roles), start=1) if result.content_type.casefold() in desired or role in desired]
    concrete = [result for result in items if ranker._is_concrete(result, ranker.role(result))]
    return {
        "top_3_contains_desired_role": any(rank <= 3 for rank in matching),
        "top_5_contains_desired_role": any(rank <= 5 for rank in matching),
        "first_desired_role_rank": matching[0] if matching else None,
        "concrete_top_5": sum(result in concrete for result in items[:5]),
        "concrete_top_10": sum(result in concrete for result in items[:10]),
    }


def render_h5_final_context_summary(runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]]) -> str:
    """Aggregate actual ContextBuilder outputs, not the intermediate composer list."""
    diagnostics = [item[1] for item in runs.values()]
    chunk_counts = [int(item.get("final_context_chunk_count", 0)) for item in diagnostics]
    token_counts = [int(item.get("final_context_estimated_tokens", 0)) for item in diagnostics]
    token_budgets = [int(item.get("final_context_token_budget", 1800)) for item in diagnostics]
    document_counts = [int(item.get("final_context_document_count", 0)) for item in diagnostics]
    max_per_document = [int(item.get("final_context_max_chunks_per_document", 0)) for item in diagnostics]
    categories = [item.get("final_context_category_counts", {}) for item in diagnostics]
    return "\n".join([
        "# H5 Final Context Metrics", "",
        f"- Mean/min/max final context chunk count: {sum(chunk_counts) / len(chunk_counts):.2f}/{min(chunk_counts)}/{max(chunk_counts)}",
        f"- Contexts reaching 6 chunks: {sum(count == 6 for count in chunk_counts)}",
        f"- Contexts with <=3 chunks: {sum(count <= 3 for count in chunk_counts)}",
        f"- Mean final token estimate: {sum(token_counts) / len(token_counts):.2f}",
        f"- Contexts exceeding final token budget: {sum(count > budget for count, budget in zip(token_counts, token_budgets))}",
        f"- Mean distinct documents / mean-max chunks per document: {sum(document_counts) / len(document_counts):.2f}/{sum(max_per_document) / len(max_per_document):.2f}-{max(max_per_document)}",
        f"- Strong concrete available primary/extended/retained: {sum(bool(item.get('strong_concrete_available_primary_top10')) for item in diagnostics)}/{sum(bool(item.get('strong_concrete_available_extended_top20')) for item in diagnostics)}/{sum(bool(item.get('final_context_concrete_retained')) for item in diagnostics)}",
        f"- Strong concrete selected by composer: {sum(bool(item.get('strong_concrete_selected_by_composer')) for item in diagnostics)}",
        f"- Total concrete/guidance/supporting final chunks: {sum(item.get(CompositionCategory.CONCRETE, 0) for item in categories)}/{sum(item.get(CompositionCategory.GUIDANCE, 0) for item in categories)}/{sum(item.get(CompositionCategory.SUPPORTING, 0) for item in categories)}",
        f"- Composer/final retained selections from extended pool: {sum(int(item.get('selected_from_extended_pool_count', 0)) for item in diagnostics)}/{sum(int(item.get('final_retained_extended_count', 0)) for item in diagnostics)}",
        f"- Composer/final retained same-document fallback fills: {sum(int(item.get('same_document_fallback_fill_count', 0)) for item in diagnostics)}/{sum(int(item.get('final_retained_same_document_fallback_count', 0)) for item in diagnostics)}",
        f"- Neighbors added/rejected as duplicates: {sum(int(item.get('neighbor_count', 0)) for item in diagnostics)}/{sum(int(item.get('neighbors_rejected_as_duplicates', 0)) for item in diagnostics)}",
        f"- Mean composer/final duplicate rate: {sum(float(item.get('composer_duplicate_rate', 0.0)) for item in diagnostics) / len(diagnostics):.3f}/{sum(float(item.get('final_context_duplicate_rate', 0.0)) for item in diagnostics) / len(diagnostics):.3f}",
        f"- Queries with final duplicates: {sum(float(item.get('final_context_duplicate_rate', 0.0)) > 0 for item in diagnostics)}",
        f"- Final contexts bounded by ContextBuilder: {sum(bool(item.get('final_context_bounded')) for item in diagnostics)}/{len(diagnostics)}",
    ])


_H6_RETRIEVAL_METRICS = (
    "TOPIC_RELEVANCE_AT_K", "DESIRED_CONTENT_TYPE_OR_ROLE_AT_K", "CONCRETE_RESOURCE_AT_K",
    "ACTIVITY_AT_K", "EXERCISE_AT_K", "DIALOGUE_OR_TASK_AT_K", "SOURCE_DIVERSITY_AT_K",
    "DUPLICATE_RATE_AT_K",
)
_H6_BOOLEAN_METRICS = frozenset({
    "DESIRED_CONTENT_TYPE_OR_ROLE_AT_K", "CONCRETE_RESOURCE_AT_K", "ACTIVITY_AT_K",
    "EXERCISE_AT_K", "DIALOGUE_OR_TASK_AT_K",
})


def _h6_stage_metrics(
    cases: list[RetrievalBenchmarkCase], runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]], *, top_k: int
) -> dict[str, float | int | str]:
    metrics = _aggregate_metrics(cases, runs, top_k=top_k, names=_H6_RETRIEVAL_METRICS)
    role = _aggregate_pedagogical_metrics(cases, runs)
    all_results = [result for case in cases for result in runs[case.id][0][:top_k]]
    metrics.update({
        "TOP_3_CONTAINS_DESIRED_ROLE": role["top_3_contains_desired_role"],
        "TOP_5_CONTAINS_DESIRED_ROLE": role["top_5_contains_desired_role"],
        "MEAN_FIRST_DESIRED_ROLE_RANK": _mean_display(role["first_desired_role_ranks"]),
        "STRONG_CONCRETE_TOP_5": sum(any(item.is_concrete_classroom_material for item in runs[case.id][0][:5]) for case in cases),
        "STRONG_CONCRETE_TOP_10": sum(any(item.is_concrete_classroom_material for item in runs[case.id][0][:top_k]) for case in cases),
        "CEFR_COMPANION_CHUNKS": sum(_is_cefr_companion(item) for item in all_results),
        "CEFR_COMPANION_RANK_1": sum(_is_cefr_companion(runs[case.id][0][0]) for case in cases if runs[case.id][0]),
    })
    return metrics


def _h6_diagnostic_classification(
    case: RetrievalBenchmarkCase,
    pedagogical_results: list[RetrievalResult],
    final_results: list[RetrievalResult],
    final_diagnostics: dict[str, Any],
) -> str:
    """Explain benchmark evidence only; this function never changes selection."""
    labels: list[str] = []
    desired_top10 = bool(calculate_metrics(case, pedagogical_results, k=10)["DESIRED_CONTENT_TYPE_OR_ROLE_AT_K"]["value"])
    desired_top20 = bool(calculate_metrics(
        case, list(final_diagnostics.get("composition_candidates") or pedagogical_results), k=20
    )["DESIRED_CONTENT_TYPE_OR_ROLE_AT_K"]["value"])
    desired_final = bool(calculate_metrics(case, final_results, k=10)["DESIRED_CONTENT_TYPE_OR_ROLE_AT_K"]["value"])
    if not desired_top10 and desired_top20:
        labels.append("RANKING_LIMITATION")
    if not desired_top20 and not final_diagnostics.get("strong_concrete_available_extended_top20"):
        labels.append("RETRIEVAL_COVERAGE_LIMITATION")
    if final_diagnostics.get("selected_from_extended_pool_count"):
        labels.append("COMPOSITION_RECOVERY")
    if desired_top10 and not desired_final:
        labels.append("COMPOSITION_LOSS")
    if not final_diagnostics.get("strong_concrete_available_extended_top20"):
        labels.append("NO_STRONG_CONCRETE_AVAILABLE")
    return ", ".join(labels) or "NO_LIMITATION_DETECTED"


def _h6_topic_transition(
    cases: list[RetrievalBenchmarkCase],
    previous: dict[str, tuple[list[RetrievalResult], dict[str, Any]]],
    current: dict[str, tuple[list[RetrievalResult], dict[str, Any]]],
    *, top_k: int,
    current_is_context: bool = False,
) -> tuple[int, int, int, list[tuple[str, float, float]]]:
    improved = unchanged = regressed = 0
    regressions: list[tuple[str, float, float]] = []
    for case in cases:
        before = float(calculate_metrics(case, previous[case.id][0], k=top_k)["TOPIC_RELEVANCE_AT_K"]["value"] or 0.0)
        after = float(calculate_metrics(case, current[case.id][0], k=top_k)["TOPIC_RELEVANCE_AT_K"]["value"] or 0.0)
        if after > before:
            improved += 1
        elif after < before:
            regressed += 1
            regressions.append((case.id, before, after))
        else:
            unchanged += 1
    return improved, unchanged, regressed, regressions


def render_final_h6_report(
    *,
    cases: Iterable[RetrievalBenchmarkCase],
    dense_runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]],
    hybrid_runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]],
    pedagogical_runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]],
    final_context_runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]],
    top_k: int = 10,
) -> str:
    """Render H6 from four already executed read-only benchmark stages."""
    case_list = list(cases)
    if not case_list:
        raise ValueError("H6 requires at least one benchmark case.")
    for runs in (dense_runs, hybrid_runs, pedagogical_runs, final_context_runs):
        if set(runs) != {case.id for case in case_list}:
            raise ValueError("Every H6 stage must use the identical fixture case ids.")

    stages = {
        "A — Dense retrieval Top-10": _h6_stage_metrics(case_list, dense_runs, top_k=top_k),
        "B — Hybrid RRF retrieval Top-10": _h6_stage_metrics(case_list, hybrid_runs, top_k=top_k),
        "C — H4.2 pedagogical retrieval Top-10": _h6_stage_metrics(case_list, pedagogical_runs, top_k=top_k),
        "D — H5.2 final bounded context": _h6_stage_metrics(case_list, final_context_runs, top_k=top_k),
    }
    final_diagnostics = [final_context_runs[case.id][1] for case in case_list]
    final_categories = [item.get("final_context_category_counts", {}) for item in final_diagnostics]
    lines = [
        "# H6 Final Retrieval and Context Benchmark", "",
        f"- Immutable fixture version: {BENCHMARK_VERSION}; cases: {len(case_list)}.",
        "- A/B/C evaluation unit: retrieval Top-10. D evaluation unit: actual bounded RAGContext.",
        "- Read-only benchmark: no indexing, no LLM evaluator, and Reranker OFF for every stage.",
        "- Structured CEFR PostgreSQL is authoritative and separate from resource slots.", "",
        "## Executive summary", "",
        "| Metric | A Dense (retrieval Top-10) | B Hybrid (retrieval Top-10) | C H4.2 (retrieval Top-10) | D H5.2 (final context) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in _H6_RETRIEVAL_METRICS:
        values = []
        for stage in stages.values():
            value = stage[metric]
            values.append(f"{value}/{len(case_list)}" if metric in _H6_BOOLEAN_METRICS else f"{float(value):.3f}")
        lines.append(f"| {metric} | {' | '.join(values)} |")
    for metric in ("TOP_3_CONTAINS_DESIRED_ROLE", "TOP_5_CONTAINS_DESIRED_ROLE", "MEAN_FIRST_DESIRED_ROLE_RANK", "STRONG_CONCRETE_TOP_5", "STRONG_CONCRETE_TOP_10"):
        values = [stage[metric] for stage in stages.values()]
        formatted = [f"{value}/{len(case_list)}" if metric != "MEAN_FIRST_DESIRED_ROLE_RANK" else str(value) for value in values]
        lines.append(f"| {metric} | {' | '.join(formatted)} |")
    lines.extend([
        "", "## Stage D final-context mechanics", "",
        f"- Mean/min/max chunks: {sum(int(item.get('final_context_chunk_count', 0)) for item in final_diagnostics) / len(final_diagnostics):.2f}/{min(int(item.get('final_context_chunk_count', 0)) for item in final_diagnostics)}/{max(int(item.get('final_context_chunk_count', 0)) for item in final_diagnostics)}",
        f"- Mean final token estimate: {sum(int(item.get('final_context_estimated_tokens', 0)) for item in final_diagnostics) / len(final_diagnostics):.2f}",
        f"- Contexts exceeding token budget: {sum(int(item.get('final_context_estimated_tokens', 0)) > int(item.get('final_context_token_budget', 1800)) for item in final_diagnostics)}",
        f"- Total final concrete / pedagogical guidance / supporting: {sum(item.get(CompositionCategory.CONCRETE, 0) for item in final_categories)} / {sum(item.get(CompositionCategory.GUIDANCE, 0) for item in final_categories)} / {sum(item.get(CompositionCategory.SUPPORTING, 0) for item in final_categories)}",
        f"- Mean distinct documents: {sum(int(item.get('final_context_document_count', 0)) for item in final_diagnostics) / len(final_diagnostics):.2f}; max chunks/document: {max(int(item.get('final_context_max_chunks_per_document', 0)) for item in final_diagnostics)}",
        f"- Neighbors added / structural neighbors rejected: {sum(int(item.get('neighbor_count', 0)) for item in final_diagnostics)} / {sum(int(item.get('neighbors_rejected_as_duplicates', 0)) for item in final_diagnostics)}",
        f"- Strong concrete primary / extended / selected / final: {sum(bool(item.get('strong_concrete_available_primary_top10')) for item in final_diagnostics)} / {sum(bool(item.get('strong_concrete_available_extended_top20')) for item in final_diagnostics)} / {sum(bool(item.get('strong_concrete_selected_by_composer')) for item in final_diagnostics)} / {sum(bool(item.get('final_context_concrete_retained')) for item in final_diagnostics)}",
        f"- Extended selections / retained: {sum(int(item.get('selected_from_extended_pool_count', 0)) for item in final_diagnostics)} / {sum(int(item.get('final_retained_extended_count', 0)) for item in final_diagnostics)}",
        f"- Same-document fallback selections / retained: {sum(int(item.get('same_document_fallback_fill_count', 0)) for item in final_diagnostics)} / {sum(int(item.get('final_retained_same_document_fallback_count', 0)) for item in final_diagnostics)}",
        f"- Mean composer / final duplicate rate: {sum(float(item.get('composer_duplicate_rate', 0.0)) for item in final_diagnostics) / len(final_diagnostics):.3f} / {sum(float(item.get('final_context_duplicate_rate', 0.0)) for item in final_diagnostics) / len(final_diagnostics):.3f}",
        "",
        "## Query-by-query matrix", "",
        "| Query | Dense topic | Hybrid topic | H4.2 topic | H5.2 topic | Dense role | Hybrid role | H4.2 role | H5.2 role | H4.2 strong Top-10 | H5.2 strong Top-20 | H5.2 strong final | Final chunks | Tokens | Docs | Dup. rate | Diagnostic |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for case in case_list:
        dense_result, hybrid_result = dense_runs[case.id][0], hybrid_runs[case.id][0]
        pedagogical_result, final_result = pedagogical_runs[case.id][0], final_context_runs[case.id][0]
        final_diag = final_context_runs[case.id][1]
        def metric(run: list[RetrievalResult], name: str) -> Any:
            return calculate_metrics(case, run, k=top_k)[name]["value"]
        diagnostic = _h6_diagnostic_classification(case, pedagogical_result, final_result, final_diag)
        lines.append(
            f"| {case.id} | {float(metric(dense_result, 'TOPIC_RELEVANCE_AT_K') or 0):.3f} | {float(metric(hybrid_result, 'TOPIC_RELEVANCE_AT_K') or 0):.3f} | {float(metric(pedagogical_result, 'TOPIC_RELEVANCE_AT_K') or 0):.3f} | {float(metric(final_result, 'TOPIC_RELEVANCE_AT_K') or 0):.3f} | "
            f"{bool(metric(dense_result, 'DESIRED_CONTENT_TYPE_OR_ROLE_AT_K'))} | {bool(metric(hybrid_result, 'DESIRED_CONTENT_TYPE_OR_ROLE_AT_K'))} | {bool(metric(pedagogical_result, 'DESIRED_CONTENT_TYPE_OR_ROLE_AT_K'))} | {bool(metric(final_result, 'DESIRED_CONTENT_TYPE_OR_ROLE_AT_K'))} | "
            f"{any(item.is_concrete_classroom_material for item in pedagogical_result[:top_k])} | {bool(final_diag.get('strong_concrete_available_extended_top20'))} | {bool(final_diag.get('final_context_concrete_retained'))} | {final_diag.get('final_context_chunk_count', 0)} | {final_diag.get('final_context_estimated_tokens', 0)} | {final_diag.get('final_context_document_count', 0)} | {float(final_diag.get('final_context_duplicate_rate', 0.0)):.3f} | {diagnostic} |"
        )

    lines.extend(["", "## Extended-pool recovery", "", "| Query | H4.2 rank | Content type | Role | Composition category | Final retained |", "| --- | ---: | --- | --- | --- | --- |"])
    recoveries = []
    for case in case_list:
        final_results, _ = final_context_runs[case.id]
        for result in final_results:
            if result.composition_selection_reason == "selected_extended_concrete_material":
                recoveries.append((case.id, result))
                lines.append(f"| {case.id} | {result.final_rank or result.rank} | {result.content_type} | {result.pedagogical_role or pedagogical_role(result)} | {result.composition_category} | yes |")
    if not recoveries:
        lines.append("| — | — | No final extended concrete recovery | — | — | — |")

    lines.extend(["", "## Topic regression analysis", ""])
    for label, previous, current in (("A → B", dense_runs, hybrid_runs), ("B → C", hybrid_runs, pedagogical_runs), ("C → D", pedagogical_runs, final_context_runs)):
        improved, unchanged, regressed, regressions = _h6_topic_transition(case_list, previous, current, top_k=top_k)
        lines.append(f"- {label}: improved {improved}; unchanged {unchanged}; regressed {regressed}.")
        for case_id, before, after in regressions:
            lines.append(f"  - {case_id}: {before:.3f} → {after:.3f}")

    lines.extend(["", "## Role regression analysis (C retrieval Top-10 → D final context)", ""])
    role_losses = 0
    for case in case_list:
        c_has_role = bool(calculate_metrics(case, pedagogical_runs[case.id][0], k=top_k)["DESIRED_CONTENT_TYPE_OR_ROLE_AT_K"]["value"])
        d_has_role = bool(calculate_metrics(case, final_context_runs[case.id][0], k=top_k)["DESIRED_CONTENT_TYPE_OR_ROLE_AT_K"]["value"])
        if c_has_role and not d_has_role:
            role_losses += 1
            diag = final_context_runs[case.id][1]
            ranks = [item.final_rank or item.rank for item in pedagogical_runs[case.id][0] if item.content_type.casefold() in {value.casefold() for value in case.desired_content_types} or (item.pedagogical_role or pedagogical_role(item)).casefold() in {value.casefold() for value in case.desired_content_types}]
            lines.append(f"- {case.id}: intent={case.intent}; desired={','.join(case.desired_content_types)}; C ranks={ranks}; primary={diag.get('strong_concrete_available_primary_top10')}; extended={diag.get('strong_concrete_available_extended_top20')}; retained_strong={diag.get('final_context_concrete_retained')}; composer_reason={diag.get('underfill_reason')}.")
    if not role_losses:
        lines.append("- None.")

    lines.extend([
        "", "## CEFR Companion diagnostic", "",
        "| Stage | CEFR Companion chunks | Queries with CEFR Companion rank 1 |", "| --- | ---: | ---: |",
        *[f"| {label} | {stage['CEFR_COMPANION_CHUNKS']} | {stage['CEFR_COMPANION_RANK_1']} |" for label, stage in stages.items()],
        f"- CEFR Companion chunks in final context: {sum(_is_cefr_companion(item) for case in case_list for item in final_context_runs[case.id][0])}.",
        "- Diagnostic only: no CEFR penalty, blacklist, or source quota is applied.",
        "", "## H6 Acceptance Evidence", "",
        f"- Hybrid topic relevance versus dense: {stages['B — Hybrid RRF retrieval Top-10']['TOPIC_RELEVANCE_AT_K']:.3f} versus {stages['A — Dense retrieval Top-10']['TOPIC_RELEVANCE_AT_K']:.3f}.",
        f"- H4.2 desired-role Top-10 coverage: {stages['C — H4.2 pedagogical retrieval Top-10']['DESIRED_CONTENT_TYPE_OR_ROLE_AT_K']}/{len(case_list)}.",
        f"- H5.2 final duplicate rate: {sum(float(item.get('final_context_duplicate_rate', 0.0)) for item in final_diagnostics) / len(final_diagnostics):.3f}; token overflows: {sum(int(item.get('final_context_estimated_tokens', 0)) > int(item.get('final_context_token_budget', 1800)) for item in final_diagnostics)}.",
        f"- Retrieval-coverage-limited cases: {sum('RETRIEVAL_COVERAGE_LIMITATION' in _h6_diagnostic_classification(case, pedagogical_runs[case.id][0], final_context_runs[case.id][0], final_context_runs[case.id][1]) for case in case_list)}.",
        "- READY_FOR_H7_REVIEW if the generated run preserves zero final duplicates and zero token overflows; otherwise NOT_READY_FOR_H7_REVIEW.",
    ])
    accepted = all(float(item.get("final_context_duplicate_rate", 0.0)) == 0.0 and int(item.get("final_context_estimated_tokens", 0)) <= int(item.get("final_context_token_budget", 1800)) for item in final_diagnostics)
    lines.append(f"- Decision for this run: {'READY_FOR_H7_REVIEW' if accepted else 'NOT_READY_FOR_H7_REVIEW'}.")
    return "\n".join(lines)


def _aggregate_metrics(cases: Iterable[RetrievalBenchmarkCase], runs: dict[str, tuple[list[RetrievalResult], dict[str, Any]]], *, top_k: int, names: Iterable[str]) -> dict[str, float | int]:
    metric_names = tuple(names)
    values: dict[str, list[Any]] = {name: [] for name in metric_names}
    for case in cases:
        metrics = calculate_metrics(case, runs[case.id][0], k=top_k)
        for name in metric_names:
            if metrics[name]["available"]:
                values[name].append(metrics[name]["value"])
    boolean_metrics = {"DESIRED_CONTENT_TYPE_OR_ROLE_AT_K", "CONCRETE_RESOURCE_AT_K", "ACTIVITY_AT_K", "EXERCISE_AT_K", "DIALOGUE_OR_TASK_AT_K"}
    return {name: (sum(bool(value) for value in values[name]) if name in boolean_metrics else (sum(values[name]) / len(values[name]) if values[name] else 0.0)) for name in metric_names}


def _hybrid_provenance_lines(results: Iterable[RetrievalResult]) -> list[str]:
    items = list(results)
    both = sum(result.appeared_in_dense and result.appeared_in_sparse for result in items)
    dense_only = sum(result.appeared_in_dense and not result.appeared_in_sparse for result in items)
    sparse_only = sum(not result.appeared_in_dense and result.appeared_in_sparse for result in items)
    return [f"- Final provenance — both arms: {both}; dense only: {dense_only}; sparse only: {sparse_only}"]


def _union_diagnostic_lines(diagnostics: dict[str, Any] | None) -> list[str]:
    if not diagnostics:
        return []
    best = diagnostics.get("best_desired_and_concrete")
    best_text = "none"
    if isinstance(best, dict):
        best_text = (
            f"fused rank {best.get('fused_rank')}, final rank {best.get('final_rank')}, "
            f"role {best.get('role')}, type {best.get('content_type')}"
        )
    return [
        f"- Desired-role candidate in union: {diagnostics.get('desired_role_candidate_exists_in_union')}",
        f"- Concrete candidate in union: {diagnostics.get('concrete_candidate_exists_in_union')}",
        f"- Desired and concrete candidate in union: {diagnostics.get('desired_and_concrete_candidate_exists_in_union')}",
        f"- Best desired/concrete candidate: {best_text}",
    ]


def _composition_diagnostic_lines(diagnostics: dict[str, Any]) -> list[str]:
    if "composition_candidate_pool_size" not in diagnostics:
        return []
    return [
        "### H5 composition diagnostics",
        f"- Composition candidate pool size: {diagnostics['composition_candidate_pool_size']}",
        f"- Selected resource count: {diagnostics.get('selected_resource_count', 'see final results')}",
        f"- Estimated token count: {diagnostics.get('estimated_token_count')}",
        f"- Category counts: {diagnostics.get('composition_category_counts')}",
        f"- Strong concrete material available: {diagnostics.get('concrete_material_available')}",
        f"- Distinct documents: {diagnostics.get('selected_document_count')}",
        f"- Maximum chunks from one document: {diagnostics.get('max_chunks_per_document')}",
        f"- Final ContextBuilder chunks/tokens: {diagnostics.get('final_context_chunk_count')} / {diagnostics.get('final_context_estimated_tokens')}",
        f"- Final ContextBuilder bounded: {diagnostics.get('final_context_bounded')} (budget {diagnostics.get('final_context_token_budget')})",
        f"- Final ContextBuilder documents / max chunks per document: {diagnostics.get('final_context_document_count')} / {diagnostics.get('final_context_max_chunks_per_document')}",
        f"- Final ContextBuilder category counts: {diagnostics.get('final_context_category_counts')}",
        f"- Strong concrete retained in final context: {diagnostics.get('final_context_concrete_retained')}",
        f"- Neighbors added by ContextBuilder: {diagnostics.get('neighbor_count')}",
        f"- Neighbors rejected as duplicates: {diagnostics.get('neighbors_rejected_as_duplicates')}",
        f"- Strong concrete primary/extended/selected/final: {diagnostics.get('strong_concrete_available_primary_top10')}/{diagnostics.get('strong_concrete_available_extended_top20')}/{diagnostics.get('strong_concrete_selected_by_composer')}/{diagnostics.get('final_context_concrete_retained')}",
        f"- Composer underfilled / reason: {diagnostics.get('composer_underfilled')} / {diagnostics.get('underfill_reason')}",
        f"- Composer/final duplicate rate: {diagnostics.get('composer_duplicate_rate')} / {diagnostics.get('final_context_duplicate_rate')}",
    ]


def _number(value: int | None) -> str:
    return "—" if value is None else str(value)


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _is_cefr_companion(result: RetrievalResult) -> bool:
    """Use the stable source title, not a deployment-specific document id."""
    return "cefr companion" in (result.document_title or "").casefold()


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _page(start: int | None, end: int | None) -> str:
    if start is None:
        return "—"
    return str(start) if end in {None, start} else f"{start}-{end}"
