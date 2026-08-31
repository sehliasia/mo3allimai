"""Safe, non-secret diagnostics for the configured retrieval pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.core.config import Settings


PIPELINE_VERSION = "pedagogical_retrieval_h6_v1"
VALIDATED_H6_DENSE_TOP_K = 20
VALIDATED_H6_SPARSE_TOP_K = 20
VALIDATED_H6_RRF_K = 60
VALIDATED_H6_COMPOSITION_POOL_SIZE = 20


@dataclass(frozen=True)
class EffectivePedagogicalRetrievalConfig:
    pipeline_version: str
    profile: str
    retrieval_mode: str
    dense: bool
    sparse: bool
    rrf: bool
    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    pedagogical_ranking: bool
    context_composition: bool
    composition_pool_size: int
    reranker: bool
    context_max_chunks: int
    context_max_tokens: int


def resolve_effective_retrieval_pipeline(settings: Settings) -> EffectivePedagogicalRetrievalConfig:
    """Resolve one profile before production services interpret individual flags."""
    profile = getattr(settings, "pedagogical_retrieval_pipeline_mode", "legacy").strip().lower()
    dense_top_k = getattr(settings, "hybrid_dense_top_k", VALIDATED_H6_DENSE_TOP_K)
    sparse_top_k = getattr(settings, "hybrid_sparse_top_k", VALIDATED_H6_SPARSE_TOP_K)
    rrf_k = getattr(settings, "hybrid_rrf_k", VALIDATED_H6_RRF_K)
    composition_pool_size = getattr(
        settings, "pedagogical_context_composition_pool_size", VALIDATED_H6_COMPOSITION_POOL_SIZE,
    )
    if profile == "legacy":
        # Defaults keep small focused fakes and legacy callers backward compatible;
        # production Settings always supplies the explicit values.
        mode = getattr(settings, "retrieval_mode", "dense")
        ranking = bool(getattr(settings, "pedagogical_ranking_enabled", False) and mode == "hybrid")
        composition = bool(getattr(settings, "pedagogical_context_composition_enabled", False))
        reranker = bool(getattr(settings, "rag_reranker_enabled", False))
    elif profile == "hybrid":
        mode, ranking, composition, reranker = "hybrid", False, False, False
    elif profile == "validated_h6":
        mode, ranking, composition, reranker = "hybrid", True, True, False
        # A profile is a complete operator-facing contract: it deliberately
        # overrides legacy/manual knobs so one variable activates the H6 setup.
        dense_top_k = VALIDATED_H6_DENSE_TOP_K
        sparse_top_k = VALIDATED_H6_SPARSE_TOP_K
        rrf_k = VALIDATED_H6_RRF_K
        composition_pool_size = VALIDATED_H6_COMPOSITION_POOL_SIZE
    else:
        raise ValueError("PEDAGOGICAL_RETRIEVAL_PIPELINE_MODE must be legacy, hybrid, or validated_h6.")
    return EffectivePedagogicalRetrievalConfig(
        pipeline_version=PIPELINE_VERSION, profile=profile, retrieval_mode=mode,
        dense=True, sparse=mode == "hybrid", rrf=mode == "hybrid",
        dense_top_k=dense_top_k, sparse_top_k=sparse_top_k,
        rrf_k=rrf_k, pedagogical_ranking=ranking,
        context_composition=composition,
        composition_pool_size=composition_pool_size,
        reranker=reranker, context_max_chunks=getattr(settings, "rag_context_max_chunks", 6),
        context_max_tokens=getattr(settings, "rag_context_max_tokens", 1800),
    )


def effective_retrieval_pipeline(settings: Settings) -> dict[str, object]:
    """Return configuration only; this neither constructs providers nor loads models."""
    return asdict(resolve_effective_retrieval_pipeline(settings))


def validate_retrieval_pipeline(settings: Settings) -> None:
    """Fail early for invalid values without imposing new feature-flag coupling."""
    if settings.retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError("RETRIEVAL_MODE must be dense or hybrid.")
    if settings.pedagogical_context_composition_pool_size < 1:
        raise ValueError("PEDAGOGICAL_CONTEXT_COMPOSITION_POOL_SIZE must be at least 1.")
    profile = settings.pedagogical_retrieval_pipeline_mode.strip().lower()
    if profile not in {"legacy", "hybrid", "validated_h6"}:
        raise ValueError("PEDAGOGICAL_RETRIEVAL_PIPELINE_MODE must be legacy, hybrid, or validated_h6.")
    # Explicit profiles override manual flags.  Legacy remains the only mode
    # where individual feature-flag combinations are interpreted directly.
    if profile in {"hybrid", "validated_h6"}:
        return
    if profile == "hybrid" and (
        settings.retrieval_mode != "hybrid"
        or settings.pedagogical_ranking_enabled
        or settings.pedagogical_context_composition_enabled
        or settings.rag_reranker_enabled
    ):
        raise ValueError("hybrid profile requires hybrid retrieval with H4/H5 and reranker disabled.")
    if settings.pedagogical_context_composition_enabled and (
        settings.retrieval_mode != "hybrid" or not settings.pedagogical_ranking_enabled
    ):
        raise ValueError(
            "PEDAGOGICAL_CONTEXT_COMPOSITION_ENABLED requires hybrid retrieval and pedagogical ranking."
        )
