from types import SimpleNamespace

import pytest

from app.services.retrieval_pipeline import effective_retrieval_pipeline, validate_retrieval_pipeline


def _settings(**changes):
    values = {
        "retrieval_mode": "dense",
        "pedagogical_retrieval_pipeline_mode": "legacy",
        "pedagogical_ranking_enabled": False,
        "pedagogical_context_composition_enabled": False,
        "pedagogical_context_composition_pool_size": 20,
        "rag_reranker_enabled": False,
        "hybrid_dense_top_k": 20,
        "hybrid_sparse_top_k": 20,
        "hybrid_rrf_k": 60,
        "rag_context_max_chunks": 6,
        "rag_context_max_tokens": 1800,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_default_flags_keep_the_existing_safe_dense_pipeline():
    state = effective_retrieval_pipeline(_settings())
    assert state["profile"] == "legacy" and state["retrieval_mode"] == "dense"
    assert not state["pedagogical_ranking"] and not state["context_composition"]


def test_h6_activation_configuration_resolves_to_the_accepted_pipeline():
    state = effective_retrieval_pipeline(_settings(
        retrieval_mode="hybrid", pedagogical_ranking_enabled=True,
        pedagogical_context_composition_enabled=True,
    ))
    assert state["retrieval_mode"] == "hybrid" and state["pedagogical_ranking"]
    assert state["context_composition"] and state["composition_pool_size"] == 20


def test_validated_h6_profile_has_one_deterministic_effective_configuration():
    settings = _settings(
        pedagogical_retrieval_pipeline_mode="validated_h6", retrieval_mode="dense",
        hybrid_dense_top_k=3, hybrid_sparse_top_k=4, hybrid_rrf_k=5,
        pedagogical_context_composition_pool_size=1, rag_reranker_enabled=True,
    )
    validate_retrieval_pipeline(settings)
    state = effective_retrieval_pipeline(settings)
    assert state["pipeline_version"] == "pedagogical_retrieval_h6_v1"
    assert state["profile"] == "validated_h6"
    assert (state["dense_top_k"], state["sparse_top_k"], state["rrf_k"]) == (20, 20, 60)
    assert state["reranker"] is False and state["context_max_chunks"] == 6 and state["context_max_tokens"] == 1800


def test_supported_flag_combinations_keep_h3_and_h4_distinct_from_full_h6():
    h3 = _settings(retrieval_mode="hybrid")
    h4 = _settings(retrieval_mode="hybrid", pedagogical_ranking_enabled=True)
    for settings in (h3, h4):
        validate_retrieval_pipeline(settings)
    assert effective_retrieval_pipeline(h3)["pedagogical_ranking"] is False
    assert effective_retrieval_pipeline(h4)["pedagogical_ranking"] is True
    assert effective_retrieval_pipeline(h4)["context_composition"] is False


def test_context_composition_without_hybrid_h4_fails_explicitly():
    with pytest.raises(ValueError, match="requires hybrid retrieval"):
        validate_retrieval_pipeline(_settings(pedagogical_context_composition_enabled=True))


def test_invalid_pipeline_values_fail_early_without_constructing_providers():
    with pytest.raises(ValueError, match="RETRIEVAL_MODE"):
        validate_retrieval_pipeline(_settings(retrieval_mode="unknown"))
    with pytest.raises(ValueError, match="POOL_SIZE"):
        validate_retrieval_pipeline(_settings(pedagogical_context_composition_pool_size=0))
    with pytest.raises(ValueError, match="PEDAGOGICAL_RETRIEVAL_PIPELINE_MODE"):
        validate_retrieval_pipeline(_settings(pedagogical_retrieval_pipeline_mode="unsupported"))
