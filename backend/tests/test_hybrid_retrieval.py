from types import SimpleNamespace

from app.services.hybrid_retrieval import fuse_rrf


def _hit(identity, score):
    return SimpleNamespace(id=identity, score=score, payload={"chunk_id": identity})


def test_rrf_combines_both_arms_with_one_based_ranks_and_dedupes():
    fused = fuse_rrf(dense_hits=[_hit("a", .9), _hit("b", .8)], sparse_hits=[_hit("b", .7), _hit("c", .6)], rrf_k=60, identity=lambda hit: hit.id)
    by_id = {item.identity: item for item in fused}
    assert len(fused) == 3
    assert by_id["b"].rrf_score == 1 / 62 + 1 / 61
    assert by_id["a"].rrf_score == 1 / 61 and by_id["c"].rrf_score == 1 / 62
    assert by_id["b"].dense_rank == 2 and by_id["b"].sparse_rank == 1


def test_rrf_empty_arms_and_ties_are_deterministic():
    assert [item.identity for item in fuse_rrf(dense_hits=[], sparse_hits=[_hit("x", .5)], rrf_k=60, identity=lambda hit: hit.id)] == ["x"]
    assert fuse_rrf(dense_hits=[], sparse_hits=[], rrf_k=60, identity=lambda hit: hit.id) == []
    tied = fuse_rrf(dense_hits=[_hit("b", .5), _hit("a", .5)], sparse_hits=[], rrf_k=60, identity=lambda hit: hit.id)
    assert [item.identity for item in tied] == ["b", "a"]
