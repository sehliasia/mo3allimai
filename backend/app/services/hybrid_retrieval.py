"""Pure deterministic Reciprocal Rank Fusion; no database, model, or Qdrant calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FusedCandidate:
    identity: str
    hit: Any
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None
    rrf_score: float


def fuse_rrf(*, dense_hits: list[Any], sparse_hits: list[Any], rrf_k: int, identity) -> list[FusedCandidate]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")
    candidates: dict[str, dict[str, Any]] = {}
    for arm, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
        for rank, hit in enumerate(hits, start=1):
            key = identity(hit)
            record = candidates.setdefault(key, {"hit": hit, "dense_rank": None, "dense_score": None, "sparse_rank": None, "sparse_score": None, "rrf_score": 0.0})
            record[f"{arm}_rank"] = rank
            record[f"{arm}_score"] = float(getattr(hit, "score", 0.0))
            # Prefer dense as the representative hit only for payload hydration;
            # ranking itself remains strictly rank-based RRF.
            if arm == "dense":
                record["hit"] = hit
            record["rrf_score"] += 1.0 / (rrf_k + rank)
    fused = [FusedCandidate(identity=key, **record) for key, record in candidates.items()]
    return sorted(
        fused,
        key=lambda item: (-item.rrf_score, min(item.dense_rank or 10**9, item.sparse_rank or 10**9), item.identity),
    )
