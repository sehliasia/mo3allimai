"""Optional cross-encoder reranking for canonical KnowledgeChunk results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from app.services.retrieval_service import RetrievalResult


class RerankingError(RuntimeError):
    """The optional reranker could not score otherwise valid retrieval candidates."""


@dataclass(frozen=True)
class RerankScore:
    index: int
    score: float


class RerankerProvider(Protocol):
    model_id: str

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankScore]: ...


def canonical_reranker_text(result: "RetrievalResult") -> str:
    """Build the compact, canonical cross-encoder input without vector payloads."""
    parts = [f"Document: {result.document_title}"]
    if result.heading_context:
        parts.append("Section: " + " > ".join(result.heading_context))
    parts.append("Content:\n" + result.content)
    return "\n\n".join(part for part in parts if part)


class RerankerService:
    def __init__(self, provider: RerankerProvider) -> None:
        self.provider = provider

    def rerank(
        self,
        query: str,
        candidates: Sequence["RetrievalResult"],
        *,
        top_k: int,
    ) -> list["RetrievalResult"]:
        if top_k < 1:
            raise RerankingError("top_k must be at least 1.")
        documents = [canonical_reranker_text(candidate) for candidate in candidates]
        try:
            scored = self.provider.rerank(query, documents, top_k)
        except Exception as exc:
            raise RerankingError("Reranker scoring failed.") from exc
        seen: set[int] = set()
        results: list["RetrievalResult"] = []
        for rank, item in enumerate(scored, start=1):
            if item.index in seen or not 0 <= item.index < len(candidates):
                raise RerankingError("Reranker returned invalid candidate indices.")
            seen.add(item.index)
            results.append(
                replace(
                    candidates[item.index],
                    rank=rank,
                    reranker_score=float(item.score),
                    reranked_rank=rank,
                )
            )
        return results
