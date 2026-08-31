"""Build bounded, canonical source context from already retrieved chunks.

This module never performs vector search, model inference, or OCR work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge_document import KnowledgeChunk

if TYPE_CHECKING:
    from app.services.retrieval_service import RetrievalResult


_TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)


@dataclass(frozen=True)
class ContextChunk:
    result: "RetrievalResult"
    priority: float
    neighbor: bool = False
    same_page_sibling: bool = False
    neighbor_of: int | None = None


@dataclass(frozen=True)
class ContextSourceBlock:
    source_number: int
    document_id: int
    document_title: str
    chunk_ids: list[int]
    page_start: int | None
    page_end: int | None
    heading_context: list[str]
    content_type: str
    structural_quality: str | None
    has_image: bool
    requires_vision: bool
    image_not_interpreted: bool
    vector_scores: list[float]
    reranker_scores: list[float | None]
    original_ranks: list[int]
    reranked_ranks: list[int | None]
    content: str
    estimated_token_count: int

    def render(self) -> str:
        if self.page_start is None:
            pages = "unknown"
        elif self.page_end is None or self.page_end == self.page_start:
            pages = str(self.page_start)
        else:
            pages = f"{self.page_start}—{self.page_end}"
        lines = [f"[SOURCE {self.source_number}]", f"Document: {self.document_title}", f"Pages: {pages}"]
        if self.heading_context:
            lines.append("Section: " + " > ".join(self.heading_context))
        if self.image_not_interpreted:
            lines.append("Image note: image present but not interpreted")
        return "\n".join([*lines, "Content:", self.content])


@dataclass(frozen=True)
class RAGContext:
    query: str
    context_text: str
    source_blocks: list[ContextSourceBlock]
    included_chunk_ids: list[int]
    excluded_chunk_ids: list[int]
    estimated_token_count: int
    has_requires_vision: bool
    warnings: list[str]
    included_results: list["RetrievalResult"]
    neighbors_rejected_as_duplicates: int = 0


class ContextBuilder:
    def __init__(self, *, max_chunks: int = 6, max_tokens: int = 1800, neighbor_expansion: bool = True) -> None:
        if max_chunks < 1 or max_tokens < 1:
            raise ValueError("Context limits must be at least 1.")
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens
        self.neighbor_expansion = neighbor_expansion

    @staticmethod
    def _tokens(text: str) -> int:
        return len(_TOKEN_RE.findall(text))

    @staticmethod
    def _normalized(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def _compatible(left: "RetrievalResult", right: "RetrievalResult") -> bool:
        if left.document_id != right.document_id or left.heading_context != right.heading_context:
            return False
        if left.source_page_start is None or right.source_page_start is None:
            return False
        return abs(left.source_page_start - right.source_page_start) <= 1

    @staticmethod
    def _group_compatible(left: "RetrievalResult", right: "RetrievalResult") -> bool:
        return (
            left.document_id == right.document_id
            and left.heading_context == right.heading_context
            and left.source_page_start == right.source_page_start
            and left.source_page_end == right.source_page_end
        )

    def _is_duplicate(self, candidate: "RetrievalResult", kept: Iterable[ContextChunk]) -> bool:
        normalized = self._normalized(candidate.content)
        if not normalized:
            return True
        for existing in kept:
            other = self._normalized(existing.result.content)
            if normalized == other:
                return True
            # Remove repeated isolated headings while retaining complementary lists.
            if len(normalized) <= 100 and normalized in other:
                return True
        return False

    @staticmethod
    def _same_diagnostic_unit(left: "RetrievalResult", right: "RetrievalResult") -> bool:
        return (
            left.document_id == right.document_id
            and left.source_page_start == right.source_page_start
            and left.source_page_end == right.source_page_end
            and left.heading_context == right.heading_context
        )

    @staticmethod
    def _is_anchor(result: "RetrievalResult") -> bool:
        """Detect a short structural introduction without document-specific terms."""
        words = re.findall(r"\w+", result.content, re.UNICODE)
        lines = [line for line in result.content.splitlines() if line.strip()]
        return len(words) <= 24 and len(lines) <= 3

    @staticmethod
    def _is_meaningful(result: "RetrievalResult") -> bool:
        letters = sum(character.isalpha() for character in result.content)
        words = re.findall(r"\w+", result.content, re.UNICODE)
        return letters >= 3 and (len(words) >= 2 or letters >= 12)

    def _same_page_siblings(self, candidates: list[ContextChunk]) -> list[ContextChunk]:
        """Promote retrieved same-page evidence before any DB-only neighbor lookup."""
        anchors = [candidate for candidate in candidates[:1] if self._is_anchor(candidate.result)]
        siblings: list[ContextChunk] = []
        for anchor in anchors:
            page = anchor.result.source_page_start
            if page is None:
                continue
            retrieved_same_page = [
                candidate for candidate in candidates
                if candidate.result.chunk_id != anchor.result.chunk_id
                and candidate.result.document_id == anchor.result.document_id
                and candidate.result.source_page_start == page
                and candidate.result.source_page_end == anchor.result.source_page_end
                and candidate.result.structural_quality != "layout_unreliable"
                and self._is_meaningful(candidate.result)
            ]
            # Keep RetrievalService relevance/order intact: same section first,
            # then other useful candidates already returned for this page.
            retrieved_same_page.sort(key=lambda candidate: (
                candidate.result.heading_context != anchor.result.heading_context,
                candidate.priority,
                candidate.result.chunk_id,
            ))
            count = 0
            for retrieved in retrieved_same_page:
                result = retrieved.result
                kept = [item for item in [*candidates, *siblings] if item.result.chunk_id != result.chunk_id]
                if self._is_duplicate(result, kept):
                    continue
                siblings.append(ContextChunk(result, anchor.priority + 0.1 + (count / 100), same_page_sibling=True))
                count += 1
                if count == 2:
                    break
        return siblings

    def _neighbors(self, db: Session, candidates: list[ContextChunk]) -> list[ContextChunk]:
        """Return at most one compatible immediate neighbor per top candidate."""
        from app.services.retrieval_service import RetrievalResult

        ranked = candidates[:2]
        ids = [candidate.result.chunk_id for candidate in ranked]
        current = {chunk.id: chunk for chunk in db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids))).all()}
        neighbors: list[ContextChunk] = []
        for candidate in ranked:
            source = current.get(candidate.result.chunk_id)
            if source is None:
                continue
            nearby = db.scalars(
                select(KnowledgeChunk)
                .where(
                    KnowledgeChunk.document_id == source.document_id,
                    KnowledgeChunk.chunk_index.in_((source.chunk_index - 1, source.chunk_index + 1)),
                )
                .order_by(KnowledgeChunk.chunk_index)
            ).all()
            for chunk in nearby:
                metadata = chunk.chunk_metadata or {}
                neighbor = RetrievalResult(
                    rank=candidate.result.rank,
                    score=candidate.result.score,
                    vector_score=candidate.result.vector_score,
                    original_rank=candidate.result.original_rank,
                    chunk_id=chunk.id,
                    document_id=candidate.result.document_id,
                    document_title=candidate.result.document_title,
                    source_page_start=chunk.source_page_start,
                    source_page_end=chunk.source_page_end,
                    content_type=chunk.content_type,
                    language=candidate.result.language,
                    cefr_level=candidate.result.cefr_level,
                    structural_quality=metadata.get("structural_quality"),
                    has_image=bool(metadata.get("has_image")),
                    requires_vision=bool(metadata.get("requires_vision")),
                    heading_context=list(chunk.heading_context or []),
                    content=chunk.content,
                    neighbor_of=candidate.result.chunk_id,
                )
                if self._compatible(candidate.result, neighbor):
                    neighbors.append(ContextChunk(neighbor, candidate.priority + 1, neighbor=True, neighbor_of=candidate.result.chunk_id))
                    break
        return neighbors

    def build(self, query: str, results: list["RetrievalResult"], *, db: Session | None = None) -> RAGContext:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("A non-empty query is required.")
        excluded: list[int] = []
        candidates: list[ContextChunk] = []
        for priority, result in enumerate(results):
            if result.structural_quality == "layout_unreliable" or not result.content.strip():
                excluded.append(result.chunk_id)
                continue
            candidate = ContextChunk(result, priority)
            if self._is_duplicate(result, candidates):
                excluded.append(result.chunk_id)
                continue
            candidates.append(candidate)
        sibling_candidates = self._same_page_siblings(candidates)
        sibling_by_id = {candidate.result.chunk_id: candidate for candidate in sibling_candidates}
        candidates = [sibling_by_id.get(candidate.result.chunk_id, candidate) for candidate in candidates]
        neighbors_rejected_as_duplicates = 0
        if self.neighbor_expansion and db is not None:
            for neighbor in self._neighbors(db, candidates):
                if neighbor.result.chunk_id in {item.result.chunk_id for item in candidates}:
                    continue
                if self._is_duplicate(neighbor.result, candidates) or any(self._same_diagnostic_unit(neighbor.result, item.result) for item in candidates):
                    neighbors_rejected_as_duplicates += 1
                    continue
                candidates.append(neighbor)

        selected: list[ContextChunk] = []
        total_tokens = 0
        # Retrieval/reranker order wins; a neighbor is only considered after its parent.
        candidates.sort(key=lambda item: (item.priority, item.neighbor, item.result.chunk_id))
        for candidate in candidates:
            if len(selected) >= self.max_chunks:
                excluded.append(candidate.result.chunk_id)
                continue
            token_count = self._tokens(candidate.result.content)
            if total_tokens + token_count > self.max_tokens:
                excluded.append(candidate.result.chunk_id)
                continue
            selected.append(candidate)
            total_tokens += token_count

        blocks: list[ContextSourceBlock] = []
        block_representatives: list["RetrievalResult"] = []
        for candidate in selected:
            result = candidate.result
            if blocks and self._group_compatible(block_representatives[-1], result):
                previous = blocks[-1]
                merged = ContextSourceBlock(
                    **{**previous.__dict__, "chunk_ids": [*previous.chunk_ids, result.chunk_id], "page_start": min(filter(None, [previous.page_start, result.source_page_start]), default=None), "page_end": max(filter(None, [previous.page_end, result.source_page_end]), default=None), "has_image": previous.has_image or result.has_image, "requires_vision": previous.requires_vision or result.requires_vision, "image_not_interpreted": previous.image_not_interpreted or result.requires_vision, "vector_scores": [*previous.vector_scores, result.vector_score], "reranker_scores": [*previous.reranker_scores, result.reranker_score], "original_ranks": [*previous.original_ranks, result.original_rank], "reranked_ranks": [*previous.reranked_ranks, result.reranked_rank], "content": previous.content + "\n\n" + result.content, "estimated_token_count": previous.estimated_token_count + self._tokens(result.content)})
                blocks[-1] = merged
                block_representatives[-1] = result
                continue
            block = ContextSourceBlock(
                source_number=len(blocks) + 1, document_id=result.document_id, document_title=result.document_title,
                chunk_ids=[result.chunk_id], page_start=result.source_page_start, page_end=result.source_page_end,
                heading_context=result.heading_context, content_type=result.content_type, structural_quality=result.structural_quality,
                has_image=result.has_image, requires_vision=result.requires_vision, image_not_interpreted=result.requires_vision,
                vector_scores=[result.vector_score], reranker_scores=[result.reranker_score], original_ranks=[result.original_rank],
                reranked_ranks=[result.reranked_rank], content=result.content, estimated_token_count=self._tokens(result.content),
            )
            blocks.append(block)
            block_representatives.append(result)
        return RAGContext(
            query=normalized_query, context_text="\n\n".join(block.render() for block in blocks), source_blocks=blocks,
            included_chunk_ids=[item.result.chunk_id for item in selected], excluded_chunk_ids=list(dict.fromkeys(excluded)),
            estimated_token_count=total_tokens, has_requires_vision=any(block.requires_vision for block in blocks),
            warnings=["Some included sources contain images that were not interpreted."] if any(block.requires_vision for block in blocks) else [],
            included_results=[item.result for item in selected],
            neighbors_rejected_as_duplicates=neighbors_rejected_as_duplicates,
        )
