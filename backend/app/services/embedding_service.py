"""Embedding preparation and synchronization state; no model or vector store is used here."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Iterator
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.knowledge_document import (
    KnowledgeChunk,
    KnowledgeChunkEmbeddingStatus,
    KnowledgeDocument,
)


@dataclass(frozen=True)
class EmbeddingEligibility:
    eligible: bool
    reason: str | None = None


@dataclass(frozen=True)
class EmbeddingCandidate:
    chunk_id: int
    document_id: int
    chunk_hash: str
    embedding_text: str
    embedding_input_hash: str
    vector_point_id: str
    payload: dict[str, object]


@dataclass(frozen=True)
class EmbeddingReport:
    scanned: int
    prepared: int
    skipped: int
    batches: int


@dataclass(frozen=True)
class EmbeddingBatch:
    """One bounded unit for a future provider + vector-store transaction."""

    candidates: list[EmbeddingCandidate]
    scanned: int
    skipped: int


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, queries: list[str]) -> list[list[float]]: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicFakeEmbeddingProvider:
    """Offline-only provider used by tests; it is not a semantic embedding model."""

    model_id = "fake-deterministic-v1"
    dimension = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                int.from_bytes(
                    hashlib.sha256(text.encode("utf-8")).digest()[index * 4 : (index + 1) * 4],
                    "big",
                )
                / 2**32
                for index in range(self.dimension)
            ]
            for text in texts
        ]

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        return self.embed_documents(queries)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible Phase 4A alias for document embeddings."""
        return self.embed_documents(texts)


def is_chunk_embedding_eligible(chunk: KnowledgeChunk) -> EmbeddingEligibility:
    text = chunk.content_for_embedding or ""
    if not text.strip():
        return EmbeddingEligibility(False, "empty_embedding_text")
    if "<!-- image" in text.lower():
        return EmbeddingEligibility(False, "image_placeholder")
    if not isinstance(chunk.token_count, int) or chunk.token_count <= 0:
        return EmbeddingEligibility(False, "invalid_token_count")
    if chunk.quality_status == "failed":
        return EmbeddingEligibility(False, "final_quality_failed")
    if (chunk.chunk_metadata or {}).get("structural_quality") == "layout_unreliable":
        return EmbeddingEligibility(False, "layout_unreliable")
    return EmbeddingEligibility(True)


def embedding_input_hash(text: str, *, model_id: str, config_version: str) -> str:
    normalized = " ".join(text.split())
    payload = json.dumps(
        {"text": normalized, "model_id": model_id, "config_version": config_version},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def vector_point_id(chunk: KnowledgeChunk) -> str:
    """Stable per semantic occurrence, including index so duplicate text cannot collide."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mo3allimai:knowledge:{chunk.document_id}:{chunk.chunk_hash}:{chunk.chunk_index}"))


def build_qdrant_payload(chunk: KnowledgeChunk, document: KnowledgeDocument) -> dict[str, object]:
    metadata = chunk.chunk_metadata or {}
    return {
        "document_id": chunk.document_id,
        "chunk_hash": chunk.chunk_hash,
        "ingestion_version": chunk.ingestion_version,
        "source_page_start": chunk.source_page_start,
        "source_page_end": chunk.source_page_end,
        "content_type": chunk.content_type,
        "headings": chunk.heading_context,
        "language": document.language or metadata.get("language"),
        "document_type": document.document_type or metadata.get("document_type"),
        "cefr_level": document.cefr_level or metadata.get("level"),
        "skill": document.skill or metadata.get("skill"),
        "structural_quality": metadata.get("structural_quality"),
        "has_image": bool(metadata.get("has_image")),
        "requires_vision": bool(metadata.get("requires_vision")),
    }


class EmbeddingService:
    def __init__(
        self,
        *,
        model_id: str | None = None,
        config_version: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
    ):
        settings = get_settings()
        self.model_id = model_id or settings.rag_embedding_model_id
        self.dimension = dimension or settings.rag_embedding_dimension
        base_config_version = config_version or settings.rag_embedding_config_version
        # Query instruction is intentionally absent: changing a query-only
        # prompt must not force document re-embedding.
        self.config_version = f"{base_config_version}|dimension={self.dimension}"
        self.batch_size = batch_size or settings.rag_embedding_batch_size

    def iter_embedding_batches(
        self,
        db: Session,
        *,
        document_ids: list[int] | None = None,
        force: bool = False,
    ) -> Iterator[EmbeddingBatch]:
        """Yield bounded pending batches without loading the corpus into memory.

        Phase 4B must embed and upsert one yielded batch before advancing to the
        next one, then call :meth:`mark_indexed` only after its upsert succeeds.
        """
        query = (
            select(KnowledgeChunk, KnowledgeDocument)
            .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
            .order_by(KnowledgeChunk.id)
        )
        if document_ids:
            query = query.where(KnowledgeChunk.document_id.in_(document_ids))
        candidates: list[EmbeddingCandidate] = []
        scanned = skipped = 0
        for chunk, document in db.execute(query).yield_per(self.batch_size):
            scanned += 1
            decision = is_chunk_embedding_eligible(chunk)
            expected_hash = embedding_input_hash(
                chunk.content_for_embedding,
                model_id=self.model_id,
                config_version=self.config_version,
            )
            if not decision.eligible:
                skipped += 1
                continue
            if (
                not force
                and chunk.embedding_status == KnowledgeChunkEmbeddingStatus.indexed
                and chunk.embedding_model == self.model_id
                and chunk.embedding_input_hash == expected_hash
            ):
                skipped += 1
                continue
            point_id = vector_point_id(chunk)
            chunk.embedding_input_hash = expected_hash
            chunk.embedding_model = self.model_id
            chunk.vector_point_id = point_id
            chunk.embedding_status = KnowledgeChunkEmbeddingStatus.pending
            chunk.embedding_error = None
            candidates.append(
                EmbeddingCandidate(
                    chunk.id,
                    chunk.document_id,
                    chunk.chunk_hash,
                    chunk.content_for_embedding,
                    expected_hash,
                    point_id,
                    build_qdrant_payload(chunk, document),
                )
            )
            if len(candidates) == self.batch_size:
                yield EmbeddingBatch(candidates, scanned, skipped)
                candidates = []
                scanned = skipped = 0

        if candidates or scanned or skipped:
            yield EmbeddingBatch(candidates, scanned, skipped)

    @staticmethod
    def mark_indexed(db: Session, candidates: list[EmbeddingCandidate]) -> None:
        """Call only after Qdrant has confirmed every upsert in a batch."""
        now = datetime.now(timezone.utc)
        for candidate in candidates:
            chunk = db.get(KnowledgeChunk, candidate.chunk_id)
            if chunk and chunk.embedding_input_hash == candidate.embedding_input_hash:
                chunk.embedding_status = KnowledgeChunkEmbeddingStatus.indexed
                chunk.embedded_at = now
                chunk.embedding_error = None

    @staticmethod
    def mark_failed(db: Session, candidates: list[EmbeddingCandidate], error: str) -> None:
        """Persist a safe batch failure without ever claiming an indexed vector."""
        safe_error = error.strip().replace("\n", " ")[:1000] or "Embedding batch failed."
        for candidate in candidates:
            chunk = db.get(KnowledgeChunk, candidate.chunk_id)
            if chunk and chunk.embedding_input_hash == candidate.embedding_input_hash:
                chunk.embedding_status = KnowledgeChunkEmbeddingStatus.failed
                chunk.embedding_error = safe_error
