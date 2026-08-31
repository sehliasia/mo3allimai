"""Explicit, lazy provider selection for future embedding jobs."""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.services.embedding_service import EmbeddingProvider

from .qwen3_provider import Qwen3EmbeddingProvider


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Construct a lazy provider; this function does not initialize its model."""
    settings = settings or get_settings()
    if settings.rag_embedding_provider == "qwen3":
        return Qwen3EmbeddingProvider(
            model_id=settings.rag_embedding_model_id,
            dimension=settings.rag_embedding_dimension,
            device=settings.rag_embedding_device,
            batch_size=settings.rag_embedding_batch_size,
            query_instruction=settings.rag_embedding_query_instruction,
        )
    raise ValueError(f"Unsupported RAG_EMBEDDING_PROVIDER: {settings.rag_embedding_provider!r}")
