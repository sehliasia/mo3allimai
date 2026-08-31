"""Qwen3 multilingual dense embedding provider for controlled benchmarking."""

from __future__ import annotations

from app.core.config import get_settings

from .sentence_transformer_provider import SentenceTransformerEmbeddingProvider


class Qwen3EmbeddingProvider(SentenceTransformerEmbeddingProvider):
    """Uses Qwen's instruction-aware query format and Matryoshka dimensions."""

    # Qwen/Qwen3-Embedding-0.6B, the chosen production model, supports
    # Matryoshka output dimensions up to 1024.
    max_dimension = 1024
    min_dimension = 32

    def __init__(
        self,
        *,
        model_id: str | None = None,
        dimension: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        query_instruction: str | None = None,
        model_loader=None,
    ) -> None:
        settings = get_settings()
        selected_dimension = dimension or settings.rag_embedding_dimension
        if not self.min_dimension <= selected_dimension <= self.max_dimension:
            raise ValueError("Qwen3-Embedding-0.6B supports dimensions from 32 to 1024.")
        super().__init__(
            model_id=model_id or settings.rag_embedding_model_id,
            dimension=selected_dimension,
            device=device or settings.rag_embedding_device,
            batch_size=batch_size or settings.rag_embedding_batch_size,
            query_instruction=query_instruction or settings.rag_embedding_query_instruction,
            max_seq_length=settings.rag_embedding_max_seq_length,
            cpu_threads=settings.rag_embedding_cpu_threads,
            model_loader=model_loader,
        )

    def format_query(self, query: str) -> str:
        return f"Instruct: {self.query_instruction}\nQuery: {query}"
