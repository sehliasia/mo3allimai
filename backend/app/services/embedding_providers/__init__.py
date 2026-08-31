"""Lazy, local embedding providers for the production embedding pipeline."""

from .factory import get_embedding_provider
from .qwen3_provider import Qwen3EmbeddingProvider

__all__ = ["Qwen3EmbeddingProvider", "get_embedding_provider"]
