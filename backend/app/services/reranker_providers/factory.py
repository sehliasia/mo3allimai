from app.core.config import Settings, get_settings
from app.services.reranker_service import RerankerProvider

from .qwen3_reranker_provider import Qwen3RerankerProvider


def get_reranker_provider(settings: Settings | None = None) -> RerankerProvider:
    settings = settings or get_settings()
    return Qwen3RerankerProvider(
        model_id=settings.rag_reranker_model_id,
        device=settings.rag_reranker_device,
        batch_size=settings.rag_reranker_batch_size,
        instruction=settings.rag_reranker_instruction,
    )
