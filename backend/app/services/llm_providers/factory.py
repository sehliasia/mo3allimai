from app.core.config import Settings, get_settings
from app.services.llm_provider import LLMProvider

from .openai_compatible_provider import OpenAICompatibleLLMProvider


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    if settings.rag_llm_provider != "openai_compatible":
        raise ValueError(f"Unsupported RAG LLM provider: {settings.rag_llm_provider}.")
    return OpenAICompatibleLLMProvider(
        base_url=settings.rag_llm_base_url, api_key=settings.rag_llm_api_key or "", model_id=settings.rag_llm_model,
        timeout_seconds=settings.rag_llm_timeout_seconds, max_retries=settings.rag_llm_max_retries,
        retry_base_delay=settings.rag_llm_retry_base_delay, retry_max_delay=settings.rag_llm_retry_max_delay,
    )


def get_arabic_review_fallback_provider(settings: Settings | None = None) -> LLMProvider | None:
    """Build an optional independent reviewer provider without sharing secrets."""
    settings = settings or get_settings()
    if not settings.arabic_review_fallback_enabled:
        return None
    if not all((
        settings.arabic_review_fallback_base_url,
        settings.arabic_review_fallback_api_key,
        settings.arabic_review_fallback_model,
    )):
        return None
    if settings.arabic_review_fallback_provider != "openai_compatible":
        raise ValueError(
            "Unsupported ARABIC_REVIEW_FALLBACK_PROVIDER: "
            f"{settings.arabic_review_fallback_provider}."
        )
    return OpenAICompatibleLLMProvider(
        base_url=settings.arabic_review_fallback_base_url,
        api_key=settings.arabic_review_fallback_api_key,
        model_id=settings.arabic_review_fallback_model,
        timeout_seconds=settings.arabic_review_fallback_timeout_seconds,
        max_retries=settings.arabic_review_fallback_max_retries,
        retry_base_delay=settings.arabic_review_retry_base_delay,
        retry_max_delay=settings.arabic_review_retry_max_delay,
    )
