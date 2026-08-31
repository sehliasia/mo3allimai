"""Vendor-neutral, text-only LLM contract for the RAG orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable, Protocol


class LLMProviderError(RuntimeError):
    """Safe provider failure details suitable for local diagnostic tooling."""

    def __init__(self, message: str, *, status_code: int | None = None, provider_message: str | None = None, category: str | None = None, response_metadata: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_message = provider_message or message
        self.category = category
        self.response_metadata = response_metadata or {}


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    finish_reason: str | None = None
    output_token_count: int | None = None


@dataclass(frozen=True)
class LLMRetryPolicy:
    """Bounded retry override for an optional, non-primary LLM operation."""

    max_retries: int
    max_wait_seconds: float
    retry_base_delay: float
    retry_max_delay: float
    retry_jitter_seconds: float = 0.0
    diagnostic_callback: Callable[["LLMRetryDiagnostic"], None] | None = None


@dataclass(frozen=True)
class LLMRetryDiagnostic:
    """Safe retry metadata for an optional caller-owned diagnostic stream."""

    event: str
    attempt: int
    max_attempts: int
    status_code: int | None = None
    exception_class: str | None = None
    category: str | None = None
    retryable: bool | None = None
    retry_delay_seconds: float | None = None
    remaining_retry_wait_seconds: float | None = None
    provider_error_type: str | None = None
    choices_count: int | None = None
    content_present: bool | None = None
    content_type: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class LLMGenerationOptions:
    """Optional, provider-capability-gated settings for one generation call."""

    reasoning_effort: str | None = None
    include_reasoning: bool | None = None


class LLMProvider(Protocol):
    model_id: str

    def generate(
        self, *, system_prompt: str, user_prompt: str, temperature: float | None = None, max_tokens: int | None = None,
        retry_policy: LLMRetryPolicy | None = None, generation_options: LLMGenerationOptions | None = None,
    ) -> LLMResult: ...

    def stream_generate(
        self, *, system_prompt: str, user_prompt: str, temperature: float | None = None,
        max_tokens: int | None = None, generation_options: LLMGenerationOptions | None = None,
    ) -> AsyncIterator[str]: ...


class FakeLLMProvider:
    """Deterministic test double; it never accesses a network or local model."""

    model_id = "fake-llm"

    def __init__(self, response: str = "Grounded test response.") -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(self, *, system_prompt: str, user_prompt: str, temperature: float | None = None, max_tokens: int | None = None, retry_policy: LLMRetryPolicy | None = None, generation_options: LLMGenerationOptions | None = None) -> LLMResult:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "temperature": temperature, "max_tokens": max_tokens, "retry_policy": retry_policy, "generation_options": generation_options})
        return LLMResult(text=self.response, model=self.model_id)

    async def stream_generate(self, **_kwargs) -> AsyncIterator[str]:
        yield self.response
