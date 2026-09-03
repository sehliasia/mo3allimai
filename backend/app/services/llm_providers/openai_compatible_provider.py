"""Minimal synchronous OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

import httpx
import logging
import random
import re
import time
import json
from collections.abc import AsyncIterator
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from urllib.parse import urlsplit

from app.services.llm_provider import LLMGenerationOptions, LLMProviderError, LLMResult, LLMRetryDiagnostic, LLMRetryPolicy

logger = logging.getLogger(__name__)
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _exception_category(exc: Exception) -> str:
    """Return a transport-only category without including request data."""
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "write_timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "pool_timeout"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connect_error"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "remote_protocol_error"
    if isinstance(exc, httpx.ProtocolError):
        return "protocol_error"
    if isinstance(exc, httpx.NetworkError):
        return "network_error"
    return "http_status_error" if isinstance(exc, httpx.HTTPStatusError) else "transport_error"


class OpenAICompatibleLLMProvider:
    def __init__(
        self, *, base_url: str, api_key: str, model_id: str, timeout_seconds: float = 60.0,
        max_retries: int = 3, retry_base_delay: float = 1.0, retry_max_delay: float = 8.0,
        sleep=time.sleep, random_source=random.random,
    ) -> None:
        if not base_url or not api_key or not model_id:
            raise ValueError("RAG LLM base URL, API key, and model must be configured.")
        if max_retries < 0 or retry_base_delay < 0 or retry_max_delay < 0:
            raise ValueError("LLM retry settings must not be negative.")
        self.base_url, self.api_key, self.model_id, self.timeout_seconds = base_url.rstrip("/"), api_key, model_id, timeout_seconds
        self.max_retries, self.retry_base_delay, self.retry_max_delay = max_retries, retry_base_delay, retry_max_delay
        self._sleep, self._random = sleep, random_source

    def generate(
        self, *, system_prompt: str, user_prompt: str, temperature: float | None = None, max_tokens: int | None = None,
        retry_policy: LLMRetryPolicy | None = None, generation_options: LLMGenerationOptions | None = None,
    ) -> LLMResult:
        max_retries = retry_policy.max_retries if retry_policy else self.max_retries
        retry_base_delay = retry_policy.retry_base_delay if retry_policy else self.retry_base_delay
        retry_max_delay = retry_policy.retry_max_delay if retry_policy else self.retry_max_delay
        if max_retries < 0 or retry_base_delay < 0 or retry_max_delay < 0 or (retry_policy and (retry_policy.max_wait_seconds < 0 or retry_policy.retry_jitter_seconds < 0)):
            raise ValueError("LLM retry policy values must not be negative.")
        final_error: LLMProviderError | None = None
        final_cause: Exception | None = None
        retry_waited_seconds = 0.0
        for attempt in range(max_retries + 1):
            response: httpx.Response | None = None
            self._emit_retry_diagnostic(
                retry_policy,
                event="attempt",
                attempt=attempt + 1,
                max_attempts=max_retries + 1,
            )
            try:
                payload = self._completion_payload(
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    temperature=temperature, max_tokens=max_tokens,
                )
                if generation_options and self._supports_reasoning_options():
                    if generation_options.reasoning_effort is not None:
                        payload["reasoning_effort"] = generation_options.reasoning_effort
                    if generation_options.include_reasoning is not None:
                        payload["include_reasoning"] = generation_options.include_reasoning
                response = httpx.post(
                    f"{self.base_url}/chat/completions", headers=self._headers(),
                    json=payload, timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload, choice, content, finish_reason = self._parse_completion_response(response)
                usage = payload.get("usage") if isinstance(payload, dict) else None
                output_tokens = None
                if isinstance(usage, dict):
                    candidate_count = usage.get("completion_tokens", usage.get("output_tokens"))
                    if isinstance(candidate_count, int) and not isinstance(candidate_count, bool):
                        output_tokens = candidate_count
                return LLMResult(
                    text=content.strip(), model=self.model_id,
                    finish_reason=finish_reason if isinstance(finish_reason, str) else None,
                    output_token_count=output_tokens,
                )
            except LLMProviderError as exc:
                final_error = exc
                final_cause = exc.__cause__ if isinstance(exc.__cause__, Exception) else exc
                retryable = exc.status_code in _TRANSIENT_STATUS_CODES
                if not retryable or attempt >= max_retries:
                    self._emit_retry_diagnostic(retry_policy, event="failure", attempt=attempt + 1, max_attempts=max_retries + 1, status_code=exc.status_code, exception=exc, category=exc.category or "provider_error", retryable=retryable, response_metadata=exc.response_metadata)
                    break
                delay = self._retry_delay(
                    response=None,
                    status_code=exc.status_code,
                    attempt=attempt,
                    retry_base_delay=retry_base_delay,
                    retry_max_delay=retry_max_delay,
                    retry_policy=retry_policy,
                )
                if retry_policy:
                    remaining_wait = retry_policy.max_wait_seconds - retry_waited_seconds
                    if remaining_wait <= 0:
                        self._emit_retry_diagnostic(retry_policy, event="failure", attempt=attempt + 1, max_attempts=max_retries + 1, status_code=exc.status_code, exception=exc, category=exc.category or "provider_error", retryable=retryable, response_metadata=exc.response_metadata)
                        break
                    delay = min(delay, remaining_wait)
                self._log_retry(
                    status_code=exc.status_code,
                    exception=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                    delay=delay,
                    remaining_wait=(retry_policy.max_wait_seconds - retry_waited_seconds - delay) if retry_policy else None,
                )
                self._sleep(delay)
                retry_waited_seconds += delay
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                # A non-2xx httpx response is falsey; preserve it explicitly so
                # its status and Retry-After header remain available to retries.
                error_response = response if response is not None else getattr(exc, "response", None)
                status_code = getattr(error_response, "status_code", None)
                final_error = LLMProviderError("The configured LLM provider did not return a valid response.", status_code=status_code, provider_message=self._safe_provider_message(error_response), category=_exception_category(exc), response_metadata={"retry_after_seconds": self._retry_after_seconds(error_response)})
                final_cause = exc
                retryable = status_code in _TRANSIENT_STATUS_CODES or isinstance(exc, httpx.TransportError)
                if not retryable or attempt >= max_retries:
                    self._emit_retry_diagnostic(retry_policy, event="failure", attempt=attempt + 1, max_attempts=max_retries + 1, status_code=status_code, exception=exc, category=_exception_category(exc), retryable=retryable)
                    break
                delay = self._retry_delay(
                    response=error_response,
                    status_code=status_code,
                    attempt=attempt,
                    retry_base_delay=retry_base_delay,
                    retry_max_delay=retry_max_delay,
                    retry_policy=retry_policy,
                )
                if retry_policy:
                    remaining_wait = retry_policy.max_wait_seconds - retry_waited_seconds
                    if remaining_wait <= 0:
                        self._emit_retry_diagnostic(retry_policy, event="failure", attempt=attempt + 1, max_attempts=max_retries + 1, status_code=status_code, exception=exc, category=_exception_category(exc), retryable=retryable)
                        break
                    delay = min(delay, remaining_wait)
                self._log_retry(
                    status_code=status_code,
                    exception=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                    delay=delay,
                    remaining_wait=(retry_policy.max_wait_seconds - retry_waited_seconds - delay) if retry_policy else None,
                )
                self._sleep(delay)
                retry_waited_seconds += delay
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                final_error = LLMProviderError("The configured LLM provider did not return a valid response.", status_code=getattr(response, "status_code", None), provider_message=self._safe_provider_message(response), category="malformed_response")
                final_cause = exc
                self._emit_retry_diagnostic(retry_policy, event="failure", attempt=attempt + 1, max_attempts=max_retries + 1, status_code=getattr(response, "status_code", None), exception=exc, category="malformed_response", retryable=False)
                break
        assert final_error is not None and final_cause is not None
        raise final_error from final_cause

    async def stream_generate(
        self, *, system_prompt: str, user_prompt: str, temperature: float | None = None,
        max_tokens: int | None = None, generation_options: LLMGenerationOptions | None = None,
    ) -> AsyncIterator[str]:
        """Yield real OpenAI-compatible deltas; exiting the iterator closes upstream."""
        payload = self._completion_payload(
            system_prompt=system_prompt, user_prompt=user_prompt,
            temperature=temperature, max_tokens=max_tokens, stream=True,
        )
        if generation_options and self._supports_reasoning_options():
            if generation_options.reasoning_effort is not None:
                payload["reasoning_effort"] = generation_options.reasoning_effort
            if generation_options.include_reasoning is not None:
                payload["include_reasoning"] = generation_options.include_reasoning
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions",
                    headers=self._headers(accept="text/event-stream"), json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            return
                        try:
                            event = json.loads(raw)
                            choices = event.get("choices", [])
                            delta = choices[0].get("delta", {}) if choices else {}
                            content = delta.get("content") if isinstance(delta, dict) else None
                            reasoning = delta.get("reasoning") if isinstance(delta, dict) else None
                            finish_reason = choices[0].get("finish_reason") if choices else None
                        except (ValueError, TypeError, IndexError) as exc:
                            raise LLMProviderError("The configured LLM provider returned an invalid stream.", category="malformed_stream") from exc
                        logger.info(
                            "assistant_llm_stream_chunk model=%s content_chars=%s reasoning_chars=%s finish_reason=%s",
                            self.model_id, len(content) if isinstance(content, str) else 0,
                            len(reasoning) if isinstance(reasoning, str) else 0, finish_reason,
                        )
                        if isinstance(content, str) and content:
                            yield content
        except LLMProviderError:
            raise
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            response = getattr(exc, "response", None)
            raise LLMProviderError(
                "The configured LLM provider stream failed.", status_code=getattr(response, "status_code", None),
                provider_message=self._safe_provider_message(response), category=_exception_category(exc),
            ) from exc

    def _completion_payload(
        self, *, system_prompt: str, user_prompt: str, temperature: float | None,
        max_tokens: int | None, stream: bool = False,
    ) -> dict[str, object]:
        """Keep model, prompts and generation limits identical across HTTP modes."""
        payload: dict[str, object] = {
            "model": self.model_id,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            payload["stream"] = True
        return payload

    def _headers(self, *, accept: str | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if accept is not None:
            headers["Accept"] = accept
        return headers

    @staticmethod
    def _log_retry(
        *, status_code: int | None, exception: Exception, attempt: int, max_retries: int,
        delay: float, remaining_wait: float | None,
    ) -> None:
        remaining_suffix = "" if remaining_wait is None else f" remaining_retry_wait={max(0.0, remaining_wait):.2f}s"
        logger.warning(
            "LLM transient failure status=%s exception=%s category=%s attempt=%s/%s retry_in=%.2fs%s",
            status_code, type(exception).__name__, _exception_category(exception),
            attempt + 1, max_retries + 1, delay, remaining_suffix,
        )

    @staticmethod
    def _emit_retry_diagnostic(
        retry_policy: LLMRetryPolicy | None,
        *, event: str, attempt: int, max_attempts: int,
        status_code: int | None = None, exception: Exception | None = None,
        category: str | None = None, retryable: bool | None = None, response_metadata: dict[str, object] | None = None,
    ) -> None:
        if retry_policy is None or retry_policy.diagnostic_callback is None:
            return
        try:
            retry_policy.diagnostic_callback(LLMRetryDiagnostic(
                event=event,
                attempt=attempt,
                max_attempts=max_attempts,
                status_code=status_code,
                exception_class=type(exception).__name__ if exception else None,
                category=category,
                retryable=retryable,
                provider_error_type=category,
                choices_count=response_metadata.get("choices_count") if response_metadata else None,
                content_present=response_metadata.get("content_present") if response_metadata else None,
                content_type=response_metadata.get("content_type") if response_metadata else None,
                finish_reason=response_metadata.get("finish_reason") if response_metadata else None,
            ))
        except Exception:
            # Observability must never alter an LLM request's retry semantics.
            return

    @staticmethod
    def _parse_completion_response(response: httpx.Response) -> tuple[dict[str, object], dict[str, object], str, str | None]:
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            raise LLMProviderError(
                "The configured LLM provider did not return valid JSON.", status_code=response.status_code,
                provider_message="Provider returned invalid JSON.", category="json_decode_failure",
            ) from exc
        if not isinstance(payload, dict):
            raise LLMProviderError(
                "The configured LLM provider did not return a valid response.", status_code=response.status_code,
                provider_message="Provider response has an invalid top-level shape.", category="malformed_response",
            )
        choices = payload.get("choices")
        choices_count = len(choices) if isinstance(choices, list) else 0
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMProviderError(
                "The configured LLM provider did not return a valid response.", status_code=response.status_code,
                provider_message="Provider response is missing a usable choice.", category="missing_choices",
                response_metadata={"choices_count": choices_count},
            )
        choice = choices[0]
        message = choice.get("message")
        finish_reason = choice.get("finish_reason")
        safe_finish_reason = finish_reason if isinstance(finish_reason, str) else None
        if not isinstance(message, dict):
            raise LLMProviderError(
                "The configured LLM provider did not return a valid response.", status_code=response.status_code,
                provider_message="Provider response is missing a usable message.", category="missing_message",
                response_metadata={"choices_count": choices_count, "finish_reason": safe_finish_reason},
            )
        content_present = "content" in message
        content = message.get("content")
        metadata = {
            "choices_count": choices_count,
            "content_present": content_present,
            "content_type": type(content).__name__ if content_present else None,
            "finish_reason": safe_finish_reason,
        }
        if not content_present or content is None:
            detail = "Provider response has no message content."
            if safe_finish_reason:
                detail += f" (finish_reason={safe_finish_reason})"
            raise LLMProviderError(
                "The configured LLM provider returned no textual completion.", status_code=response.status_code,
                provider_message=detail, category="missing_content",
                response_metadata=metadata,
            )
        if not isinstance(content, str):
            raise LLMProviderError(
                "The configured LLM provider returned an invalid content type.", status_code=response.status_code,
                provider_message="Provider message content is not text.", category="invalid_content_type",
                response_metadata=metadata,
            )
        if not content.strip():
            raise LLMProviderError(
                "The configured LLM provider returned no textual completion.", status_code=response.status_code,
                provider_message="Provider message content is empty.", category="empty_content",
                response_metadata=metadata,
            )
        return payload, choice, content.strip(), safe_finish_reason

    def _retry_delay(
        self, *, response: httpx.Response | None, status_code: int | None, attempt: int,
        retry_base_delay: float, retry_max_delay: float, retry_policy: LLMRetryPolicy | None,
    ) -> float:
        jitter = retry_policy.retry_jitter_seconds * self._random() if retry_policy else 0.0
        if retry_policy and status_code == 429:
            retry_after = self._retry_after_seconds(response)
            if retry_after is not None:
                return retry_after + jitter
        delay = min(retry_max_delay, retry_base_delay * (2 ** attempt))
        return delay * (0.8 + (0.4 * self._random())) + jitter

    def _supports_reasoning_options(self) -> bool:
        hostname = urlsplit(self.base_url).hostname or ""
        return hostname.casefold() == "api.groq.com" and self.model_id.casefold().startswith("openai/gpt-oss-")

    @staticmethod
    def _retry_after_seconds(response: httpx.Response | None) -> float | None:
        if response is None:
            return None
        retry_after = response.headers.get("Retry-After")
        if not retry_after:
            return None
        try:
            seconds = float(retry_after)
            return seconds if seconds >= 0 else None
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, IndexError, OverflowError):
                return None

    @staticmethod
    def _safe_provider_message(response: httpx.Response | None) -> str:
        if response is None:
            return "Provider request failed before a response was received."
        try:
            payload = response.json()
            error = payload.get("error", payload) if isinstance(payload, dict) else {}
            message = error.get("message") if isinstance(error, dict) else None
            if isinstance(message, str) and message.strip():
                return OpenAICompatibleLLMProvider._redact(message)
        except (ValueError, TypeError):
            pass
        return f"Provider returned HTTP {response.status_code}."

    @staticmethod
    def _redact(message: str) -> str:
        message = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", message)
        message = re.sub(r"AIza[\w-]+", "[REDACTED]", message)
        return message[:500]
