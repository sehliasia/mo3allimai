"""Minimal, provider-only connectivity diagnostic for the configured LLM."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

# Keep this diagnostic directly runnable from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.llm_provider import LLMProviderError
from app.services.llm_providers import get_llm_provider
from app.services.llm_providers.openai_compatible_provider import OpenAICompatibleLLMProvider


def _safe_base_url(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _raw_httpx_smoke(*, base_url: str, api_key: str, model: str, timeout_seconds: float) -> int:
    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Réponds uniquement par OK."}],
                "max_tokens": 32,
            },
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        print("direct_httpx_result=error")
        print("http_status=None")
        print(f"exception_class={type(exc).__name__}")
        print(f"elapsed_ms={round((time.perf_counter() - started) * 1000)}")
        return 1

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    if not response.is_success:
        print("direct_httpx_result=error")
        print(f"http_status={response.status_code}")
        print(f"provider_message={OpenAICompatibleLLMProvider._safe_provider_message(response)}")
        print(f"elapsed_ms={elapsed_ms}")
        return 1

    payload = response.json()
    choice = payload["choices"][0]
    print("direct_httpx_result=success")
    print(f"http_status={response.status_code}")
    print(f"finish_reason={choice.get('finish_reason')}")
    print(f"elapsed_ms={elapsed_ms}")
    return 0


def _models_only(*, base_url: str, api_key: str, model: str, timeout_seconds: float) -> int:
    """Verify authenticated provider access without requesting inference."""
    started = time.perf_counter()
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        print("models_result=error")
        print("http_status=None")
        print(f"exception_class={type(exc).__name__}")
        print("safe_error_message=Provider request failed before a response was received.")
        print(f"elapsed_ms={round((time.perf_counter() - started) * 1000)}")
        return 1

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    if not response.is_success:
        print("models_result=error")
        print(f"http_status={response.status_code}")
        print("exception_class=None")
        print(f"safe_error_message={OpenAICompatibleLLMProvider._safe_provider_message(response)}")
        print(f"elapsed_ms={elapsed_ms}")
        return 1

    try:
        payload = response.json()
        models = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids = {
            item.get("id") for item in models
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    except (TypeError, ValueError):
        print("models_result=error")
        print(f"http_status={response.status_code}")
        print("exception_class=ValueError")
        print("safe_error_message=Provider returned an invalid models response.")
        print(f"elapsed_ms={elapsed_ms}")
        return 1

    print("models_result=success")
    print(f"http_status={response.status_code}")
    print("exception_class=None")
    print("safe_error_message=None")
    print(f"configured_model_available={model in model_ids}")
    print(f"models_count={len(model_ids)}")
    print(f"elapsed_ms={elapsed_ms}")
    return 0


def _native_models_only(*, base_url: str, api_key: str, model: str, timeout_seconds: float) -> int:
    """Check native Gemini API-key authentication without inference."""
    parsed = urlsplit(base_url)
    native_models_url = urlunsplit((parsed.scheme, parsed.netloc, "/v1beta/models", "", ""))
    started = time.perf_counter()
    try:
        response = httpx.get(
            native_models_url,
            headers={"x-goog-api-key": api_key},
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        print("native_models_result=error")
        print("http_status=None")
        print(f"exception_class={type(exc).__name__}")
        print("safe_error_message=Provider request failed before a response was received.")
        print(f"elapsed_ms={round((time.perf_counter() - started) * 1000)}")
        return 1

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    if not response.is_success:
        print("native_models_result=error")
        print(f"http_status={response.status_code}")
        print("exception_class=None")
        print(f"safe_error_message={OpenAICompatibleLLMProvider._safe_provider_message(response)}")
        print("configured_model_available=unknown")
        print(f"elapsed_ms={elapsed_ms}")
        return 1

    try:
        payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        model_ids = {
            item.get("name", "").removeprefix("models/") for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
    except (TypeError, ValueError):
        print("native_models_result=error")
        print(f"http_status={response.status_code}")
        print("exception_class=ValueError")
        print("safe_error_message=Provider returned an invalid models response.")
        print("configured_model_available=unknown")
        print(f"elapsed_ms={elapsed_ms}")
        return 1

    print("native_models_result=success")
    print(f"http_status={response.status_code}")
    print("exception_class=None")
    print("safe_error_message=None")
    print(f"configured_model_available={model in model_ids}")
    print(f"models_count={len(model_ids)}")
    print(f"elapsed_ms={elapsed_ms}")
    return 0


def _list_models(*, base_url: str, api_key: str, model: str, timeout_seconds: float) -> int:
    """List the configured account's relevant OpenAI-compatible model IDs only."""
    started = time.perf_counter()
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        print("models_result=error")
        print("http_status=None")
        print(f"exception_class={type(exc).__name__}")
        print("safe_error_message=Provider request failed before a response was received.")
        print(f"elapsed_ms={round((time.perf_counter() - started) * 1000)}")
        return 1

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    if not response.is_success:
        print("models_result=error")
        print(f"http_status={response.status_code}")
        print(f"safe_error_message={OpenAICompatibleLLMProvider._safe_provider_message(response)}")
        print(f"elapsed_ms={elapsed_ms}")
        return 1

    try:
        payload = response.json()
        raw_ids = [
            item.get("id") for item in payload.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
    except (TypeError, ValueError):
        print("models_result=error")
        print(f"http_status={response.status_code}")
        print("safe_error_message=Provider returned an invalid models response.")
        print(f"elapsed_ms={elapsed_ms}")
        return 1

    normalized_ids = {model_id.removeprefix("models/") for model_id in raw_ids}
    relevant_ids = [
        model_id for model_id in raw_ids
        if any(marker in model_id.casefold() for marker in ("gemini", "flash", "pro"))
    ][:30]
    print("models_result=success")
    print(f"http_status={response.status_code}")
    print(f"models_count={len(raw_ids)}")
    print(f"configured_model={model}")
    print(f"exact_match={model in raw_ids}")
    print(f"normalized_match={model in normalized_ids}")
    print(f"configured_model_available={model in normalized_ids}")
    print("available_model_ids:")
    for model_id in relevant_ids:
        print(f"- {model_id}")
    print(f"elapsed_ms={elapsed_ms}")
    return 0


def _raw_chat_low(*, base_url: str, api_key: str, model: str, timeout_seconds: float) -> int:
    """Run one minimal low-reasoning chat request without provider retries."""
    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply only with OK."}],
                "reasoning_effort": "low",
                "max_tokens": 32,
            },
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        print("raw_chat_low_result=error")
        print("http_status=None")
        print(f"exception_class={type(exc).__name__}")
        print(f"elapsed_ms={round((time.perf_counter() - started) * 1000)}")
        print("finish_reason=None")
        print("safe_error_message=Provider request failed before a response was received.")
        return 1

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    if not response.is_success:
        print("raw_chat_low_result=error")
        print(f"http_status={response.status_code}")
        print("exception_class=None")
        print(f"elapsed_ms={elapsed_ms}")
        print("finish_reason=None")
        print(f"safe_error_message={OpenAICompatibleLLMProvider._safe_provider_message(response)}")
        return 1

    try:
        payload = response.json()
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        print("raw_chat_low_result=error")
        print(f"http_status={response.status_code}")
        print("exception_class=ValueError")
        print(f"elapsed_ms={elapsed_ms}")
        print("finish_reason=None")
        print("safe_error_message=Provider returned an invalid chat response.")
        return 1

    print("raw_chat_low_result=success")
    print(f"http_status={response.status_code}")
    print("exception_class=None")
    print(f"elapsed_ms={elapsed_ms}")
    print(f"finish_reason={choice.get('finish_reason')}")
    print(f"response_text={content[:100] if isinstance(content, str) else ''}")
    print("safe_error_message=None")
    return 0


def _raw_chat_ipv4(*, base_url: str, api_key: str, model: str, timeout_seconds: float) -> int:
    """Run one minimal IPv4-only chat request without provider retries."""
    started = time.perf_counter()
    transport = httpx.HTTPTransport(local_address="0.0.0.0")
    try:
        with httpx.Client(transport=transport, timeout=timeout_seconds) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply only with OK."}],
                    "max_tokens": 32,
                },
            )
    except httpx.HTTPError as exc:
        print("raw_chat_ipv4_result=error")
        print("http_status=None")
        print(f"exception_class={type(exc).__name__}")
        print(f"elapsed_ms={round((time.perf_counter() - started) * 1000)}")
        print("finish_reason=None")
        print("safe_error_message=Provider request failed before a response was received.")
        return 1

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    if not response.is_success:
        print("raw_chat_ipv4_result=error")
        print(f"http_status={response.status_code}")
        print("exception_class=None")
        print(f"elapsed_ms={elapsed_ms}")
        print("finish_reason=None")
        print(f"safe_error_message={OpenAICompatibleLLMProvider._safe_provider_message(response)}")
        return 1

    try:
        payload = response.json()
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        print("raw_chat_ipv4_result=error")
        print(f"http_status={response.status_code}")
        print("exception_class=ValueError")
        print(f"elapsed_ms={elapsed_ms}")
        print("finish_reason=None")
        print("safe_error_message=Provider returned an invalid chat response.")
        return 1

    print("raw_chat_ipv4_result=success")
    print(f"http_status={response.status_code}")
    print("exception_class=None")
    print(f"elapsed_ms={elapsed_ms}")
    print(f"finish_reason={choice.get('finish_reason')}")
    print(f"response_text={content[:100] if isinstance(content, str) else ''}")
    print("safe_error_message=None")
    return 0


def _compare_chat_auth(*, base_url: str, api_key: str, model: str) -> int:
    """Compare fake and configured Bearer keys with one IPv4-only client."""
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply only with OK."}],
        "max_tokens": 32,
    }
    transport = httpx.HTTPTransport(local_address="0.0.0.0")

    def send(client: httpx.Client, key: str, prefix: str) -> tuple[bool, httpx.Response | None]:
        started = time.perf_counter()
        try:
            response = client.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            print(f"{prefix}_result=error")
            print(f"{prefix}_http_status=None")
            print(f"{prefix}_exception_class={type(exc).__name__}")
            print(f"{prefix}_elapsed_ms={round((time.perf_counter() - started) * 1000)}")
            print(f"{prefix}_safe_error=Provider request failed before a response was received.")
            return False, None

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        if not response.is_success:
            print(f"{prefix}_result=error")
            print(f"{prefix}_http_status={response.status_code}")
            print(f"{prefix}_exception_class=None")
            print(f"{prefix}_elapsed_ms={elapsed_ms}")
            print(f"{prefix}_safe_error={OpenAICompatibleLLMProvider._safe_provider_message(response)}")
            return False, response

        print(f"{prefix}_result=success")
        print(f"{prefix}_http_status={response.status_code}")
        print(f"{prefix}_exception_class=None")
        print(f"{prefix}_elapsed_ms={elapsed_ms}")
        print(f"{prefix}_safe_error=None")
        return True, response

    with httpx.Client(transport=transport, timeout=20.0) as client:
        send(client, "FAKE_TEST_KEY_123", "fake")
        real_success, real_response = send(client, api_key, "real")

    if not real_success or real_response is None:
        print("real_finish_reason=None")
        return 1
    try:
        choice = real_response.json()["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        print("real_finish_reason=None")
        print("real_response_text=")
        return 1
    print(f"real_finish_reason={choice.get('finish_reason')}")
    print(f"real_response_text={content[:100] if isinstance(content, str) else ''}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-httpx-only", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--native-models-only", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--raw-chat-low", action="store_true")
    parser.add_argument("--raw-chat-ipv4", action="store_true")
    parser.add_argument("--compare-chat-auth", action="store_true")
    arguments = parser.parse_args()
    selected_modes = sum((
        arguments.raw_httpx_only, arguments.models_only, arguments.native_models_only,
        arguments.list_models, arguments.raw_chat_low, arguments.raw_chat_ipv4, arguments.compare_chat_auth,
    ))
    if selected_modes > 1:
        parser.error("Diagnostic modes cannot be used together")
    settings = get_settings()
    print(f"provider_type={settings.rag_llm_provider}")
    print(f"provider_model={settings.rag_llm_model}")
    print(f"base_url={_safe_base_url(settings.rag_llm_base_url)}")
    print(f"api_key_present={bool(settings.rag_llm_api_key)}")
    print(f"timeout_seconds={settings.rag_llm_timeout_seconds}")
    print(f"max_retries={settings.rag_llm_max_retries}")
    print(f"httpx_version={httpx.__version__}")

    if arguments.raw_httpx_only:
        return _raw_httpx_smoke(
            base_url=settings.rag_llm_base_url,
            api_key=settings.rag_llm_api_key or "",
            model=settings.rag_llm_model,
            timeout_seconds=settings.rag_llm_timeout_seconds,
        )
    if arguments.models_only:
        return _models_only(
            base_url=settings.rag_llm_base_url,
            api_key=settings.rag_llm_api_key or "",
            model=settings.rag_llm_model,
            timeout_seconds=settings.rag_llm_timeout_seconds,
        )
    if arguments.native_models_only:
        return _native_models_only(
            base_url=settings.rag_llm_base_url,
            api_key=settings.rag_llm_api_key or "",
            model=settings.rag_llm_model,
            timeout_seconds=settings.rag_llm_timeout_seconds,
        )
    if arguments.list_models:
        return _list_models(
            base_url=settings.rag_llm_base_url,
            api_key=settings.rag_llm_api_key or "",
            model=settings.rag_llm_model,
            timeout_seconds=settings.rag_llm_timeout_seconds,
        )
    if arguments.raw_chat_low:
        return _raw_chat_low(
            base_url=settings.rag_llm_base_url,
            api_key=settings.rag_llm_api_key or "",
            model=settings.rag_llm_model,
            timeout_seconds=settings.rag_llm_timeout_seconds,
        )
    if arguments.raw_chat_ipv4:
        return _raw_chat_ipv4(
            base_url=settings.rag_llm_base_url,
            api_key=settings.rag_llm_api_key or "",
            model=settings.rag_llm_model,
            timeout_seconds=settings.rag_llm_timeout_seconds,
        )
    if arguments.compare_chat_auth:
        return _compare_chat_auth(
            base_url=settings.rag_llm_base_url,
            api_key=settings.rag_llm_api_key or "",
            model=settings.rag_llm_model,
        )

    started = time.perf_counter()
    try:
        result = get_llm_provider(settings).generate(
            system_prompt="You are a connectivity smoke test.",
            user_prompt="Réponds uniquement par OK.",
            temperature=0,
            max_tokens=32,
        )
    except LLMProviderError as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        cause = exc.__cause__
        print("result=error")
        print(f"http_status={exc.status_code}")
        print(f"provider_message={exc.provider_message[:500]}")
        print(f"exception_class={type(cause).__name__ if cause else type(exc).__name__}")
        print(f"elapsed_ms={elapsed_ms}")
        return 1
    except Exception as exc:  # Configuration errors are still reported without request internals.
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        print("result=error")
        print("http_status=None")
        print(f"exception_class={type(exc).__name__}")
        print(f"elapsed_ms={elapsed_ms}")
        return 1

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    print("result=success")
    print("http_status=200")
    print(f"finish_reason={result.finish_reason}")
    print(f"response={result.text[:80]}")
    print(f"elapsed_ms={elapsed_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
