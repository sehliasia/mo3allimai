import httpx
import pytest

from app.services.llm_provider import LLMGenerationOptions, LLMProviderError, LLMRetryPolicy
from app.services.llm_providers.openai_compatible_provider import OpenAICompatibleLLMProvider
from app.services.rag_service import RAGServiceError
from scripts.ask_knowledge import _diagnostic


def _provider(**kwargs):
    return OpenAICompatibleLLMProvider(base_url="https://gemini.example/v1beta/openai", api_key="AIza-secret-key", model_id="gemini-test", **kwargs)


def _groq_gpt_oss_provider(**kwargs):
    return OpenAICompatibleLLMProvider(base_url="https://api.groq.com/openai/v1", api_key="gsk-secret-key", model_id="openai/gpt-oss-20b", **kwargs)


def test_gemini_like_openai_response_parses_and_preserves_arabic_json(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(200, json={"choices": [{"message": {"content": "إجابة عربية"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 17}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    provider = _provider()
    query = "ما أهداف درس الدول المتأهلة لكأس العالم؟"
    result = provider.generate(system_prompt="Answer in Arabic.", user_prompt=f"QUESTION:\n{query}", max_tokens=1200)
    assert result.text == "إجابة عربية"
    assert query in captured["json"]["messages"][1]["content"]
    assert captured["json"]["messages"][1]["content"].encode("utf-8").decode("utf-8").endswith(query)
    assert captured["json"]["max_tokens"] == 1200
    assert result.finish_reason == "stop" and result.output_token_count == 17


def test_reviewer_reasoning_options_are_capability_gated_without_changing_primary_payload(monkeypatch):
    payloads = []
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: (payloads.append(kwargs["json"]), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}, request=httpx.Request("POST", url)))[1])

    provider = _groq_gpt_oss_provider()
    provider.generate(system_prompt="primary", user_prompt="request", max_tokens=1800, temperature=0.2)
    provider.generate(
        system_prompt="review", user_prompt="answer", max_tokens=3000, temperature=0,
        generation_options=LLMGenerationOptions(reasoning_effort="medium", include_reasoning=False),
    )

    assert payloads[0]["max_tokens"] == 1800 and "reasoning_effort" not in payloads[0] and "include_reasoning" not in payloads[0]
    assert payloads[1]["max_tokens"] == 3000 and payloads[1]["reasoning_effort"] == "medium" and payloads[1]["include_reasoning"] is False


def test_unsupported_openai_compatible_model_omits_reviewer_reasoning_options(monkeypatch):
    captured = {}
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: (captured.update(kwargs["json"]), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}, request=httpx.Request("POST", url)))[1])

    _provider().generate(
        system_prompt="review", user_prompt="answer", max_tokens=3000,
        generation_options=LLMGenerationOptions(reasoning_effort="medium", include_reasoning=False),
    )

    assert "reasoning_effort" not in captured and "include_reasoning" not in captured


def test_http_failure_keeps_safe_provider_message_without_api_key(monkeypatch):
    def post(url, **kwargs):
        response = httpx.Response(400, json={"error": {"message": "Unsupported language; key AIza-secret-key"}}, request=httpx.Request("POST", url))
        response.raise_for_status()

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMProviderError) as raised:
        _provider().generate(system_prompt="s", user_prompt="u")
    error = raised.value
    assert error.status_code == 400
    assert "Unsupported language" in error.provider_message
    assert "AIza-secret-key" not in error.provider_message
    assert isinstance(error.__cause__, httpx.HTTPStatusError)
    diagnostic = _diagnostic(RAGServiceError("Grounded answer generation failed.", status_code=error.status_code, provider_message=error.provider_message))
    assert "ERROR TYPE:" in diagnostic and "HTTP STATUS: 400" in diagnostic
    assert "AIza-secret-key" not in diagnostic


def test_empty_gemini_choice_reports_finish_reason_without_optional_metadata(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(200, json={"choices": [{"message": {"content": None}, "finish_reason": "SAFETY"}]}, request=httpx.Request("POST", url)))
    with pytest.raises(LLMProviderError) as raised:
        _provider().generate(system_prompt="s", user_prompt="u")
    assert "finish_reason=SAFETY" in raised.value.provider_message


@pytest.mark.parametrize(("payload", "category"), [
    ({}, "missing_choices"),
    ({"choices": [{"finish_reason": "stop"}]}, "missing_message"),
    ({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}, "missing_content"),
    ({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}, "empty_content"),
    ({"choices": [{"message": {"content": ["SECRET_CONTENT"]}, "finish_reason": "stop"}]}, "invalid_content_type"),
])
def test_http_200_malformed_completion_shapes_keep_safe_validation_categories(monkeypatch, payload, category):
    monkeypatch.setattr(httpx, "post", lambda url, **_kwargs: httpx.Response(200, json=payload, request=httpx.Request("POST", url)))

    with pytest.raises(LLMProviderError) as raised:
        _provider().generate(system_prompt="s", user_prompt="u")

    assert raised.value.status_code == 200 and raised.value.category == category
    assert "SECRET_CONTENT" not in raised.value.provider_message


def test_length_finish_reason_is_preserved_for_truncation_diagnostics(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(200, json={"choices": [{"message": {"content": "Réponse incomplète"}, "finish_reason": "length"}], "usage": {"completion_tokens": 1200}}, request=httpx.Request("POST", url)))
    result = _provider().generate(system_prompt="s", user_prompt="Question française", max_tokens=1200)
    assert result.finish_reason == "length" and result.output_token_count == 1200


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
def test_transient_http_status_retries_then_succeeds(monkeypatch, status):
    calls, sleeps = [], []
    responses = [httpx.Response(status, json={"error": {"message": "temporarily unavailable"}}, request=httpx.Request("POST", "https://gemini.example")), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", "https://gemini.example"))]

    def post(*_args, **_kwargs):
        calls.append(1)
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", post)
    assert _provider(max_retries=3, sleep=sleeps.append, random_source=lambda: 0.5).generate(system_prompt="s", user_prompt="u").text == "ok"
    assert len(calls) == 2 and sleeps == [1.0]


@pytest.mark.parametrize("status", [400, 401])
def test_429_retries_but_client_errors_do_not(monkeypatch, status):
    calls, sleeps = [], []
    responses = [httpx.Response(429, json={"error": {"message": "capacity"}}, request=httpx.Request("POST", "https://gemini.example")), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", "https://gemini.example"))]
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: (calls.append(1), responses.pop(0))[1])
    assert _provider(sleep=sleeps.append, random_source=lambda: 0.5).generate(system_prompt="s", user_prompt="u").text == "ok"
    assert len(calls) == 2 and sleeps

    calls.clear()
    def bad_request(url, **kwargs):
        calls.append(1)
        return httpx.Response(status, json={"error": {"message": "bad request"}}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", bad_request)
    with pytest.raises(LLMProviderError) as raised:
        _provider(sleep=sleeps.append).generate(system_prompt="s", user_prompt="u")
    assert raised.value.status_code == status and calls == [1]


def test_retry_limit_and_timeout_preserve_final_safe_error(monkeypatch):
    calls, sleeps = [], []
    def post(url, **kwargs):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMProviderError) as raised:
        _provider(max_retries=2, sleep=sleeps.append, random_source=lambda: 0.5).generate(system_prompt="s", user_prompt="u")
    assert len(calls) == 3 and sleeps == [1.0, 2.0]
    assert raised.value.status_code is None
    assert "AIza-secret-key" not in raised.value.provider_message
    assert isinstance(raised.value.__cause__, httpx.ReadTimeout)


def test_reviewer_retry_policy_prefers_retry_after_and_is_bounded(monkeypatch):
    calls, sleeps = [], []
    responses = [
        httpx.Response(429, headers={"Retry-After": "4"}, request=httpx.Request("POST", "https://gemini.example")),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", "https://gemini.example")),
    ]
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: (calls.append(1), responses.pop(0))[1])

    result = _provider(sleep=sleeps.append, random_source=lambda: 0.5).generate(
        system_prompt="s", user_prompt="u",
        retry_policy=LLMRetryPolicy(max_retries=4, max_wait_seconds=20, retry_base_delay=2, retry_max_delay=8),
    )

    assert result.text == "ok" and len(calls) == 2 and sleeps == [4.0]


def test_reviewer_retry_policy_continues_after_two_retry_after_responses(monkeypatch):
    calls, sleeps = [], []
    responses = [
        httpx.Response(429, headers={"Retry-After": "6"}, request=httpx.Request("POST", "https://gemini.example")),
        httpx.Response(429, headers={"Retry-After": "6"}, request=httpx.Request("POST", "https://gemini.example")),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", "https://gemini.example")),
    ]
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: (calls.append(1), responses.pop(0))[1])

    result = _provider(sleep=sleeps.append, random_source=lambda: 0.5).generate(
        system_prompt="s", user_prompt="u",
        retry_policy=LLMRetryPolicy(max_retries=4, max_wait_seconds=20, retry_base_delay=2, retry_max_delay=8),
    )

    assert result.text == "ok" and len(calls) == 3 and sleeps == [6.0, 6.0]


def test_reviewer_retry_policy_uses_exponential_fallback_and_preserves_primary_behavior(monkeypatch):
    calls, sleeps = [], []
    responses = [
        httpx.Response(429, headers={"Retry-After": "not-a-delay"}, request=httpx.Request("POST", "https://gemini.example")),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", "https://gemini.example")),
    ]
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: (calls.append(1), responses.pop(0))[1])
    _provider(sleep=sleeps.append, random_source=lambda: 0.5).generate(
        system_prompt="s", user_prompt="u",
        retry_policy=LLMRetryPolicy(max_retries=4, max_wait_seconds=20, retry_base_delay=2, retry_max_delay=8),
    )
    assert sleeps == [2.0]

    calls.clear(); sleeps.clear()
    responses[:] = [
        httpx.Response(429, headers={"Retry-After": "10"}, request=httpx.Request("POST", "https://gemini.example")),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", "https://gemini.example")),
    ]
    _provider(sleep=sleeps.append, random_source=lambda: 0.5).generate(system_prompt="s", user_prompt="u")
    assert sleeps == [1.0]


def test_reviewer_retry_policy_stops_after_its_wait_budget(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(httpx, "post", lambda url, **_kwargs: (calls.append(1), httpx.Response(429, request=httpx.Request("POST", url)))[1])
    with pytest.raises(LLMProviderError) as raised:
        _provider(sleep=sleeps.append, random_source=lambda: 0.5).generate(
            system_prompt="s", user_prompt="u",
            retry_policy=LLMRetryPolicy(max_retries=4, max_wait_seconds=3, retry_base_delay=2, retry_max_delay=8),
        )
    assert raised.value.status_code == 429
    assert sleeps == [2.0, 1.0] and len(calls) == 3


def test_typed_rate_limit_provider_error_is_retried_with_the_reviewer_policy(monkeypatch):
    calls, sleeps = [], []
    responses = [
        httpx.Response(429, request=httpx.Request("POST", "https://gemini.example")),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=httpx.Request("POST", "https://gemini.example")),
    ]
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: (calls.append(1), responses.pop(0))[1])
    assert _provider(sleep=sleeps.append, random_source=lambda: 0.5).generate(
        system_prompt="s", user_prompt="u",
        retry_policy=LLMRetryPolicy(max_retries=4, max_wait_seconds=20, retry_base_delay=2, retry_max_delay=8),
    ).text == "ok"
    assert len(calls) == 2 and sleeps == [2.0]
