import logging
import json
from pathlib import Path

import httpx

from app.services.arabic_linguistic_review_service import ArabicLinguisticReviewService
from app.services.llm_provider import LLMProviderError, LLMResult, LLMRetryPolicy
from app.services.llm_providers.openai_compatible_provider import OpenAICompatibleLLMProvider


class SequenceProvider:
    model_id = "review-fake"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResult(text=outcome, model=self.model_id)


def _review(provider, content="هذا أبي."):
    return ArabicLinguisticReviewService(llm=provider, max_tokens=1800).review(
        content=content,
        response_language="ar",
        cefr_level="A1",
        request_context="نشاط حول الأسرة",
    )


def test_arabic_content_triggers_one_minimal_review_with_original_content_and_cefr_constraint():
    provider = SequenceProvider(["هذا أبي."])

    result = _review(provider)

    assert result.applied and not result.fallback and result.content == "هذا أبي."
    assert len(provider.calls) == 1
    assert provider.calls[0]["user_prompt"].endswith("هذا أبي.")
    assert "Resolved CEFR level (preserve, do not reinterpret): A1" in provider.calls[0]["user_prompt"]


def test_non_arabic_content_skips_review_without_a_provider_call():
    provider = SequenceProvider(["unused"])
    result = _review(provider, content="A grounded French explanation.")
    assert not result.attempted and result.reason == "no_arabic_content" and provider.calls == []


def test_empty_short_provider_or_timeout_output_falls_back_to_original():
    original = "هذا مثال عربي صالح وطويل بما يكفي."
    for outcome in ("", "قصير", LLMProviderError("unavailable", status_code=503), TimeoutError("timeout")):
        result = _review(SequenceProvider([outcome]), content=original)
        assert result.fallback and not result.applied and result.content == original


def test_internal_markers_must_be_preserved_exactly_by_the_reviewer():
    original = "هذا أبي. [RESOURCE-1]"
    result = _review(SequenceProvider(["هذا أبي."]), content=original)
    assert result.fallback and result.reason == "internal_markers_changed" and result.content == original


def test_review_prompt_covers_linguistic_regressions_and_terminology_without_hardcoded_replacements():
    provider = SequenceProvider(["هذا أبي."])
    _review(provider)
    prompt = provider.calls[0]["system_prompt"]
    for expected in (
        "أبي طبيب", "جدتي", "ما صلة القرابة؟", "المتعلم and نصًا", "مدة تقريبية", "يُصغي",
        "apprenant(s)", "élève(s)", "étudiant(s)", "التلميذ/التلاميذ", "الطالب/الطلاب",
        "Preserve its Markdown structure", "Make the smallest possible edits", "complete final linguistic pass",
        "subject/verb, nominal, gender and singular/plural agreement", "participant roles", "teacher, learner, partner, group and class",
        "idiomatic Modern Standard Arabic", "resolved response language is Arabic", "unnecessary English or French glosses",
        "For A1, locally prefer short", "duration", "no explanation, change log, reviewer comments or reasoning",
        "global full-document terminology sweep", "every heading, table, instruction, criterion", "not only sentence by sentence",
        "possessive family vocabulary coherent", "including tables and lists", "lexical-semantic precision sweep",
        "skipped step numbering", "complete final checklist across the entire answer", "protected markers",
        "أنت مدقق لغوي وتربوي", "السلامة النحوية والصرفية", "ملاءمة المفردات والتراكيب للمستوى المحدد وفق CEFR",
        "عبارات التعبير عن الرأي", "العبارات المقترحة",
    ):
        assert expected in prompt


def test_rate_limit_exhaustion_falls_open_once_with_a_specific_reason():
    provider = SequenceProvider([LLMProviderError("rate limited", status_code=429)])

    result = _review(provider, content="هذا مثال عربي صالح وطويل بما يكفي.")

    assert result.fallback and not result.applied and result.reason == "rate_limit_exhausted"
    assert result.content == "هذا مثال عربي صالح وطويل بما يكفي." and len(provider.calls) == 1
    assert result.primary_status == 429 and not result.fallback_used


def test_non_rate_limit_failures_keep_a_distinct_safe_fallback_reason():
    provider = SequenceProvider([LLMProviderError("provider failure", status_code=500)])

    result = _review(provider, content="هذا مثال عربي صالح وطويل بما يكفي.")

    assert result.fallback and result.reason == "provider_failure" and len(provider.calls) == 1


def _reviewer_provider(*, sleep):
    return OpenAICompatibleLLMProvider(
        base_url="https://provider.example", api_key="secret-not-for-logs", model_id="test-model",
        sleep=sleep, random_source=lambda: 0.5,
    )


def test_reviewer_logs_safe_second_attempt_and_final_http_status(monkeypatch, caplog):
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}, request=httpx.Request("POST", "https://provider.example")),
        httpx.Response(400, json={"error": {"message": "invalid request secret-not-for-logs"}}, request=httpx.Request("POST", "https://provider.example")),
    ]
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: responses.pop(0))
    caplog.set_level(logging.INFO, logger="app.services.arabic_linguistic_review_service")
    monkeypatch.setattr(logging.getLogger("app.services.arabic_linguistic_review_service"), "propagate", True)
    reviewer = ArabicLinguisticReviewService(
        llm=_reviewer_provider(sleep=lambda _delay: None), max_tokens=1800,
        retry_policy=LLMRetryPolicy(max_retries=4, max_wait_seconds=20, retry_base_delay=2, retry_max_delay=8),
    )

    result = reviewer.review(content="هذا مثال عربي صالح وطويل بما يكفي.", response_language="ar", cefr_level="A1", request_context="اختبار")

    assert result.fallback and result.reason == "provider_failure"
    messages = [record.message for record in caplog.records]
    assert "arabic_review_provider_attempt provider=primary attempt=2/5" in messages
    assert any("arabic_review_provider_failure provider=primary status=400 exception=HTTPStatusError category=http_status_error retryable=False attempt=2/5" in message for message in messages)
    assert all("secret-not-for-logs" not in message for message in messages)


def test_reviewer_timeout_diagnostic_is_safe_and_has_a_timeout_reason(monkeypatch, caplog):
    attempts = [httpx.Response(429, headers={"Retry-After": "0"}, request=httpx.Request("POST", "https://provider.example"))]

    def post(url, **_kwargs):
        if not attempts:
            raise httpx.ReadTimeout("slow", request=httpx.Request("POST", url))
        return attempts.pop(0)

    monkeypatch.setattr(httpx, "post", post)
    caplog.set_level(logging.INFO, logger="app.services.arabic_linguistic_review_service")
    monkeypatch.setattr(logging.getLogger("app.services.arabic_linguistic_review_service"), "propagate", True)
    reviewer = ArabicLinguisticReviewService(
        llm=_reviewer_provider(sleep=lambda _delay: None), max_tokens=1800,
        retry_policy=LLMRetryPolicy(max_retries=4, max_wait_seconds=20, retry_base_delay=2, retry_max_delay=8),
    )

    result = reviewer.review(content="هذا مثال عربي صالح وطويل بما يكفي.", response_language="ar", cefr_level="A1", request_context="اختبار")

    assert result.fallback and result.reason == "timeout"
    assert any("arabic_review_provider_failure provider=primary status=None exception=ReadTimeout category=read_timeout retryable=True attempt=5/5" in record.message for record in caplog.records)


def test_reviewer_http_200_missing_content_logs_its_safe_response_stage(monkeypatch, caplog):
    monkeypatch.setattr(httpx, "post", lambda url, **_kwargs: httpx.Response(
        200,
        json={"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        request=httpx.Request("POST", url),
    ))
    caplog.set_level(logging.INFO, logger="app.services.arabic_linguistic_review_service")
    monkeypatch.setattr(logging.getLogger("app.services.arabic_linguistic_review_service"), "propagate", True)
    reviewer = ArabicLinguisticReviewService(
        llm=_reviewer_provider(sleep=lambda _delay: None), max_tokens=1800,
        retry_policy=LLMRetryPolicy(max_retries=4, max_wait_seconds=20, retry_base_delay=2, retry_max_delay=8),
    )

    result = reviewer.review(content="هذا مثال عربي صالح وطويل بما يكفي.", response_language="ar", cefr_level="A1", request_context="اختبار")

    assert result.fallback and result.reason == "missing_content"
    assert any("arabic_review_provider_response_failure provider=primary status=200 stage=missing_content choices_count=1 content_present=True content_type=NoneType finish_reason=stop exception=LLMProviderError" in record.message for record in caplog.records)


def test_truncated_reviewer_completion_keeps_the_original_answer():
    provider = SequenceProvider([])
    provider.generate = lambda **_kwargs: LLMResult(text="هذا مثال عربي صالح وطويل بما يكفي.", model="review-fake", finish_reason="length")

    result = _review(provider, content="هذا مثال عربي صالح وطويل بما يكفي.")

    assert result.fallback and result.reason == "truncated_review"


def test_primary_rate_limit_uses_configured_fallback_reviewer():
    original = "يجب أن يكون المتعلمين قادرين على التعبير عن آرائهم."
    primary = SequenceProvider([LLMProviderError("rate limited", status_code=429)])
    fallback = SequenceProvider(["يجب أن يكون المتعلمون قادرين على التعبير عن آرائهم."])

    result = ArabicLinguisticReviewService(llm=primary, fallback_llm=fallback, max_tokens=1800).review(
        content=original, response_language="ar", cefr_level="A1", request_context="نشاط",
    )

    assert result.applied and result.provider == "fallback" and result.fallback_used
    assert result.content == "يجب أن يكون المتعلمون قادرين على التعبير عن آرائهم."
    assert len(primary.calls) == len(fallback.calls) == 1


def test_all_reviewers_failed_preserves_the_generated_answer():
    original = "استخدم الجمل المعلّمة للتعبير عن رأيك."
    result = ArabicLinguisticReviewService(
        llm=SequenceProvider([LLMProviderError("rate limited", status_code=429)]),
        fallback_llm=SequenceProvider([LLMProviderError("unavailable", status_code=503)]), max_tokens=1800,
    ).review(content=original, response_language="ar", cefr_level="A1", request_context="نشاط")

    assert not result.applied and result.fallback and result.fallback_used
    assert result.reason == "all_review_providers_failed" and result.content == original


def test_linguistic_regression_fixture_is_explicit_and_deterministic():
    fixture = Path(__file__).with_name("fixtures") / "arabic_linguistic_review_regressions.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))

    assert len(cases) == 4
    assert cases[0]["expected"] == "يجب أن يكون المتعلمون قادرين على التعبير عن آرائهم."
    assert cases[1]["expected"] == "هل يتبع المتحدث تسلسلًا منطقيًا؟"
    assert cases[2]["expected"] == "استخدم العبارات المقترحة للتعبير عن رأيك."
    assert cases[3]["expected"] == "عبارات التعبير عن الرأي"
