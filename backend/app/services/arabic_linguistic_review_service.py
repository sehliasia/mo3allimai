"""Narrow, fail-open Arabic linguistic review for completed assistant answers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import re
from time import perf_counter

from app.services.llm_provider import LLMGenerationOptions, LLMProvider, LLMProviderError, LLMRetryDiagnostic, LLMRetryPolicy


logger = logging.getLogger(__name__)
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_INTERNAL_MARKER = re.compile(r"\[(?:RESOURCE|CEFR(?:-MISSING)?)-\d+\]")


@dataclass(frozen=True)
class ArabicLinguisticReviewResult:
    content: str
    attempted: bool
    applied: bool
    fallback: bool
    reason: str
    elapsed_ms: int
    provider: str | None = None
    attempt_count: int = 0
    primary_status: int | str | None = None
    fallback_used: bool = False


class ArabicLinguisticReviewService:
    """Proofread Arabic locally without changing pedagogical design or provenance."""

    def __init__(self, *, llm: LLMProvider, max_tokens: int, fallback_llm: LLMProvider | None = None, retry_policy: LLMRetryPolicy | None = None, generation_options: LLMGenerationOptions | None = None) -> None:
        self.llm = llm
        self.fallback_llm = fallback_llm
        self.max_tokens = max_tokens
        self.retry_policy = retry_policy
        self.generation_options = generation_options

    @staticmethod
    def needs_review(content: str) -> bool:
        return bool(_ARABIC_SCRIPT.search(content))

    @staticmethod
    def _markers(content: str) -> list[str]:
        return _INTERNAL_MARKER.findall(content)

    @staticmethod
    def _review_prompt(*, response_language: str, cefr_level: str | None, request_context: str) -> tuple[str, str]:
        system = (
            "أنت مدقق لغوي وتربوي متخصص في اللغة العربية الفصحى وتعليم العربية للناطقين بغيرها. "
            "راجع النص مراجعة لغوية دقيقة قبل إعادته. تحقق من السلامة النحوية والصرفية والإملائية "
            "وعلامات الترقيم، ومطابقة الفعل والفاعل، والمذكر والمؤنث، والمفرد والمثنى والجمع، "
            "وحالات الرفع والنصب والجر عند الحاجة، والهمزات، والتاء المربوطة والهاء، والألف المقصورة والياء. "
            "تحقق من طبيعية الأسلوب العربي، وتجنب الترجمة الحرفية من الفرنسية أو الإنجليزية، ودقة المصطلحات "
            "التربوية وملاءمة المفردات والتراكيب للمستوى المحدد وفق CEFR. لا تغير الهدف التربوي ولا تحذف "
            "المعلومات المفيدة ولا تضف محتوى جديدًا، وحافظ على بنية النشاط قدر الإمكان. انتبه خصوصًا إلى "
            "التراكيب الصحيحة حرفيًا وغير الطبيعية: قد تصبح «جمل التبديل» «عبارات التعبير عن الرأي» بحسب "
            "السياق، وقد تصبح «الجمل المعلّمة» «العبارات المقترحة». أعد النسخة المصححة فقط دون شرح الأخطاء. "
            "You are an Arabic linguistic proofreader, not the author. Return the complete corrected answer only. "
            "Preserve its Markdown structure, headings, lists, tables, section order, pedagogical sequence, "
            "communicative task, factual grounding, CEFR level, response language, duration, and internal source markers exactly. "
            "Make the smallest possible edits. Do not enrich, summarize, reorganize, add activities, add exercises, "
            "add scenarios, change sources, or translate the whole response. Perform one complete final linguistic pass "
            "over the entire answer before returning it: check subject/verb, nominal, gender and singular/plural agreement; "
            "pronouns, possessives, prepositions and normal written Arabic forms; then check that local participant roles "
            "are coherent (teacher, learner, partner, group and class). Correct only Arabic grammar, morphology, agreement, "
            "possessives, pronoun/reference consistency, natural verb/preposition usage, idiomatic Modern Standard Arabic, "
            "pedagogical terminology, and directly related local scenario coherence. In particular, correct an obvious local "
            "role inversion when an instruction incorrectly makes the learner give the teacher's instruction, without redesigning "
            "the activity. Preserve short A1 scaffolding; do not simplify B1 reasoning into A1. For A1, locally prefer short, "
            "familiar, direct utterances and questions over an unnecessary comparative or other advanced construction; do not "
            "replan the activity. Use المتعلم/المتعلمون for generic Arabic pedagogy, التلميذ/التلاميذ only for explicit school-pupil "
            "contexts, and الطالب/الطلاب only when a genuine student context requires it. Before returning, determine the "
            "pedagogical context once and perform a global full-document terminology sweep: use the selected learner term "
            "consistently in every heading, table, instruction, criterion, teacher note, conclusion and optional extension; "
            "do not leave isolated generic طالب/طلاب in a CEFR context using المتعلم/المتعلمون. Review the answer globally, "
            "not only sentence by sentence: keep possessive family vocabulary coherent when the text is about one's own family, "
            "and check relations across sections, tables and bullet points. Perform a global grammar sweep of every section, "
            "including tables and lists, for agreement, active/passive misuse, role coherence and singular/plural consistency. "
            "Perform a final lexical-semantic precision sweep so that labels and instructions refer naturally to their intended "
            "pedagogical object; correct obvious local wording without turning this into a redesigned activity. "
            "In French generic pedagogy prefer apprenant(s), reserve élève(s) for explicit school contexts and étudiant(s) for "
            "higher education or adult students. When the resolved response language is Arabic, keep all teacher-facing prose, "
            "headings, labels, table cells, instructions and assessment wording in natural Arabic: remove unnecessary English "
            "or French glosses unless the user explicitly requested bilingual content. For A1, make this simplicity check over "
            "the whole answer, including optional extensions. Correct obvious local structural inconsistencies such as skipped "
            "step numbering, duplicated labels or conflicting singular/plural labels, but never reorder phases or add sections. "
            "Do not force diacritics. Regression "
            "guidance: prefer natural forms such as أبي طبيب or أبي يعمل طبيبًا over أبي يعمل طبيب; جدتي over جديّة when "
            "the intended meaning is grandmother; a natural kinship question such as ما صلة القرابة؟ over ما هو صلة العائلة؟; "
            "المتعلم and نصًا in generic pedagogy where appropriate; مدة تقريبية; and يُصغي rather than يُصغى for an active "
            "teacher action. Before returning, use this complete final checklist across the entire answer: grammar and natural MSA; "
            "one consistent learner term; coherent teacher/learner roles and possessives; no unnecessary foreign-language leakage; "
            "CEFR-appropriate complexity; checked tables, bullets, numbering and labels; and no omitted useful material or altered "
            "protected markers. Return only the complete corrected "
            "answer, with no explanation, change log, reviewer comments or reasoning. Do not use these as blind substitutions: "
            "preserve meaning and local context."
        )
        user = (
            f"Resolved response language: {response_language}\n"
            f"Resolved CEFR level (preserve, do not reinterpret): {cefr_level or 'unspecified'}\n"
            f"Current request context (preserve): {request_context}\n\n"
            "ANSWER TO PROOFREAD:\n"
        )
        return system, user

    def review(
        self,
        *,
        content: str,
        response_language: str,
        cefr_level: str | None,
        request_context: str,
    ) -> ArabicLinguisticReviewResult:
        if not self.needs_review(content):
            return ArabicLinguisticReviewResult(content, False, False, False, "no_arabic_content", 0)
        started = perf_counter()
        system_prompt, user_prefix = self._review_prompt(
            response_language=response_language, cefr_level=cefr_level, request_context=request_context,
        )
        reviewed, primary_reason, primary_status, primary_attempts = self._review_once(
            provider=self.llm, provider_name="primary", content=content,
            system_prompt=system_prompt, user_prefix=user_prefix, retry_policy=self.retry_policy,
        )
        if reviewed is not None:
            return self._success(
                reviewed, provider="primary", elapsed_ms=int((perf_counter() - started) * 1000),
                attempt_count=primary_attempts, primary_status="success", fallback_used=False,
            )
        if self.fallback_llm is not None:
            fallback_reviewed, fallback_reason, _fallback_status, fallback_attempts = self._review_once(
                provider=self.fallback_llm, provider_name="fallback", content=content,
                system_prompt=system_prompt, user_prefix=user_prefix, retry_policy=None,
            )
            elapsed_ms = int((perf_counter() - started) * 1000)
            if fallback_reviewed is not None:
                return self._success(
                    fallback_reviewed, provider="fallback", elapsed_ms=elapsed_ms,
                    attempt_count=primary_attempts + fallback_attempts, primary_status=primary_status,
                    fallback_used=True, reason=f"primary_{primary_reason}",
                )
            return self._fallback(
                content, "all_review_providers_failed", elapsed_ms, attempt_count=primary_attempts + fallback_attempts,
                primary_status=primary_status, fallback_used=True,
            )
        return self._fallback(
            content, primary_reason, int((perf_counter() - started) * 1000), attempt_count=primary_attempts,
            primary_status=primary_status, fallback_used=False,
        )

    def _review_once(
        self, *, provider: LLMProvider, provider_name: str, content: str, system_prompt: str,
        user_prefix: str, retry_policy: LLMRetryPolicy | None,
    ) -> tuple[str | None, str, int | str | None, int]:
        attempts: list[LLMRetryDiagnostic] = []

        def diagnostic(item: LLMRetryDiagnostic) -> None:
            attempts.append(item)
            self._provider_diagnostic(item, provider_name)

        try:
            result = provider.generate(
                system_prompt=system_prompt, user_prompt=user_prefix + content, temperature=0,
                max_tokens=self.max_tokens,
                retry_policy=replace(retry_policy, diagnostic_callback=diagnostic) if retry_policy else None,
                generation_options=self.generation_options,
            )
            reason = self._output_failure_reason(result.text, result.finish_reason, content)
            if reason:
                return None, reason, "invalid_output", max(1, self._attempt_count(attempts))
            return result.text.strip(), "applied", "success", max(1, self._attempt_count(attempts))
        except (LLMProviderError, TimeoutError, OSError, ValueError) as exc:
            status = exc.status_code if isinstance(exc, LLMProviderError) else None
            return None, self._failure_reason(exc), status, max(1, self._attempt_count(attempts))
        except Exception:
            return None, "review_failure", None, max(1, self._attempt_count(attempts))

    @staticmethod
    def _attempt_count(diagnostics: list[LLMRetryDiagnostic]) -> int:
        return sum(item.event == "attempt" for item in diagnostics)

    def _output_failure_reason(self, reviewed: object, finish_reason: str | None, content: str) -> str | None:
        if (finish_reason or "").casefold() in {"length", "max_tokens"}:
            return "truncated_review"
        if not isinstance(reviewed, str) or not reviewed.strip():
            return "empty_output"
        if len(reviewed.strip()) < max(8, int(len(content.strip()) * 0.35)):
            return "output_too_short"
        if self._markers(reviewed) != self._markers(content):
            return "internal_markers_changed"
        return None

    @staticmethod
    def _success(
        content: str, *, provider: str, elapsed_ms: int, attempt_count: int,
        primary_status: int | str | None, fallback_used: bool, reason: str = "applied",
    ) -> ArabicLinguisticReviewResult:
        logger.info(
            "arabic_linguistic_review arabic_review_attempted=true arabic_review_applied=true "
            "arabic_review_provider=%s arabic_review_attempt_count=%s arabic_review_primary_status=%s "
            "arabic_review_fallback_used=%s arabic_review_failure_reason=%s arabic_review_elapsed_ms=%s",
            provider, attempt_count, primary_status, str(fallback_used).lower(), reason, elapsed_ms,
        )
        return ArabicLinguisticReviewResult(
            content, True, True, False, reason, elapsed_ms, provider, attempt_count, primary_status, fallback_used,
        )

    @staticmethod
    def _provider_diagnostic(diagnostic: LLMRetryDiagnostic, provider: str) -> None:
        if diagnostic.event == "attempt":
            logger.info(
                "arabic_review_provider_attempt provider=%s attempt=%s/%s",
                provider, diagnostic.attempt,
                diagnostic.max_attempts,
            )
            return
        if diagnostic.event == "failure":
            if diagnostic.status_code is not None and 200 <= diagnostic.status_code < 300:
                logger.warning(
                    "arabic_review_provider_response_failure provider=%s status=%s stage=%s choices_count=%s content_present=%s content_type=%s finish_reason=%s exception=%s",
                    provider, diagnostic.status_code,
                    diagnostic.provider_error_type,
                    diagnostic.choices_count,
                    diagnostic.content_present,
                    diagnostic.content_type,
                    diagnostic.finish_reason,
                    diagnostic.exception_class,
                )
                return
            logger.warning(
                "arabic_review_provider_failure provider=%s status=%s exception=%s category=%s retryable=%s attempt=%s/%s",
                provider, diagnostic.status_code,
                diagnostic.exception_class,
                diagnostic.category,
                diagnostic.retryable,
                diagnostic.attempt,
                diagnostic.max_attempts,
            )

    @staticmethod
    def _failure_reason(exc: Exception) -> str:
        if isinstance(exc, LLMProviderError) and exc.status_code == 429:
            return "rate_limit_exhausted"
        if isinstance(exc, LLMProviderError) and exc.category in {
            "connect_timeout", "read_timeout", "write_timeout", "pool_timeout", "timeout",
        }:
            return "timeout"
        if isinstance(exc, LLMProviderError) and exc.category in {
            "json_decode_failure", "missing_choices", "missing_message", "missing_content",
            "empty_content", "invalid_content_type", "malformed_response", "truncated_review",
        }:
            return exc.category
        if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "__cause__", None), TimeoutError):
            return "timeout"
        if isinstance(exc, LLMProviderError):
            return "provider_failure"
        if isinstance(exc, OSError):
            return "transport_failure"
        return "review_failure"

    @staticmethod
    def _fallback(
        content: str, reason: str, elapsed_ms: int, *, attempt_count: int = 0,
        primary_status: int | str | None = None, fallback_used: bool = False,
    ) -> ArabicLinguisticReviewResult:
        logger.info(
            "arabic_linguistic_review arabic_review_attempted=true arabic_review_applied=false "
            "arabic_review_provider=none arabic_review_attempt_count=%s arabic_review_primary_status=%s "
            "arabic_review_fallback_used=%s arabic_review_failure_reason=%s arabic_review_elapsed_ms=%s",
            attempt_count, primary_status, str(fallback_used).lower(), reason, elapsed_ms,
        )
        return ArabicLinguisticReviewResult(
            content, True, False, True, reason, elapsed_ms, None, attempt_count, primary_status, fallback_used,
        )
