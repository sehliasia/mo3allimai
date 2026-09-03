"""One-call, validated course (contenu pédagogique) generation using the
existing RAG boundary. A course teaches real structured content (introduction,
vocabulary, expressions, grammar, content, dialogue, comprehension, guided
practice, communicative practice, production, summary, homework) following the
pedagogical progression Découvrir → Comprendre → Observer → Apprendre →
Pratiquer → Réutiliser → Produire → Bilan. It is deliberately distinct from a
lesson plan (how to run a session) and from an activity (one communicative
task)."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable
from pydantic import ValidationError
from app.core.config import Settings, get_settings
from app.schemas.course_generator import CourseGenerateIn, CourseOut
from app.services.llm_provider import LLMGenerationOptions, LLMProvider, LLMProviderError, LLMRetryPolicy
from app.services.pedagogical_knowledge_service import PedagogicalContext

logger = logging.getLogger(__name__)

# Controlled provider rate-limit handling (HTTP 429 / TPM). Retries are bounded
# and wait on the delay recommended by the provider, never immediately.
_MAX_PROVIDER_RETRIES = 1  # exactly one controlled retry on a 429
_RATE_LIMIT_RETRY_MARGIN_SECONDS = 1.0  # safety margin added to the wait
_RATE_LIMIT_FALLBACK_DELAY_SECONDS = 2.0  # used when no delay is available
_MAX_RATE_LIMIT_TOTAL_WAIT_SECONDS = 30.0  # absolute cap; never wait minutes

# Disables the provider's internal retry loop so the course service owns the
# single, controlled rate-limit retry with its own logging and margin.
_NO_PROVIDER_RETRY = LLMRetryPolicy(
    max_retries=0, max_wait_seconds=0.0, retry_base_delay=0.0, retry_max_delay=0.0,
)

_RATE_LIMIT_USER_MESSAGE = (
    "Le service IA est temporairement très sollicité. "
    "Veuillez patienter quelques secondes puis réessayer."
)


class CourseGenerationError(RuntimeError):
    pass


class CourseRateLimitError(CourseGenerationError):
    """Raised when the provider rate limit is still exhausted after the
    controlled retries. Distinct from a course validation error so the API can
    return a clear, retryable message to the UI."""


class CourseGenerationService:
    def __init__(self, *, llm: LLMProvider, settings: Settings | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self._sleep = sleep

    @staticmethod
    @staticmethod
    def _scan_course_objects(text: str) -> list[dict]:
        """Scan a LLM response for every complete top-level JSON object.

        Uses json.JSONDecoder.raw_decode so it never rejects a response that
        contains several objects or is surrounded by prose/reasoning fragments.
        Returns the full list of dict candidates; the caller scores them."""
        decoder = json.JSONDecoder()
        objects: list[dict] = []
        start = text.find("{")
        while start >= 0:
            try:
                value, _ = decoder.raw_decode(text, start)
                if isinstance(value, dict):
                    objects.append(value)
            except json.JSONDecodeError:
                pass
            start = text.find("{", start + 1)
        return objects

    # Strong CourseOut root signature: only objects exposing BOTH the required
    # root identity fields (level + theme) can be treated as a course root. This
    # prevents a nested grammar/content section {title, body, examples} from ever
    # being mistaken for the course.
    _COURSE_ROOT_SIGNATURE = {"level", "theme"}
    _COURSE_ROOT_WEIGHTS = {
        "title": 2, "level": 3, "theme": 3, "duration": 2,
        "objectives": 3, "skills": 2, "vocabulary": 2, "introduction": 2,
        "content": 3, "grammar": 2, "dialogue": 2, "comprehension": 2,
        "guided_practice": 2, "communicative_practice": 2, "production": 2,
        "summary": 2, "expressions": 1, "homework": 1,
    }

    @classmethod
    def _course_root_score(cls, obj: object) -> int:
        """Score a candidate by its CourseOut signature. Returns 0 when the
        object is not a plausible course root (missing level/theme, i.e. likely
        a nested section)."""
        if not isinstance(obj, dict):
            return 0
        if not cls._COURSE_ROOT_SIGNATURE.issubset(obj):
            return 0
        return sum(weight for key, weight in cls._COURSE_ROOT_WEIGHTS.items() if key in obj)

    @classmethod
    def _select_course_root(cls, candidates: list[dict]) -> dict | None:
        """Return the candidate with the strongest CourseOut signature.

        A real course root (contains level+theme, i.e. a non-zero root score)
        always wins, so a nested grammar/content section is never chosen over a
        genuine course. Only when NO candidate has the root signature do we fall
        back to the best-scoring object so Pydantic still validates/rejects a
        malformed response through its normal path."""
        best_root: dict | None = None
        best_root_score = 0
        best_any: dict | None = None
        best_any_score = -1
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            root_score = cls._course_root_score(candidate)
            if root_score > best_root_score:
                best_root = candidate
                best_root_score = root_score
            any_score = sum(weight for key, weight in cls._COURSE_ROOT_WEIGHTS.items() if key in candidate) if candidate else 0
            if candidate and any_score > best_any_score:
                best_any = candidate
                best_any_score = any_score
        return best_root if best_root is not None else best_any

    @classmethod
    def _debug_log_selection(cls, candidates: list[dict], best: dict) -> None:
        logger.debug(
            "course_json_candidates=%d selected_course_candidate=root selected_keys=%s",
            len(candidates), sorted(best.keys()),
        )

    @classmethod
    def _json_object(cls, raw: str) -> dict:
        """Extract the single course root JSON object from a LLM response.

        Always scores every candidate against the CourseOut signature (never
        trusts a raw whole-text json.loads, which would happily return a nested
        grammar section when it is the first/only JSON object)."""
        text = raw.lstrip("\ufeff \t\r\n")
        candidates = cls._scan_course_objects(text)
        best = cls._select_course_root(candidates)
        if best is not None:
            cls._debug_log_selection(candidates, best)
            return best
        raise CourseGenerationError("Le JSON du cours est incomplet ou invalide.")

    @staticmethod
    def _to_string(value: object) -> object:
        """Join a list of strings into a single string (information-preserving),
        for fields the schema expects as a single string."""
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return "\n".join(value)
        return value

    @staticmethod
    def _to_list(value: object) -> object:
        """Coerce a single string into a one-element list for list-typed fields."""
        if isinstance(value, str):
            return [value]
        return value

    @staticmethod
    def _coerce_examples(value: object) -> list[dict]:
        """Normalize example payloads (string or {title, body}) into CourseExample dicts."""
        if isinstance(value, str):
            return [{"title": "Exemple", "body": value}]
        if not isinstance(value, list):
            return []
        normalized: list[dict] = []
        for item in value:
            if isinstance(item, str):
                normalized.append({"title": "Exemple", "body": item})
            elif isinstance(item, dict):
                normalized.append({"title": item.get("title", "Exemple"), "body": item.get("body", item.get("example", ""))})
            else:
                normalized.append(item)
        return normalized

    @staticmethod
    def _normalize_payload(payload: dict) -> dict:
        """Normalize only known LLM type slips and legacy shapes before strict
        Pydantic validation."""
        normalized = payload.copy()

        def duration_as_int(value: object) -> object:
            if isinstance(value, str):
                match = re.fullmatch(r"\s*(\d+)\s+minutes?\s*", value, flags=re.IGNORECASE)
                if match:
                    return int(match.group(1))
            return value

        normalized["duration"] = duration_as_int(normalized.get("duration"))

        for field in ("vocabulary", "objectives", "skills", "expressions"):
            if normalized.get(field) is not None:
                normalized[field] = CourseGenerationService._to_list(normalized[field])

        summary = normalized.get("summary")
        if summary is None:
            normalized["summary"] = []
        elif isinstance(summary, str):
            normalized["summary"] = [line.strip() for line in summary.splitlines()] or [summary]
        else:
            normalized["summary"] = summary

        dialogue = normalized.get("dialogue")
        if isinstance(dialogue, str):
            lines = [line.strip() for line in re.split(r"\n| / ", dialogue) if line.strip()]
            normalized["dialogue"] = {"context": "", "lines": lines}
        elif isinstance(dialogue, dict):
            lines = dialogue.get("lines")
            if isinstance(lines, str):
                lines = [lines]
            normalized["dialogue"] = {
                "context": dialogue.get("context", ""),
                "lines": lines if isinstance(lines, list) else [],
            }

        for section_list, is_grammar in (("grammar", True), ("content", False)):
            items = normalized.get(section_list)
            if not isinstance(items, list):
                continue
            normalized[section_list] = []
            for item in items:
                if isinstance(item, str):
                    normalized[section_list].append({"title": "", "body": item, "examples": []})
                elif isinstance(item, dict):
                    if is_grammar:
                        normalized[section_list].append({
                            "title": item.get("topic") or item.get("title", ""),
                            "body": item.get("explanation") or item.get("body", ""),
                            "examples": CourseGenerationService._coerce_examples(item.get("example")),
                        })
                    else:
                        normalized[section_list].append({
                            "title": item.get("title", ""),
                            "body": item.get("body", ""),
                            "examples": CourseGenerationService._coerce_examples(item.get("examples")),
                        })
                else:
                    normalized[section_list].append(item)

        for exercise_list in ("comprehension", "guided_practice", "communicative_practice", "production"):
            items = normalized.get(exercise_list)
            if not isinstance(items, list):
                continue
            normalized[exercise_list] = []
            for item in items:
                if isinstance(item, str):
                    normalized[exercise_list].append({"title": item, "instructions": "", "example": None})
                elif isinstance(item, dict):
                    example = item.get("example")
                    if isinstance(example, str):
                        example = {"title": "Exemple", "body": example}
                    normalized[exercise_list].append({
                        "title": item.get("title", ""),
                        "instructions": item.get("instructions", ""),
                        "example": example,
                    })
                else:
                    normalized[exercise_list].append(item)

        return normalized

    @staticmethod
    def _normalized(value: str) -> str:
        import unicodedata
        return " ".join(
            "".join(
                character for character in unicodedata.normalize("NFD", value.casefold())
                if unicodedata.category(character) != "Mn"
            ).split()
        )

    def _validate_pedagogical_consistency(self, course: CourseOut, request: CourseGenerateIn) -> None:
        """Reject only objective output defects that a JSON schema cannot express."""
        if course.level != request.level:
            raise CourseGenerationError(
                f"Le niveau annoncé ({course.level}) doit correspondre au niveau demandé ({request.level})."
            )
        if course.duration != request.duration_minutes:
            raise CourseGenerationError(
                f"La durée annoncée ({course.duration} min) doit correspondre à la durée demandée "
                f"({request.duration_minutes} min)."
            )
        if not course.title.strip():
            raise CourseGenerationError("Le cours généré doit avoir un titre.")

        theme_norm = self._normalized(request.theme)
        course_theme_norm = self._normalized(course.theme)
        if not theme_norm or not course_theme_norm or not (
            theme_norm == course_theme_norm
            or (len(theme_norm) >= 3 and theme_norm in course_theme_norm)
            or (len(course_theme_norm) >= 3 and course_theme_norm in theme_norm)
        ):
            raise CourseGenerationError(
                f"Le cours doit rester centré sur le thème demandé ({request.theme})."
            )

        if not course.objectives or any(not item.strip() for item in course.objectives):
            raise CourseGenerationError("Le cours généré doit indiquer au moins un objectif.")
        if not course.skills or any(not item.strip() for item in course.skills):
            raise CourseGenerationError("Le cours généré doit indiquer au moins une compétence.")
        if not course.vocabulary or any(not item.strip() for item in course.vocabulary):
            raise CourseGenerationError("Le cours généré doit présenter un vocabulaire non vide.")
        if not course.introduction.strip():
            raise CourseGenerationError("Le cours généré doit présenter une introduction.")
        if not course.content or any(not item.title.strip() or not item.body.strip() for item in course.content):
            raise CourseGenerationError("Le cours généré doit contenir du contenu pédagogique.")
        for section in course.grammar:
            if not section.title.strip() or not section.body.strip():
                raise CourseGenerationError("Chaque point grammatical doit avoir un intitulé et une explication.")
        if not course.summary or any(not item.strip() for item in course.summary):
            raise CourseGenerationError("Le cours généré doit proposer une synthèse.")

        self._validate_exercises(course.comprehension, "compréhension")
        self._validate_exercises(course.guided_practice, "pratique guidée")
        self._validate_exercises(course.communicative_practice, "pratique communicative")
        self._validate_exercises(course.production, "production")

        if self._has_duplicated_exercises(
            course.comprehension, course.guided_practice,
            course.communicative_practice, course.production,
        ):
            raise CourseGenerationError(
                "Plusieurs sections contiennent la même consigne ou le même exemple : veuillez différencier les exercices."
            )

        if course.dialogue is not None and (
            not course.dialogue.lines or any(not line.strip() for line in course.dialogue.lines)
        ):
            raise CourseGenerationError("Le dialogue généré doit contenir des répliques.")

        if request.level in {"A1", "A2", "B1"} and self._demands_complex_sentences(
            *course.summary,
            *(section.body for section in course.grammar),
            *(section.body for section in course.content),
            *(item.instructions for section in (
                course.comprehension, course.guided_practice, course.communicative_practice, course.production,
            ) for item in section),
        ):
            raise CourseGenerationError(
                "Le cours exige à tort des phrases complexes pour ce niveau CECRL. Précisez un contenu simple, clair et accessible."
            )

    @staticmethod
    def _has_duplicated_exercises(*groups: list) -> bool:
        """Detect verbatim content duplication across practice/production
        sections (requirement: no duplicated text between sections). Only an
        identical (title + instructions) pair is treated as duplication, so
        similarly-titled but differently-worded exercises are not rejected."""
        seen: set[str] = set()
        for group in groups:
            for item in group:
                signature = (
                    CourseGenerationService._normalized(item.title)
                    + "||"
                    + CourseGenerationService._normalized(item.instructions)
                )
                if not CourseGenerationService._normalized(item.title):
                    continue
                if signature in seen:
                    return True
                seen.add(signature)
        return False

    def _validate_exercises(self, exercises: list, label: str) -> None:
        """A structure check only: exercises carry a title; optional examples,
        when present, must have a body (objectives, practice and production are
        not forced to be present for every level)."""
        for exercise in exercises:
            if not exercise.title.strip():
                raise CourseGenerationError(f"Chaque exercice de {label} doit avoir un intitulé.")
            if exercise.example is not None and not exercise.example.body.strip():
                raise CourseGenerationError(
                    f"L'exemple de l'exercice « {exercise.title} » doit avoir un contenu."
                )

    @staticmethod
    def _demands_complex_sentences(*values: str) -> bool:
        """Detect LLM slippage demanding 'complex sentences' at low/mid CEFR levels."""
        forbidden = {
            CourseGenerationService._normalized(pattern)
            for pattern in ("جمل معقدة", "جمل صعبة", "عبارات معقدة", "phrases complexes", "جمل معقدة ومعقدة")
        }
        return any(
            any(forbidden_item in CourseGenerationService._normalized(value) for forbidden_item in forbidden)
            for value in values
            if value
        )

    _REQUIRED_KEYS = [
        "title", "level", "theme", "duration",
        "objectives", "skills", "vocabulary", "expressions",
        "introduction", "grammar", "content", "dialogue",
        "comprehension", "guided_practice", "communicative_practice", "production",
        "summary", "homework",
    ]

    @staticmethod
    def _is_incomplete_course(payload: object, errors: list[dict]) -> bool:
        """A syntactically valid dict that is clearly a course (title/level/theme
        present) but missing required fields entirely (no wrong-type errors)."""
        if not isinstance(payload, dict):
            return False
        if not {"title", "level", "theme"}.issubset(payload):
            return False
        if not errors:
            return False
        return all(error.get("type") == "missing" for error in errors)

    @staticmethod
    def _missing_keys(errors: list[dict]) -> list[str]:
        return [error["loc"][0] for error in errors if error.get("type") == "missing" and error.get("loc")]

    def _build_retry_prompt(self, missing_keys: list[str]) -> str:
        listed = missing_keys or self._REQUIRED_KEYS
        return (
            "La réponse précédente est incomplète.\n"
            "Retourne à nouveau le même cours sous forme d'un objet JSON COMPLET respectant exactement le schéma demandé.\n"
            "Toutes les clés obligatoires suivantes doivent être présentes : " + ", ".join(listed) + ".\n"
            "Ne change pas inutilement le contenu déjà produit. Ajoute les clés manquantes. N'ajoute aucun texte hors JSON."
        )

    def _generate_once(self, system: str, user: str):
        attempts = 0
        total_waited = 0.0
        while True:
            started = time.perf_counter()
            try:
                result = self.llm.generate(
                    system_prompt=system, user_prompt=user, temperature=0.2,
                    max_tokens=self.settings.course_max_output_tokens,
                    retry_policy=_NO_PROVIDER_RETRY,
                    generation_options=LLMGenerationOptions(reasoning_effort="low", include_reasoning=False),
                )
            except LLMProviderError as exc:
                duration_seconds = time.perf_counter() - started
                if exc.status_code != 429:
                    self._log_provider_request_failed(exc, model_id=self.llm.model_id, duration_seconds=duration_seconds)
                    raise CourseGenerationError(exc.provider_message) from exc
                delay = self._compute_retry_delay(exc)
                retry_after = self._retry_after_from(exc)
                self._log_rate_limit(self.llm.model_id, exc, retry_after)
                if attempts >= _MAX_PROVIDER_RETRIES:
                    logger.warning(
                        "[LLM_RATE_LIMIT_EXHAUSTED] provider=%s attempts=%s total_waited=%.2fs",
                        self.llm.model_id, attempts, total_waited,
                    )
                    raise CourseRateLimitError(_RATE_LIMIT_USER_MESSAGE) from exc
                if total_waited + delay > _MAX_RATE_LIMIT_TOTAL_WAIT_SECONDS:
                    logger.warning(
                        "[LLM_RATE_LIMIT_EXHAUSTED] provider=%s attempts=%s cap_hit=True total_waited=%.2fs",
                        self.llm.model_id, attempts, total_waited,
                    )
                    raise CourseRateLimitError(_RATE_LIMIT_USER_MESSAGE) from exc
                self._sleep(delay)
                total_waited += delay
                attempts += 1
                logger.warning("[LLM_RATE_LIMIT_RETRY] attempt=%s delay=%.2fs", attempts, delay)
                continue
            if (result.finish_reason or "").casefold() in {"length", "max_tokens", "max_output_tokens"}:
                raise CourseGenerationError("La génération a été tronquée. Réduisez les consignes puis réessayez.")
            return result

    @staticmethod
    def _log_provider_request_failed(exc: LLMProviderError, *, model_id: str, duration_seconds: float) -> None:
        """Log the ORIGINAL provider failure (never secrets, never the API key)
        so the root cause of a 503 is traceable while the user message stays clean."""
        cause = exc.__cause__ if isinstance(exc.__cause__, Exception) else exc
        category = exc.category or type(cause).__name__ or "provider_error"
        metadata = exc.response_metadata or {}
        error_type = metadata.get("error_type") or category
        status_code = metadata.get("status_code") if metadata.get("status_code") is not None else exc.status_code
        logger.error(
            "provider_request_failed provider=%s model=%s error_type=%s status_code=%s "
            "error_message=%s duration_seconds=%.3f",
            "groq",
            model_id,
            error_type,
            status_code if status_code is not None else "none",
            exc.provider_message or "none",
            duration_seconds,
        )

    @staticmethod
    def _log_rate_limit(model: str, exc: LLMProviderError, retry_after: float | None) -> None:
        metadata = exc.response_metadata or {}
        used = metadata.get("usage_used_tokens") or metadata.get("used_tokens")
        requested = metadata.get("usage_requested_tokens") or metadata.get("requested_tokens")
        limit = metadata.get("usage_limit") or metadata.get("tpm_limit")
        message = exc.provider_message or ""
        if used is None:
            used = CourseGenerationService._extract_token_value(message, "used")
        if requested is None:
            requested = CourseGenerationService._extract_token_value(message, "requested")
        if limit is None:
            limit = CourseGenerationService._extract_token_value(message, ("limit", "tpm"))
        logger.warning(
            "[LLM_RATE_LIMIT] model=%s used_tokens=%s requested_tokens=%s limit=%s retry_after=%s",
            model, used, requested, limit,
            f"{retry_after:.2f}" if retry_after is not None else "none",
        )

    @staticmethod
    def _extract_token_value(message: str, labels: str | tuple[str, ...]) -> int | None:
        """Parse 'Used 6206', 'Requested 2990', 'Limit 8000' / 'TPM limit 8000'
        patterns from a provider error message."""
        if not message:
            return None
        if isinstance(labels, str):
            labels = (labels,)
        for label in labels:
            match = re.search(rf"(?i)\b{re.escape(label)}\b\s*[:=]?\s*(\d+)", message)
            if match:
                try:
                    return int(match.group(1))
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _retry_after_from(exc: LLMProviderError) -> float | None:
        metadata = exc.response_metadata or {}
        value = metadata.get("retry_after_seconds") or metadata.get("retry_after")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
        return None

    @staticmethod
    def _compute_retry_delay(exc: LLMProviderError) -> float:
        """Delay before retrying a 429, with safety margin.

        Priority: 1) Retry-After header (via response metadata), 2) a reliable
        delay embedded in the provider message, 3) a bounded fallback.
        """
        retry_after = CourseGenerationService._retry_after_from(exc)
        if retry_after is not None:
            return retry_after + _RATE_LIMIT_RETRY_MARGIN_SECONDS
        message_delay = CourseGenerationService._extract_retry_delay(exc.provider_message)
        if message_delay is not None:
            return message_delay + _RATE_LIMIT_RETRY_MARGIN_SECONDS
        return _RATE_LIMIT_FALLBACK_DELAY_SECONDS + _RATE_LIMIT_RETRY_MARGIN_SECONDS

    @staticmethod
    def _extract_retry_delay(message: str | None) -> float | None:
        """Extract a provider-recommended wait from the error message (e.g. Groq's
        'Please try again in 8.97s'). Returns None when no reliable delay is found."""
        if not message:
            return None
        for pattern in (r"try again in\s+(\d+(?:\.\d+)?)", r"in\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b"):
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                seconds = float(match.group(1))
                return seconds if seconds >= 0 else None
            except ValueError:
                continue
        return None

    def generate(self, request: CourseGenerateIn, context: PedagogicalContext) -> CourseOut:
        sources = [
            {"title": block.document_title, "pages": [block.page_start, block.page_end], "content": block.content[:700]}
            for block in context.resource_blocks[:2]
        ]
        system = self._build_system_prompt()
        user = json.dumps({"request": request.model_dump(), "rag_resources": sources}, ensure_ascii=False)
        result = self._generate_once(system, user)
        payload = self._json_object(result.text)
        try:
            course = CourseOut.model_validate(self._normalize_payload(payload))
        except ValidationError as exc:
            errors = exc.errors(include_url=False)
            if self._is_incomplete_course(payload, errors):
                logger.warning(
                    "course_incomplete provider=%s missing=%s -> retry", result.model, self._missing_keys(errors),
                )
                retry_user = self._build_retry_prompt(self._missing_keys(errors))
                result = self._generate_once(system, retry_user)
                payload = self._json_object(result.text)
                try:
                    course = CourseOut.model_validate(self._normalize_payload(payload))
                except ValidationError as retry_exc:
                    logger.warning(
                        "course_incomplete_retry_failed provider=%s errors=%s raw_response=%r",
                        result.model, retry_exc.errors(include_url=False), result.text,
                    )
                    raise CourseGenerationError("Le cours généré reste incomplet après une nouvelle tentative.") from retry_exc
            else:
                logger.warning(
                    "course_schema_validation_failed provider=%s errors=%s raw_response=%r",
                    result.model, errors, result.text,
                )
                raise CourseGenerationError("Le cours généré ne respecte pas le format attendu.") from exc
        self._validate_pedagogical_consistency(course, request)
        return course.model_copy(update={"rag_sources_used": len(sources), "provider_model": result.model})

    @staticmethod
    def _build_system_prompt() -> str:
        return """Tu es un expert en didactique de la langue arabe, en CECRL et en conception de contenus pédagogiques structurés. Tu conçois des cours qui répondent à « qu'est-ce que l'enseignant va enseigner et quel contenu l'apprenant doit apprendre ? » : un contenu pédagogique riche, réellement exploitable et prêt à l'usage, jamais un formulaire générique rempli par une IA.
Retourne uniquement un objet JSON valide, sans Markdown, sans bloc ```json et sans aucun texte avant ou après le JSON. Respecte exactement le schéma Pydantic existant.
Tu dois obligatoirement retourner un seul objet JSON complet. Toutes les clés suivantes sont obligatoires et doivent toujours être présentes : title, level, theme, duration, objectives, skills, vocabulary, expressions, introduction, grammar, content, dialogue, comprehension, guided_practice, communicative_practice, production, summary, homework. Ne supprime jamais une clé parce qu'elle semble vide ou non pertinente ; si une information est facultative, retourne null ou [] selon le schéma. Ne retourne jamais un JSON partiel ni du texte avant ou après le JSON.
Utilise exactement ces clés anglaises (ne les traduis pas) : title, level, theme, duration, objectives, skills, vocabulary, expressions, introduction, grammar, content, dialogue, comprehension, guided_practice, communicative_practice, production, summary, homework.
Formats attendus :
- objectives, skills, vocabulary, expressions, summary sont des tableaux JSON de chaînes. summary est TOUJOURS un tableau de 2 à 4 points de synthèse, jamais une chaîne unique.
- grammar et content sont des tableaux d'objets {title, body, examples} où examples est une liste d'objets {title, body} (une phrase ou réplique modèle). Pour grammar, title = intitulé du point, body = explication (accessible au niveau), examples = phrases modèles.
- dialogue est un objet {context, lines} : context = une courte mise en situation (peut être vide), lines = les répliques du dialogue, une chaîne par tour (ex. « الأب : من هذا؟ »). Ne retourne jamais une chaîne brute.
- comprehension, guided_practice, communicative_practice, production sont des tableaux d'objets exercices {title, instructions, example?}. title = consigne courte, instructions = brève explicitation pour l'apprenant, example = modèle de réponse {title, body} (optionnel, à fournir aux niveaux guidés).
- homework est une chaîne ou null. duration est un entier JSON, jamais « 60 minutes ».
Chaque exercice et chaque section du cours a un intitulé distinct : ne duplique jamais le même texte, la même consigne ou le même exemple de façon identique entre deux sections ; chaque section a un rôle spécifique dans la progression.
Gestion de l'objectif (paramètre « objectif » du request) :
- CAS A — une objectif non vide est fournie : elle est prioritaire et guide TOUT le contenu (cohérence objectif → contenu → pratique → production). Formule les objectives, le contenu, les pratiques et la production en accord avec cette objectif.
- CAS B — « objectif » est absence, null ou vide : définis toi-même un objectif pédagogique pertinent, précis et adapté au thème et au niveau CECRL (ex. « Apprendre à présenter les membres de sa famille »), puis utilise cet objectif auto-défini de manière cohérente sur tout le cours (objectives, contenu, pratiques, production) sans jamais l'afficher comme une consigne mais comme un vrai fil conducteur.
Ne laisse jamais l'objectif indéfini : en l'absence de valeur fournie, choisis-en toujours une cohérente avec le thème et le niveau.
Construis le cours selon la progression pédagogique Découvrir → Comprendre → Observer → Apprendre → Pratiquer → Réutiliser → Produire → Bilan :
- Découvrir : introduction = amorce courte et motivante qui met le thème et l'objectif en contexte.
- Comprendre : objectives, skills, vocabulary et expressions posent ce qui va être appris. Le vocabulaire découle du thème (cohérence thème → vocabulaire), l'objectif guide tout le contenu (cohérence objectif → contenu → pratique → production).
- Observer : grammar présente UN point grammatical expliqué simplement avec des examples ; dialogue est un modèle court à observer, suivi de questions de comprehension.
- Apprendre : content développe le contenu pédagogique : présentation, explications, textes courts, applications guidées, toujours en arabe.
- Pratiquer : guided_practice = exercices très guidés et progressifs ; communicative_practice = exercices semi-guidés ou libres en situation de communication, qui réutilisent le vocabulaire, les expressions et le point grammatical.
- Réutiliser : comprehension repart du dialogue ou du contenu pour vérifier la compréhension (questions de compréhension).
- Produire : production = une tâche de production orale ou écrite fidèle à l'objectif et au niveau.
- Bilan : summary récapitule l'essentiel appris en 2 à 4 points ; homework propose un prolongement utile et réaliste (révision, court exercice, préparation d'une courte production), jamais un devoir vague.
Adapte strictement la richesse, la longueur et la complexité au niveau CECRL (la durée détermine la quantité : 15 min très court, 30 min complet, 60 min et plus développé ; ne remplis jamais artificiellement) :
- A1 : phrases COURTES, simples et claires — événements d'aujourd'hui ; N'écris JAMAIS « جمل معقدة » ni « phrases complexes » et ne demande jamais de produire des phrases complexes. Vocabulaire très fréquent et concret (5 à 8 items), un point grammatical simple (ex. démonstratifs : هذا + masculin, هذه + féminin), un dialogue court (4 à 6 répliques), une forte guidance, des exercices guidés très simples, une production minimale (se présenter, nommer, dire un mot).
- A2 : situations quotidiennes, vocabulaire élargi (8 à 12 items), phrases simples mais plus variées, courts paragraphes descriptifs, dialogues un peu plus développés, exercices semi-guidés.
- B1 : interaction, narration simple, justification simple (parce que → لأن), connecteurs (لأن، لكن، ثم، بعد ذلك، لذلك، أولاً، أخيراً), vocabulaire contextualisé, textes et dialogues développés, exercices semi-guidés. Exige « جمل واضحة ومترابطة مع تقديم أسباب وتفاصيل مناسبة للمستوى », JAMAIS « جمل معقدة ».
- B2 : argumentation, comparaison, hypothèse, justification développée, nuances, vocabulaire plus riche, textes plus longs, structures grammaticales plus complexes, activités autonomes.
- C1 : nuances, registre, analyse et abstraction, lexique précis, autonomie importante.
- C2 : maîtrise avancée de la langue, complexité linguistique par la nuance et la précision du style — pas une simple quantité de texte.
Ne produis jamais un contenu plus complexe que le niveau demandé (ex. à A1 : pas de phrases complexes, pas de temps avancés inexpliqués) ni plus simple que le public : l'introduction, les consignes et les explications doivent rester accessibles au niveau tout en enseignant l'arabe.
Enseigne réellement l'arabe : le contenu doit être en arabe standard (الفصحى) naturel, correct et non traduit mécaniquement du français. Si request.language est ar, rédige en arabe le vocabulaire, les expressions, les exemples, le dialogue, le contenu, les questions et les consignes ; ne fournis une traduction française qu'en soutien. Si request.language est fr (ou en/es), rédige les explications en français et garde les exemples, le dialogue et le contenu d'apprentissage en arabe. Vérifie mentalement toute langue arabe avant la sortie : orthographe, grammaire, accords, genre, nombre, pronoms, démonstratifs, prépositions, conjugaison et formulation naturelle des questions. Ne mélange pas darija et fusha sans demande explicite. Utilise une terminologie cohérente.
Utilise réellement tous les paramètres de request : niveau CECRL, thème, objectif, compétences, durée, public/âge, nombre d'apprenants, langue et instructions supplémentaires. Ne laisse aucun paramètre pertinent sans effet. Les structures et mots arabes introduits doivent être réutilisés dans le contenu et les activités (cohérence thème → vocabulaire → exemples → pratiques), sinon le cours enseigne des mots qu'on n'emploie jamais.
Les sources RAG sont un appui à sélectionner, synthétiser et adapter lorsqu'elles sont pertinentes : ne copie pas aveuglément, n'invente aucune provenance ni ressource, et ne contredis jamais une source pertinente. Si aucune source pertinente n'est disponible, génère quand même un cours complet et de qualité à partir de ta propre expertise. Reste structuré, exploitable en classe et sans longues explications théoriques.
Avant de répondre, contrôle silencieusement : niveau CECRL respecté (complexité, lexique, longueur, nature du dialogue ; jamais « جمل معقدة » à A1/A2/B1), thème et objectif respectés, cohérence thème → vocabulaire, cohérence objectif → contenu → pratique → production, progression Découvrir → Bilan respectée, arabe correct et naturel, structures et mots introduits réutilisés, aucune duplication de texte entre sections, synthèse présente, prolongement utile et aucune contradiction. En cas de conflit, priorise la pertinence pédagogique, la cohérence, l'exactitude linguistique, l'adaptation au niveau et l'exploitabilité en classe."""